

from itertools import chain

import torch
import yaml
import numpy as np
from models.base_model import BaseModel

from models.cpc.ts.s4_modules.s4_model import S4Model
from models.cpc.ts.encoder import RNNEncoder, RNNEncoderConfig
from models.utils import get_normalization_layer

class CPCWrapper(BaseModel):
    def __init__(self, config, config_path=None, chunk_size=600,  feature_classification=False, sleep_apnea=False):
        super().__init__()
        self.config_path = config_path
        self.split_signal = config.split_signal
        self.linear_probing = config.linear_probing
        self.num_classes = config.num_classes
        self.config = config
        self.feature_classification = feature_classification
        self.sleep_apnea = sleep_apnea
        self.chunk_size = chunk_size
        self.patch_size = 2

        self.ts_encoder, self.config = self.load_model_from_config(
            config_path=self.config_path
        )

    def forward(self, x):
        bs = x.shape[0]
        x, n_chunks = self.chunk_signal_if_needed(x.transpose(1, 2))
        x = self.ts_encoder(x)
        x = self.unchunk_signal_if_needed(x, n_chunks, bs)

        return torch.nan_to_num(x)
    
    def chunk_signal_if_needed(self, x):
        B, C, L = x.shape
        # Split each input into durations of length self.chunk_size
        if self.split_signal:
            # in sleep apnea the chunks are every minute, so chunk_size is sampling_freq * 60                
            n_chunks = L // self.chunk_size
        
            if n_chunks == 0:
                raise ValueError(f"Signal length {L} is shorter than required segment length {self.chunk_size}")
            
            # Truncate to ensure divisibility
            x = x[..., :n_chunks * self.chunk_size]

            # Reshape: (B, C, n_chunks, chunk_size) -> (B, n_chunks, C, chunk_size) -> (B * n_chunks, C, chunk_size)
            x = x.view(B, C, n_chunks, self.chunk_size).permute(0, 2, 1, 3).reshape(-1, C, self.chunk_size)
        return (x, n_chunks) if self.split_signal else (x, None)
    
    def unchunk_signal_if_needed(self, x, n_chunks, batch_size):
        # Combine outputs from same original signal
        if self.split_signal:
            # Reshape back and average: (B, n_chunks, Output_Dim) -> (B, Output_Dim)
            x = x.view(batch_size, n_chunks, -1).mean(dim=1)
        return x

    def load_model_from_config(self, config_path):
        with open(config_path, "r") as fp:
            config = yaml.safe_load(fp)

        encoder_hparams = config["rnn_hyperparameters"]
        encoder_hparams["hparams_encoder"] = RNNEncoderConfig(**encoder_hparams["hparams_encoder"])
        s4_hparams = config["s4_hyperparamters"]
        cpc_hparams = config["cpc_hyperparameters"]
        cpc_hparams["eval_mode"] = "linear" if self.linear_probing else "finetuning"
        cpc_hparams['num_classes'] = self.num_classes
        cpc_hparams['sleep_apnea'] = self.sleep_apnea
        s4_hparams['pooling'] = not self.feature_classification and not self.sleep_apnea # only use pooling if not feature classification or if not sleep apnea (since in sleep apnea the signal is already 1 minute long)

        model = CPCModel(
            encoder_hparams=encoder_hparams,
            s4_hparams=s4_hparams,
            config=self.config,
            **cpc_hparams
        )
        
        return model, config
    
    def load_weights_from_checkpoint(self, checkpoint):
        """ Function that loads the weights from a given checkpoint file. 
        based on https://github.com/PyTorchLightning/pytorch-lightning/issues/525
        """
        checkpoint = torch.load(checkpoint, map_location=lambda storage, loc: storage,)
        if "state_dict" in checkpoint.keys():
            pretrained_dict = checkpoint["state_dict"]
        else:
            pretrained_dict = checkpoint
        model_dict = self.state_dict()
            
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)
    
    def load_state_dict(self, state_dict, strict=True):
        #S4-compatible load_state_dict
        for name, param in self.named_parameters():
            if name in state_dict:
                param.data = state_dict[name].data.to(param.device)
            elif strict:
                raise KeyError(f"Key {name} not found in state_dict")
        
        for name, param in self.named_buffers():
            if name in state_dict:
                param.data = state_dict[name].data.to(param.device)
            elif strict:
                raise KeyError(f"Buffer {name} not found in state_dict")
            
    def training_params(self):
        """
        Defines the parameters to be optimized during training. These parameters will receive the main learning rate ([config.lr_head]).
        """
        return self.ts_encoder.head.parameters()
    
    def finetuning_params(self):
        """
        Defines the parameters to be optimized during finetuning. These parameters will receive a smaller learning rate ([config.lr_core]).
        """
        params = list(self.ts_encoder.encoder.parameters()) + list(self.ts_encoder.predictor.parameters())
        return params
    
    def set_eval_linear_probing(self):
        self.eval()
        self.ts_encoder.head.train()

    def get_params_layerwise_decay(self, lr_decay, lr, wd):
        encoder_params = self.ts_encoder.encoder.parameters()
        predictor_params = self.ts_encoder.predictor.parameters()

        return [
            {"params": predictor_params, "lr": lr * lr_decay, "weight_decay": wd},
            {"params": encoder_params, "lr": lr * lr_decay * lr_decay, "weight_decay": wd}
        ]


class S4Wrapper(torch.nn.Module):
    """Just to match naming in the pretrained checkpoint."""
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.predictor = S4Model(*args, **kwargs)
    
    def forward(self, x):
        return self.predictor(x)


class CPCModel(torch.nn.Module):
    def __init__(self, encoder_hparams, config, s4_hparams, num_classes, feature_dim=512, eval_mode="finetuning", lr=1e-3, discriminative_lr_factor=0.1, sleep_apnea=False):
        super().__init__()
        self.encoder_hparams = encoder_hparams
        self.s4_hparams = s4_hparams
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.eval_mode = eval_mode
        self.sleep_apnea = sleep_apnea
        self.context_size = config.context_size
        self.sampling_freq = config.sampling_freq
        self.window_size = config.window_size
        self.patch_size = config.patch_size
        self.lr = lr
        self.discriminative_lr_factor = discriminative_lr_factor

        self.encoder = RNNEncoder(**self.encoder_hparams)
        self.predictor = S4Wrapper(**self.s4_hparams)

        self.head = torch.nn.Sequential(
            get_normalization_layer(config, self.feature_dim, permute_for_batchnorm=not s4_hparams['pooling']),
            torch.nn.Linear(self.feature_dim, num_classes)
        )
        
        if self.eval_mode == "linear":
            for p in self.encoder.parameters():
                p.requires_grad = False
            self.encoder.eval()

            for p in self.predictor.parameters():
                p.requires_grad = False
            self.predictor.eval()
    
    def get_features(self, x):
        x = self.encoder(x)
        x = self.predictor(x)
        return x

    def forward(self, x):
        features = self.get_features(x)
        # print(f'features before head pool', features.shape)

        if self.sleep_apnea:
            if self.context_size > 0:
                context_patches = (self.context_size * self.sampling_freq) // self.patch_size // 2
                window_patches = (self.window_size * self.sampling_freq) // self.patch_size
                start = context_patches
                end = context_patches + window_patches
                # print(f"Features shape before removing context patches: {features.shape}")
                # print(f"Removing context patches: start {start}, end {end}")
                features = features[:, start:end, :]
                # print(f"Features shape after removing context patches: {features.shape}")
            features = features.mean(dim=1) # average pool over time dimension

        out = self.head(features)
        return out # TODO: is torch.nan_to_num necessary?
