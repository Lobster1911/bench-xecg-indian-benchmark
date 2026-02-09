

from itertools import chain

import torch
import yaml
import numpy as np

from cpc.ts.s4_modules.s4_model import S4Model
from cpc.ts.encoder import RNNEncoder, RNNEncoderConfig


class CPCWrapper(torch.nn.Module):
    def __init__(self, config_path=None, chunk_size=600):
        super().__init__()
        self.config_path = config_path
        self.chunk_size = chunk_size

        self.ts_encoder, self.config = self.load_model_from_config(
            config_path=self.config_path
        )

    def forward(self, x, split_signal=False):
        B, C, L = x.shape
        # Split each input into durations of length self.chunk_size
        if split_signal:
            n_chunks = L // self.chunk_size
        
            if n_chunks == 0:
                raise ValueError(f"Signal length {L} is shorter than required segment length {self.chunk_size}")
            
            # Truncate to ensure divisibility
            x = x[..., :n_chunks * self.chunk_size]

            # Reshape: (B, C, n_chunks, chunk_size) -> (B, n_chunks, C, chunk_size) -> (B * n_chunks, C, chunk_size)
            x = x.view(B, C, n_chunks, self.chunk_size).permute(0, 2, 1, 3).reshape(-1, C, self.chunk_size)

        # Pass through model
        x = self.ts_encoder(x)

        # Combine outputs from same original signal
        if split_signal:
            # Reshape back and average: (B, n_chunks, Output_Dim) -> (B, Output_Dim)
            x = x.view(B, n_chunks, -1).mean(dim=1)

        return torch.nan_to_num(x)

    def load_model_from_config(self, config_path):
        with open(config_path, "r") as fp:
            config = yaml.safe_load(fp)

        encoder_hparams = config["rnn_hyperparameters"]
        encoder_hparams["hparams_encoder"] = RNNEncoderConfig(**encoder_hparams["hparams_encoder"])
        s4_hparams = config["s4_hyperparamters"]
        cpc_hparams = config["cpc_hyperparameters"]

        model = CPCModel(
            encoder_hparams=encoder_hparams,
            s4_hparams=s4_hparams,
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


class S4Wrapper(torch.nn.Module):
    """Just to match naming in the pretrained checkpoint."""
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.predictor = S4Model(*args, **kwargs)
    
    def forward(self, x):
        return self.predictor(x)


class CPCModel(torch.nn.Module):
    def __init__(self, encoder_hparams, s4_hparams, num_classes, feature_dim=512, eval_mode="finetuning", lr=1e-3, discriminative_lr_factor=0.1):
        super().__init__()
        self.encoder_hparams = encoder_hparams
        self.s4_hparams = s4_hparams
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.eval_mode = eval_mode
        self.lr = lr
        self.discriminative_lr_factor = discriminative_lr_factor

        self.encoder = RNNEncoder(**self.encoder_hparams)
        self.predictor = S4Wrapper(**self.s4_hparams)
        self.head = torch.nn.Linear(self.feature_dim, num_classes)

        if self.eval_mode == "linear":
            for p in self.rnn_encoder.parameters():
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
        pooled_features = features.mean(dim=1) # Pool the features
        out = self.head(pooled_features)
        return out # TODO: is torch.nan_to_num necessary?
    
    def get_params(self):
        head_params = list(self.head.parameters())

        if self.eval_mode == "linear":
            return [{"params": head_params, "lr": self.lr}]

        encoder_params = list(chain(*[e.parameters() for e in self.encoder]))
        predictor_params = list(chain(*[p.parameters() for p in self.predictor]))

        return [
            {"params": head_params, "lr": self.lr},
            {"params": predictor_params, "lr": self.lr * self.discriminative_lr_factor},
            {"params": encoder_params, "lr": self.lr * self.discriminative_lr_factor * self.discriminative_lr_factor}
        ]
