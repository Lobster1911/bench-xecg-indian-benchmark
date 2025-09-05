from models.xLSTM import pretrainedxLSTM
import torch
import torch.nn as nn
from models.utils import get_normalization_layer

class xLSTMClassification(pretrainedxLSTM):
    def __init__(
            self, 
            config,
            num_classes,
            num_channels,
        ): 
        self.linear_probing = config.linear_probing
        super(xLSTMClassification, self).__init__(num_channels, config, reconstruction=False)

        emb_size = config.embedding_size * 2 if config.cls_type == 'mix' else config.embedding_size
        self.head = nn.Sequential(
            get_normalization_layer(config, emb_size),
            nn.Linear(emb_size, num_classes)
        )

    def forward(self, x):
        padding_mask = self.get_padding_mask(x)

        if self.linear_probing:
            with torch.no_grad():
                x = self.patch_embedding(x)
                cls, _ = self.forward_core(x, padding_mask)
        else:  
            x = self.patch_embedding(x)
            cls, _ = self.forward_core(x, padding_mask)

        res = self.head(cls)
        return res

class xLSTMFeatureClassification(pretrainedxLSTM):
    def __init__(
            self, 
            config,
            num_classes,
            num_channels,
            minute_aggregation=False,
        ): 
        self.minute_aggregation = minute_aggregation
        self.linear_probing = config.linear_probing
        super(xLSTMFeatureClassification, self).__init__(num_channels, config, reconstruction=False)

        # emb_size = config.embedding_size * 2 if config.cls_type == 'mix' else config.embedding_size
        self.head = nn.Sequential(
            get_normalization_layer(config, config.embedding_size),
            nn.Linear(config.embedding_size, num_classes)
        )

    def forward(self, x):
        if self.linear_probing:
            with torch.no_grad():
                x = self.patch_embedding(x)
                _, features = self.forward_core(x)
        else:  
            x = self.patch_embedding(x)
            _, features = self.forward_core(x)

        if self.minute_aggregation:
            features = self.aggregate_per_minute(features)

        res = self.head(features)
        return res
    
    def aggregate_per_minute(self, features):
        patches_per_segment = (60 * self.sampling_freq) // self.patch_size
        n_minutes = int(features.size(1)) // patches_per_segment  

        features = features.reshape(
            features.shape[0],
            n_minutes,
            patches_per_segment,
            features.shape[-1]
        )
        features = features.permute(0, 2, 1, 3).contiguous()
        features, _ = self.pooling(features)
        return features

