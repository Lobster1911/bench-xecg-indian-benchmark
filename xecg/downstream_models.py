from xecg.xECG import xECG
import torch
import torch.nn as nn

class xECGClassification(xECG):
    def __init__(
            self, 
            config,
            num_classes,
            linear_probing=False,
            cls_type='avg',
        ): 
        self.linear_probing = linear_probing
        super(xECGClassification, self).__init__(cls_type=cls_type, config=config)

        self.head = nn.Sequential(
            get_normalization_layer(config, config['embedding_size']),
            nn.Linear(config['embedding_size'], num_classes)
        )

    def forward(self, x):
        if self.linear_probing:
            with torch.no_grad():
                cls, _ = super().forward(x)
        else:  
            cls, _ = super().forward(x)

        res = self.head(cls)
        return res

class xECGFeatureClassification(xECG):
    def __init__(
            self, 
            config,
            num_classes,
            linear_probing=False,
        ): 
        self.linear_probing = linear_probing
        super(xECGFeatureClassification, self).__init__(cls_type=None, config=config)

        self.head = nn.Sequential(
            get_normalization_layer(config, config['embedding_size']),
            nn.Linear(config['embedding_size'], num_classes)
        )

    def forward(self, x):
        if self.linear_probing:
            with torch.no_grad():
                _, features = super().forward(x)
        else:  
            _, features = super().forward(x)


        res = self.head(features)
        return res
    
    
class xECGMinuteLevelClassification(xECG):
    """ 
    
    This model aggregates features per minute before classification.
    This is used for the Sleep Apnea-ECG dataset, where annotations are a minute level 
    
    """
    def __init__(
        self, 
        config,
        num_classes,
        linear_probing=False,
        cls_type='max'
    ): 
        self.linear_probing = linear_probing
        super(xECGFeatureClassification, self).__init__(cls_type=cls_type, config=config)

        self.head = nn.Sequential(
            get_normalization_layer(config, config['embedding_size']),
            nn.Linear(config['embedding_size'], num_classes)
        )

    def forward(self, x):
        if self.linear_probing:
            with torch.no_grad():
                _, features = super().forward(x)
        else:  
            _, features = super().forward(x)

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


def get_normalization_layer(config, embedding_size=None):
    if config['cls_normalization'] == 'layer':
        return nn.LayerNorm(embedding_size)
    elif config['cls_normalization'] == 'batch':
        return nn.BatchNorm1d(embedding_size)
    elif config['cls_normalization'] == 'instance':
        return nn.InstanceNorm1d(embedding_size)
    else:
        return nn.Identity()