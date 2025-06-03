from models.xLSTM import pretrainedxLSTM
import torch
import torch.nn as nn

class xLSTMClassification(pretrainedxLSTM):
    def __init__(
            self, 
            config,
            num_classes,
            num_channels
        ): 
        self.linear_probing = config.linear_probing
        super(xLSTMClassification, self).__init__(num_channels, config, reconstruction=False)


        self.fc = nn.Sequential(            
            nn.Dropout(config.dropout),
            nn.Linear(config.embedding_size, num_classes)
        )


    def forward(self, x):
        padding_mask = self.get_padding_mask(x)

        if self.linear_probing:
            with torch.no_grad():
                x = self.patch_embedding(x)
                cls, _ = self.forward_xlstm(x, padding_mask)
        else:  
            x = self.patch_embedding(x)
            cls, _ = self.forward_xlstm(x, padding_mask)

        res = self.fc(cls)
        return res
    
    def finetuning_params(self):
        params = [param for name, param in self.named_parameters() if 'fc' not in name]
        return params
    
    def set_eval_linear_probing(self):
        self.eval()
        self.fc.train()

    def training_params(self):
        params = []
        params.extend(self.fc.parameters())
        return params
    

class xLSTMFeatureClassification(pretrainedxLSTM):
    def __init__(
            self, 
            config,
            num_classes,
            num_channels
        ): 
        self.linear_probing = config.linear_probing
        super(xLSTMFeatureClassification, self).__init__(num_channels, config, reconstruction=False)

        self.fc = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(config.embedding_size, num_classes)
        )

    def forward(self, x):
        if self.linear_probing:
            with torch.no_grad():
                x = self.patch_embedding(x)
                _, features = self.forward_xlstm(x)
        else:  
            x = self.patch_embedding(x)
            _, features = self.forward_xlstm(x)

        res = self.fc(features)
        return res
    
    def finetuning_params(self):
        params = [param for name, param in self.named_parameters() if 'fc' not in name]
        return params
    
    def set_eval_linear_probing(self):
        self.eval()
        self.fc.train()

    def training_params(self):
        params = []
        params.extend(self.fc.parameters())
        return params
    

