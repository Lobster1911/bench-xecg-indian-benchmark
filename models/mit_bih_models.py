from models.xLSTM import pretrainedxLSTM  
import torch
import torch.nn as nn

class xLSTMClassificationMIT_BIH(pretrainedxLSTM):
    def __init__(
            self, 
            config,
            num_classes,
            num_channels
        ): 

        self.linear_probing = config.linear_probing
        super(xLSTMClassificationMIT_BIH, self).__init__(num_channels, config, reconstruction=False)

        self.fc = nn.Sequential(
            nn.Linear(config.embedding_size, num_classes)
        )

        self.r_peak_pos_fc = nn.Sequential(
            nn.Linear(config.embedding_size, config.patch_size)
        )

    def get_features(self, x):
        x = self.patch_embedding(x)

        if self.num_reg_tokens > 0:
            x = self.add_reg_tokens(x)

        out = self.xlstm(x, need_expansion=False) # [batch_size, embedding_dim]

        if self.num_reg_tokens > 0:
            out = self.remove_reg_tokens(out)

        out = self.layer_norm(out)
        return out  
    

    def forward(self, x):
        if self.linear_probing:
            with torch.no_grad():
                out = self.get_features(x)
        else:  
            out = self.get_features(x)

        cls = self.fc(out)
        r_peak_pos = self.r_peak_pos_fc(out)
        return cls, r_peak_pos

    def finetuning_params(self):
        params = [param for name, param in self.named_parameters() if 'fc' not in name]
        return params

    def training_params(self):
        params = []
        params.extend(self.fc.parameters())
        params.extend(self.r_peak_pos_fc.parameters())
        return params
    
    def set_eval_linear_probing(self):
        self.eval()
        self.fc.train()
        self.r_peak_pos_fc.train()