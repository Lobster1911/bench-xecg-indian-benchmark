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

    def get_cls_token(self, x):
        x = self.patch_embedding(x)

        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([x, cls_token], dim=1)

        if self.num_reg_tokens > 0:
            x = self.add_reg_tokens(x)

        out = self.xlstm(x, need_expansion=False)# [:, -1, :]

        if self.num_reg_tokens > 0:
            out = self.remove_reg_tokens(out)

        out = self.layer_norm(out)

        if self.cls_type == 'max':
            cls = out.max(dim=1)[0]
        elif self.cls_type == 'mean' or self.cls_type == 'avg':
            cls = out.mean(dim=1)
        else:
            cls = out[:, -1, :]

        return cls

    def forward(self, x):
        if self.linear_probing:
            with torch.no_grad():
                cls = self.get_cls_token(x)
        else:  
            cls = self.get_cls_token(x)

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
