from torch import nn

class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()

    def training_params(self):
        """
        Defines the parameters to be optimized during training. These parameters will receive the main learning rate ([config.lr_head]).
        """
        return self.head.parameters()
    
    def finetuning_params(self):
        """
        Defines the parameters to be optimized during finetuning. These parameters will receive a smaller learning rate ([config.lr_core]).
        """
        params = [param for name, param in self.named_parameters() if 'head' not in name]
        return params
    
    def set_eval_linear_probing(self):
        self.eval()
        self.head.train()

    def get_layers(self):
        """
        This function should return the layers of the model where to apply the layerwise decay
        """
        raise NotImplementedError()
    
    def additional_params(self, lr, last_layer_lr, wd):
        """
        This fucntion should return additional parameters used by a model (like classification token and so on...)
        """
        raise NotImplementedError()

    def get_features(self, x, feature_classification=False):
        """
        This function should be the complete forward pass apart from the classification head.
        """
        raise NotImplementedError

    def get_params_layerwise_decay(self, lr_decay, lr, wd):
        layers = self.get_layers()
        num_layers = len(layers) + 1 

        params = []

        for i, layer in enumerate(layers):
            layer_lr = lr * (lr_decay ** (num_layers - i - 1))  # Earlier layers get smaller LR
            layer_params = layer.parameters()
            params.append({"params": layer_params, "lr": layer_lr, "name": f"layer_{i}", "weight_decay": wd})

        layer_lr = lr * (lr_decay ** num_layers)

        params.extend(self.additional_params(lr, layer_lr, wd))
        return params
    

        