import numpy as np
import torch
from torch import nn
from sklearn import random_projection

class VanillaQuine(nn.Module):
    
    def __init__(self, n_hidden, n_layers, act_func):
        super().__init__()
    
        # Create layers and parameter lists
        self.param_list = []
        self.param_names = []
        layers = []
        for i in range(n_layers):
            n_out = 1 if i == (n_layers-1) else n_hidden
            current_layer = nn.Linear(n_hidden, n_out, bias=True)
            layers.append(current_layer)
            layers.append(act_func())

            self.param_list.append(current_layer.weight)
            self.param_names.append("layer{}_weight".format(i+1))
            self.param_list.append(current_layer.bias)
            self.param_names.append("layer{}_bias".format(i+1))
        layers.pop(-1)  # Remove final activation
        
        # Create the parameter counting function
        self.num_params_arr = np.array([np.prod(p.shape) for p in self.param_list])
        self.cum_params_arr = np.cumsum(self.num_params_arr)
        self.num_params = int(self.cum_params_arr[-1])
        
        # Create random projection matrix
        X = np.random.rand(1, self.num_params)
        transformer = random_projection.GaussianRandomProjection(n_components=n_hidden)
        transformer.fit(X)
        rand_proj_matrix = transformer.components_
        rand_proj_layer = nn.Linear(self.num_params, n_hidden, bias=False)
        rand_proj_layer.weight.data = torch.tensor(rand_proj_matrix, dtype=torch.float32)
        for p in rand_proj_layer.parameters():
            p.requires_grad_(False)
        layers.insert(0, rand_proj_layer)
        
        # Create the final layers
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)
        
    def get_param(self, idx):
        assert idx < self.num_params
        subtract = 0
        param = None
        normalized_idx = None
        for i, n_params in enumerate(self.cum_params_arr):
            if idx < n_params:
                param = self.param_list[i]
                normalized_idx = idx - subtract
                break
            else:
                subtract = n_params
        return param.view(-1)[normalized_idx]

