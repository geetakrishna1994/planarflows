import torch
import torch.nn as nn
from planarflow import PlanarFlow 

class Flow(nn.Module) :
    def __init__(self, dim=2, n_layers = 8) :
        super().__init__()
        self.layers = nn.ModuleList([PlanarFlow(dim=dim) for _ in range(n_layers)])

    def forward(self, z) :
        z_new = z
        total_log_jacobian = 0.
        for layer in self.layers :
            z_new, logjacobian = layer(z_new)
            total_log_jacobian += logjacobian
        
        return z_new, total_log_jacobian



