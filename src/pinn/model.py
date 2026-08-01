"""
Physics-Informed Neural Network (PINN) architecture and derivative computation utilities.
"""

import torch
import torch.nn as nn
import torch.autograd as autograd
from typing import List, Tuple

class PINN(nn.Module):
    """
    Physics-Informed Neural Network (PINN) architecture.
    """
    def __init__(self, layers: List[int] = [2, 64, 64, 64, 64, 1]):
        super(PINN, self).__init__()
        self.activation = nn.Tanh()
        
        self.layers = nn.ModuleList()
        for i in range(len(layers)-1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
            
        # Initialize weights using Xavier initialization
        for layer in self.layers:
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        :param x: Spatial dimension tensor
        :param t: Time dimension tensor
        """
        # Concatenate x and t
        z = torch.cat([x, t], dim=1)
        for i in range(len(self.layers) - 1):
            z = self.activation(self.layers[i](z))
        return self.layers[-1](z)

def compute_derivatives(model: nn.Module, x: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes the spatial and temporal derivatives required for Burgers' equation.
    
    :param model: The PINN model.
    :param x: Spatial coordinates.
    :param t: Temporal coordinates.
    :return: Tuple of (u, u_t, u_x, u_xx).
    """
    x.requires_grad_(True)
    t.requires_grad_(True)
    
    u = model(x, t)
    
    # First derivatives (u_t, u_x)
    u_x = autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_t = autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    
    # Second derivative (u_xx)
    u_xx = autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
    
    return u, u_t, u_x, u_xx
