"""
Data generation utilities for PINN training.
"""

import torch
import numpy as np
from typing import Tuple

def generate_data(
    N_ic: int = 200, 
    N_bc: int = 200, 
    N_f: int = 10000, 
    device: torch.device = torch.device('cpu')
) -> Tuple[torch.Tensor, ...]:
    """
    Generates initial conditions, boundary conditions, and collocation points for Burgers' equation.
    
    :param N_ic: Number of points for the initial condition.
    :param N_bc: Number of points for the boundary conditions.
    :param N_f: Number of collocation points for the PDE residual.
    :param device: The PyTorch device to create tensors on.
    :return: A tuple of tensors (x_ic, t_ic, u_ic, x_bc, t_bc, u_bc, x_f, t_f).
    """
    # Initial condition: t=0, x in [-1,1]
    x_ic = torch.rand(N_ic, 1, device=device) * 2 - 1
    t_ic = torch.zeros(N_ic, 1, device=device)
    u_ic = -torch.sin(np.pi * x_ic)
    
    # Boundary conditions: x=-1 and x=1, t in [0,1]
    t_bc = torch.rand(N_bc, 1, device=device)
    x_bc = torch.where(torch.rand_like(t_bc) < 0.5, 
                       -torch.ones_like(t_bc), torch.ones_like(t_bc))
    u_bc = torch.zeros(N_bc, 1, device=device) # u=0 at boundaries
    
    # Collocation points
    x_f = torch.rand(N_f, 1, device=device) * 2 - 1
    t_f = torch.rand(N_f, 1, device=device)
    
    return x_ic, t_ic, u_ic, x_bc, t_bc, u_bc, x_f, t_f
