"""
Milestone 2: Hessian Spectrum Analysis
Demonstrates how to compute the exact Hessian matrix of a PINN using double backpropagation
and analyzes its eigenvalue spectrum to study optimization conditioning.
"""

import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.func import functional_call
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42)
np.random.seed(42)

class TinyPINN(nn.Module):
    def __init__(self):
        super(TinyPINN, self).__init__()
        self.fc1 = nn.Linear(2, 10)
        self.fc2 = nn.Linear(10, 1)
        self.act = nn.Tanh()
        
    def forward(self, x, t):
        z = torch.cat([x, t], dim=1)
        z = self.act(self.fc1(z))
        return self.fc2(z)

def compute_derivatives_func(model, params, x, t):
    x.requires_grad_(True)
    t.requires_grad_(True)
    u = functional_call(model, params, (x, t))
    
    u_x = autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_t = autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    u_xx = autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
    return u, u_t, u_x, u_xx

def loss_fn_flat(theta_flat, model, param_names, shapes, x_ic, t_ic, u_ic, x_bc, t_bc, u_bc, x_f, t_f, nu):
    params = {}
    idx = 0
    for name, shape in zip(param_names, shapes):
        numel = int(np.prod(shape))
        params[name] = theta_flat[idx:idx+numel].view(shape)
        idx += numel
        
    u_pred_ic = functional_call(model, params, (x_ic, t_ic))
    loss_ic = torch.mean((u_pred_ic - u_ic)**2)
    
    u_pred_bc = functional_call(model, params, (x_bc, t_bc))
    loss_bc = torch.mean((u_pred_bc - u_bc)**2)
    
    u_pred_f, u_t, u_x, u_xx = compute_derivatives_func(model, params, x_f, t_f)
    residual = u_t + u_pred_f * u_x - nu * u_xx
    loss_pde = torch.mean(residual**2)
    
    return loss_ic + loss_bc + loss_pde

def main():
    tiny_model = TinyPINN().to(device)
    nu = 0.01 / np.pi

    x_ic = torch.rand(10, 1, device=device) * 2 - 1
    t_ic = torch.zeros(10, 1, device=device)
    u_ic = -torch.sin(np.pi * x_ic)

    t_bc = torch.rand(10, 1, device=device)
    x_bc = torch.where(torch.rand_like(t_bc) < 0.5, -torch.ones_like(t_bc), torch.ones_like(t_bc))
    u_bc = torch.zeros(10, 1, device=device)

    x_f = torch.rand(50, 1, device=device) * 2 - 1
    t_f = torch.rand(50, 1, device=device)

    param_names = [name for name, _ in tiny_model.named_parameters()]
    shapes = [p.shape for p in tiny_model.parameters()]
    flat_params = torch.cat([p.view(-1) for p in tiny_model.parameters()]).clone().detach().requires_grad_(True)
    W = flat_params.numel()
    print(f"Total parameters (W): {W}")

    print("Computing exact Hessian... (This may take a moment)")
    loss = loss_fn_flat(flat_params, tiny_model, param_names, shapes, 
                        x_ic, t_ic, u_ic, x_bc, t_bc, u_bc, x_f, t_f, nu)
    grad_loss = torch.autograd.grad(loss, flat_params, create_graph=True)[0]
    H = torch.zeros((W, W), device=device)

    for i in range(W):
        grad_outputs = torch.zeros_like(grad_loss)
        grad_outputs[i] = 1.0
        hessian_row = torch.autograd.grad(grad_loss, flat_params, grad_outputs=grad_outputs, retain_graph=True)[0]
        H[i] = hessian_row

    H_np = H.detach().cpu().numpy()
    eigenvalues = np.linalg.eigvalsh(H_np)
    eigenvalues = eigenvalues[eigenvalues > 1e-12]
    
    if len(eigenvalues) > 0:
        cond_number = np.max(eigenvalues) / np.min(eigenvalues)
        print(f"Max Eigenvalue: {np.max(eigenvalues):.4e}")
        print(f"Min Eigenvalue: {np.min(eigenvalues):.4e}")
        print(f"Condition Number: {cond_number:.4e}")

        plt.figure(figsize=(10, 5))
        plt.hist(np.log10(eigenvalues), bins=50, edgecolor='black')
        plt.axvline(np.log10(np.max(eigenvalues)), color='r', linestyle='dashed', linewidth=2, label=f'Max EV: {np.max(eigenvalues):.1e}')
        plt.axvline(np.log10(np.min(eigenvalues)), color='g', linestyle='dashed', linewidth=2, label=f'Min EV: {np.min(eigenvalues):.1e}')
        plt.title('Eigenvalue Spectrum of PINN Hessian (Log Scale)')
        plt.xlabel('Log10(Eigenvalue)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        os.makedirs('results', exist_ok=True)
        plt.savefig('results/milestone2_hessian.png')
        print("Plot saved to results/milestone2_hessian.png")
    else:
        print("Could not compute valid eigenvalues.")

if __name__ == '__main__':
    main()
