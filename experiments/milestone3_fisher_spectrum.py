"""
Milestone 3: Fisher Information Spectrum
Computes the Fisher Information Matrix (empirical approximation) via the Jacobian
of the residual, demonstrating an alternative measure of local geometry.
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

def residual_single_point(model, params, x, t, nu):
    x_in = x.reshape(1, 1)
    t_in = t.reshape(1, 1)
    x_in.requires_grad_(True)
    t_in.requires_grad_(True)
    
    u = functional_call(model, params, (x_in, t_in))
    u_x = autograd.grad(u, x_in, torch.ones_like(u), create_graph=True)[0]
    u_t = autograd.grad(u, t_in, torch.ones_like(u), create_graph=True)[0]
    u_xx = autograd.grad(u_x, x_in, torch.ones_like(u_x), create_graph=True)[0]
    
    r = u_t + u * u_x - nu * u_xx
    return r.squeeze()

def main():
    tiny_model = TinyPINN().to(device)
    nu = 0.01 / np.pi

    x_f = torch.rand(50, 1, device=device) * 2 - 1
    t_f = torch.rand(50, 1, device=device)

    param_names = [name for name, _ in tiny_model.named_parameters()]
    shapes = [p.shape for p in tiny_model.parameters()]
    flat_params = torch.cat([p.view(-1) for p in tiny_model.parameters()]).clone().detach().requires_grad_(True)
    W = flat_params.numel()

    print("Computing Residual Jacobian (J)...")
    N_f = x_f.shape[0]
    J = torch.zeros((N_f, W), device=device)

    for i in range(N_f):
        r_i = residual_single_point(tiny_model, 
                                    {name: p.view(s) for name, p, s in zip(param_names, flat_params.split([int(np.prod(s)) for s in shapes]), shapes)}, 
                                    x_f[i], t_f[i], nu)
        grad_r = torch.autograd.grad(r_i, flat_params, retain_graph=True, create_graph=False)[0]
        J[i] = grad_r

    print("Computing Fisher Information Matrix (F = J^T J)...")
    F = (2.0 / N_f) * (J.T @ J)

    F_np = F.detach().cpu().numpy()
    fisher_eigenvalues = np.linalg.eigvalsh(F_np)
    fisher_eigenvalues = fisher_eigenvalues[fisher_eigenvalues > 1e-12]

    if len(fisher_eigenvalues) > 0:
        print(f"Fisher Max EV: {np.max(fisher_eigenvalues):.4e}")
        print(f"Fisher Min EV: {np.min(fisher_eigenvalues):.4e}")
        print(f"Fisher Condition Number: {np.max(fisher_eigenvalues) / (np.min(fisher_eigenvalues)+1e-12):.4e}")

        plt.figure(figsize=(10, 5))
        plt.hist(np.log10(fisher_eigenvalues), bins=50, edgecolor='black', color='orange', alpha=0.7, label='Fisher Spectrum')
        plt.axvline(np.log10(np.max(fisher_eigenvalues)), color='r', linestyle='dashed', linewidth=2, label=f'Max EV: {np.max(fisher_eigenvalues):.1e}')
        plt.axvline(np.log10(np.min(fisher_eigenvalues)), color='g', linestyle='dashed', linewidth=2, label=f'Min EV: {np.min(fisher_eigenvalues):.1e}')
        plt.title('Eigenvalue Spectrum of PINN Fisher Information Matrix (Log Scale)')
        plt.xlabel('Log10(Eigenvalue)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        os.makedirs('results', exist_ok=True)
        plt.savefig('results/milestone3_fisher.png')
        print("Plot saved to results/milestone3_fisher.png")

if __name__ == '__main__':
    main()
