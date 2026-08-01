"""
Milestone 4: K-FAC Update Geometry
Demonstrates the computation of Kronecker-Factored Approximate Curvature (K-FAC)
and the calculation of the natural gradient for a single-layer PINN.
"""

import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42)
np.random.seed(42)

class SingleLayerPINN(nn.Module):
    def __init__(self):
        super(SingleLayerPINN, self).__init__()
        self.fc1 = nn.Linear(2, 10)
        self.fc2 = nn.Linear(10, 1)
        self.act = nn.Tanh()
        
    def forward(self, x, t):
        z = torch.cat([x, t], dim=1)
        z = self.act(self.fc1(z))
        return self.fc2(z)

def compute_residuals(model, x, t, nu):
    x.requires_grad_(True)
    t.requires_grad_(True)
    u = model(x, t)
    u_x = autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_t = autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    u_xx = autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
    return u_t + u * u_x - nu * u_xx

def main():
    model = SingleLayerPINN().to(device)
    nu = 0.01 / np.pi
    x_f = torch.rand(100, 1, device=device) * 2 - 1
    t_f = torch.rand(100, 1, device=device)

    activations = {}
    def hook_fn(module, input, output):
        activations[module] = input[0].detach()

    grad_outputs = {}
    def grad_hook_fn(module, grad_input, grad_output):
        grad_outputs[module] = grad_output[0].detach()

    model.fc1.register_forward_hook(hook_fn)
    model.fc1.register_full_backward_hook(grad_hook_fn)

    residuals = compute_residuals(model, x_f, t_f, nu)
    loss_pde = torch.mean(residuals**2)
    loss_pde.backward()

    with torch.no_grad():
        layer = model.fc1
        
        a_in = activations[layer]
        ones = torch.ones(a_in.shape[0], 1, device=device)
        a_in_aug = torch.cat([a_in, ones], dim=1)
        A = (a_in_aug.T @ a_in_aug) / a_in_aug.shape[0]
        
        delta = grad_outputs[layer]
        B = (delta.T @ delta) / delta.shape[0]
        
        A_inv = torch.linalg.inv(A + 1e-4 * torch.eye(A.shape[0], device=device))
        B_inv = torch.linalg.inv(B + 1e-4 * torch.eye(B.shape[0], device=device))
        
        G_weight = layer.weight.grad.detach()
        G_bias = layer.bias.grad.detach()
        
        G_aug = torch.cat([G_weight, G_bias.unsqueeze(1)], dim=1)
        
        Delta_W_aug = B_inv @ G_aug @ A_inv.T
        
        nat_grad_weight = Delta_W_aug[:, :-1]
        nat_grad_bias = Delta_W_aug[:, -1]
        
        print("Euclidean Gradient Norm (Weight):", torch.linalg.norm(G_weight).item())
        print("Natural Gradient Norm (Weight):", torch.linalg.norm(nat_grad_weight).item())
        
        lr = 1e-3
        layer.weight.data -= lr * nat_grad_weight
        layer.bias.data -= lr * nat_grad_bias
        print("K-FAC update applied successfully via Riemannian geometry insights.")

if __name__ == "__main__":
    main()
