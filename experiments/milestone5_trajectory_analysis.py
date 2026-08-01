"""
Milestone 5: Optimization Trajectory Analysis
Monitors the condition number, cosine similarity (gradient interference), and
loss curves during PINN training.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pinn.model import PINN, compute_derivatives
from src.utils.data import generate_data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42)
np.random.seed(42)

activations = {}
grad_outputs = {}
def hook_fn(module, input, output): activations[module] = input[0].detach()
def grad_hook_fn(module, grad_input, grad_output): grad_outputs[module] = grad_output[0].detach()

def flatten_grads(model):
    grads = []
    for p in model.parameters():
        if p.grad is not None:
            grads.append(p.grad.view(-1))
    if len(grads) > 0:
        return torch.cat(grads)
    return torch.tensor([], device=device)

def compute_gradient_metrics(model, optimizer, x_ic, t_ic, u_ic, x_bc, t_bc, u_bc, x_f, t_f, nu):
    optimizer.zero_grad()
    u_pred_ic = model(x_ic, t_ic)
    u_pred_bc = model(x_bc, t_bc)
    loss_data = torch.mean((u_pred_ic - u_ic)**2) + torch.mean((u_pred_bc - u_bc)**2)
    loss_data.backward()
    g_data = flatten_grads(model).detach().clone()
    
    optimizer.zero_grad()
    u_pred_f, u_t, u_x, u_xx = compute_derivatives(model, x_f, t_f)
    r = u_t + u_pred_f * u_x - nu * u_xx
    loss_pde = torch.mean(r**2)
    loss_pde.backward()
    g_pde = flatten_grads(model).detach().clone()
    
    if len(g_data) > 0 and len(g_pde) > 0:
        cos_sim = torch.dot(g_data, g_pde) / (torch.linalg.norm(g_data) * torch.linalg.norm(g_pde) + 1e-8)
    else:
        cos_sim = torch.tensor(0.0)
    
    layer = model.fc1
    if layer in activations and layer in grad_outputs:
        a_in = activations[layer]
        ones = torch.ones(a_in.shape[0], 1, device=device)
        a_in_aug = torch.cat([a_in, ones], dim=1)
        A = (a_in_aug.T @ a_in_aug) / a_in_aug.shape[0]
        
        delta = grad_outputs[layer]
        B = (delta.T @ delta) / delta.shape[0]
        
        eig_A = torch.linalg.eigvalsh(A)
        eig_B = torch.linalg.eigvalsh(B)
        
        eig_A = eig_A[eig_A > 1e-8]
        eig_B = eig_B[eig_B > 1e-8]
        
        cond_A = (torch.max(eig_A) / torch.min(eig_A)).item() if len(eig_A) > 0 else float('nan')
        cond_B = (torch.max(eig_B) / torch.min(eig_B)).item() if len(eig_B) > 0 else float('nan')
    else:
        cond_A, cond_B = float('nan'), float('nan')
    
    return loss_data.item(), loss_pde.item(), cos_sim.item(), cond_A, cond_B

def main():
    nu = 0.01 / np.pi
    x_ic, t_ic, u_ic, x_bc, t_bc, u_bc, x_f, t_f = generate_data(N_f=1000, device=device)

    model = PINN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.fc1.register_forward_hook(hook_fn)
    model.fc1.register_full_backward_hook(grad_hook_fn)

    history = {'data': [], 'pde': [], 'cos_sim': [], 'cond_A': [], 'cond_B': []}
    epochs = 5000

    print("Starting training with geometric diagnostics...")
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        u_pred_ic = model(x_ic, t_ic)
        u_pred_bc = model(x_bc, t_bc)
        u_pred_f, u_t, u_x, u_xx = compute_derivatives(model, x_f, t_f)
        
        loss_data = torch.mean((u_pred_ic - u_ic)**2) + torch.mean((u_pred_bc - u_bc)**2)
        r = u_t + u_pred_f * u_x - nu * u_xx
        loss_pde = torch.mean(r**2)
        
        total_loss = loss_data + loss_pde
        total_loss.backward()
        optimizer.step()
        
        if epoch % 250 == 0:
            ld, lp, cs, cA, cB = compute_gradient_metrics(model, optimizer, x_ic, t_ic, u_ic, x_bc, t_bc, u_bc, x_f, t_f, nu)
            history['data'].append(ld)
            history['pde'].append(lp)
            history['cos_sim'].append(cs)
            history['cond_A'].append(cA)
            history['cond_B'].append(cB)
            print(f"Epoch {epoch:4d} | Data: {ld:.4f} | PDE: {lp:.4f} | CosSim: {cs:.4f} | Cond(A): {cA:.2f} | Cond(B): {cB:.2f}")

    fig, axs = plt.subplots(3, 1, figsize=(10, 12))

    axs[0].plot(history['data'], label='Data Loss', marker='o')
    axs[0].plot(history['pde'], label='PDE Loss', marker='s')
    axs[0].set_yscale('log')
    axs[0].set_title('Loss Curves')
    axs[0].legend()

    axs[1].plot(history['cos_sim'], label='Cosine Similarity', color='red', marker='^')
    axs[1].axhline(0, color='black', linestyle='--')
    axs[1].set_title('Gradient Interference (Cosine Similarity)')
    axs[1].legend()

    cond_A_plot = [x for x in history['cond_A'] if not np.isnan(x)]
    cond_B_plot = [x for x in history['cond_B'] if not np.isnan(x)]

    if len(cond_A_plot) > 0 and len(cond_B_plot) > 0:
        axs[2].plot(cond_A_plot, label='Condition Number A (Inputs)', color='green', marker='d')
        axs[2].plot(cond_B_plot, label='Condition Number B (Gradients)', color='purple', marker='x')
        axs[2].set_yscale('log')
        axs[2].set_title('K-FAC Condition Numbers (Layer 1)')
        axs[2].legend()

    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/milestone5_trajectory.png')
    print("Plot saved to results/milestone5_trajectory.png")

if __name__ == "__main__":
    main()
