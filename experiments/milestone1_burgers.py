"""
Milestone 1: Burgers' Equation using PINNs
Demonstrates the foundational Physics-Informed Neural Network approach to solving Burgers' equation.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pinn.model import PINN, compute_derivatives
from src.utils.data import generate_data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
torch.manual_seed(42)
np.random.seed(42)

def loss_fn(model, x_ic, t_ic, u_ic, x_bc, t_bc, u_bc, x_f, t_f, nu=0.01/np.pi):
    u_pred_ic = model(x_ic, t_ic)
    loss_ic = torch.mean((u_pred_ic - u_ic)**2)
    
    u_pred_bc = model(x_bc, t_bc)
    loss_bc = torch.mean((u_pred_bc - u_bc)**2)
    
    u_pred_f, u_t, u_x, u_xx = compute_derivatives(model, x_f, t_f)
    residual = u_t + u_pred_f * u_x - nu * u_xx
    loss_pde = torch.mean(residual**2)
    
    return loss_ic, loss_bc, loss_pde

def main():
    model = PINN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    x_ic, t_ic, u_ic, x_bc, t_bc, u_bc, x_f, t_f = generate_data(device=device)

    loss_history = {'ic': [], 'bc': [], 'pde': [], 'total': []}

    print("Starting training...")
    for epoch in range(5000):
        optimizer.zero_grad()
        
        loss_ic, loss_bc, loss_pde = loss_fn(model, x_ic, t_ic, u_ic, 
                                             x_bc, t_bc, u_bc, 
                                             x_f, t_f)
        
        total_loss = loss_ic + loss_bc + loss_pde
        total_loss.backward()
        optimizer.step()
        
        if epoch % 500 == 0:
            loss_history['ic'].append(loss_ic.item())
            loss_history['bc'].append(loss_bc.item())
            loss_history['pde'].append(loss_pde.item())
            loss_history['total'].append(total_loss.item())
            print(f"Epoch {epoch:4d} | IC: {loss_ic.item():.4f} | BC: {loss_bc.item():.4f} | PDE: {loss_pde.item():.4f} | Total: {total_loss.item():.4f}")

    plt.figure(figsize=(10, 6))
    plt.plot(loss_history['ic'], label='IC Loss', marker='o')
    plt.plot(loss_history['bc'], label='BC Loss', marker='s')
    plt.plot(loss_history['pde'], label='PDE Loss', marker='^')
    plt.yscale('log')
    plt.xlabel('Epochs (x500)')
    plt.ylabel('Loss (log scale)')
    plt.legend()
    plt.title('PINNs Optimization Dynamics (Burgers Equation)')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/milestone1_loss.png')
    print("Saved plot to results/milestone1_loss.png")

if __name__ == "__main__":
    main()
