"""
Milestone 7: Visualization
Generates 3D plots for the spatio-temporal solution and the loss landscape
around an optimization ravine.
"""

import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import quad
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pinn.model import PINN, compute_derivatives
from src.utils.data import generate_data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42)
np.random.seed(42)

def analytical_burgers(x, t, nu):
    if t == 0: return -np.sin(np.pi * x)
    x = np.asarray(x)
    u = np.zeros_like(x)
    def u_single(x_val, t_val):
        def integrand(xi): return np.exp(-np.cos(np.pi * xi) / (2 * np.pi * nu) - (x_val - xi)**2 / (4 * nu * t_val))
        def num_integrand(xi): return np.sin(np.pi * xi) * integrand(xi)
        denom, _ = quad(integrand, -1, 1, limit=100)
        num, _ = quad(num_integrand, -1, 1, limit=100)
        return - num / denom if abs(denom) > 1e-12 else 0.0
    for i, xi in enumerate(x): u[i] = u_single(xi, t)
    return u

def create_directions(model):
    directions = []
    weights_norm = torch.sqrt(sum(torch.norm(p)**2 for p in model.parameters() if p.requires_grad))
    for p in model.parameters():
        if p.requires_grad:
            d = torch.randn_like(p)
            directions.append(d)
        
    dir_norm = torch.sqrt(sum(torch.norm(d)**2 for d in directions))
    scale = weights_norm / dir_norm
    return [d * scale for d in directions]

def apply_directions(model, dirs, alpha, beta):
    with torch.no_grad():
        i = 0
        for p in model.parameters():
            if p.requires_grad:
                p.add_(alpha * dirs[0][i] + beta * dirs[1][i])
                i += 1

def remove_directions(model, dirs, alpha, beta):
    with torch.no_grad():
        i = 0
        for p in model.parameters():
            if p.requires_grad:
                p.sub_(alpha * dirs[0][i] + beta * dirs[1][i])
                i += 1

def main():
    nu = 0.01 / np.pi
    x_ic, t_ic, u_ic, x_bc, t_bc, u_bc, x_f, t_f = generate_data(N_f=1000, device=device)

    model = PINN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print("Training PINN for visualization...")
    for epoch in range(3000):
        optimizer.zero_grad()
        u_pred_ic = model(x_ic, t_ic)
        u_pred_bc = model(x_bc, t_bc)
        u_pred_f, u_t, u_x, u_xx = compute_derivatives(model, x_f, t_f)
        loss_data = torch.mean((u_pred_ic - u_ic)**2) + torch.mean((u_pred_bc - u_bc)**2)
        r = u_t + u_pred_f * u_x - nu * u_xx
        loss_pde = torch.mean(r**2)
        (loss_data + loss_pde).backward()
        optimizer.step()

    print("Generating 3D Spatio-Temporal Plot...")
    x_grid = np.linspace(-1, 1, 100)
    t_grid = np.linspace(0, 1, 100)
    X, T = np.meshgrid(x_grid, t_grid)

    U_true = np.zeros_like(X)
    for i, t_val in enumerate(t_grid):
        U_true[i, :] = analytical_burgers(x_grid, t_val, nu)

    X_flat = torch.tensor(X.flatten(), dtype=torch.float32).reshape(-1, 1).to(device)
    T_flat = torch.tensor(T.flatten(), dtype=torch.float32).reshape(-1, 1).to(device)
    with torch.no_grad():
        U_pred = model(X_flat, T_flat).cpu().numpy().reshape(X.shape)

    fig = plt.figure(figsize=(16, 7))

    ax1 = fig.add_subplot(121, projection='3d')
    surf1 = ax1.plot_surface(X, T, U_true, cmap=cm.viridis, linewidth=0, antialiased=True, rstride=1, cstride=1)
    ax1.set_title('Analytical Solution (Burgers Equation)', fontsize=14, pad=20)
    ax1.set_xlabel('Space (x)', fontsize=12, labelpad=10)
    ax1.set_ylabel('Time (t)', fontsize=12, labelpad=10)
    ax1.set_zlabel('u(x,t)', fontsize=12, labelpad=10)
    ax1.view_init(elev=30, azim=-60)
    ax1.set_zlim(-1.2, 1.2)

    ax2 = fig.add_subplot(122, projection='3d')
    surf2 = ax2.plot_surface(X, T, U_pred, cmap=cm.inferno, linewidth=0, antialiased=True, rstride=1, cstride=1)
    ax2.set_title('PINN Prediction (Spectral Bias)', fontsize=14, pad=20)
    ax2.set_xlabel('Space (x)', fontsize=12, labelpad=10)
    ax2.set_ylabel('Time (t)', fontsize=12, labelpad=10)
    ax2.set_zlabel('u(x,t)', fontsize=12, labelpad=10)
    ax2.view_init(elev=30, azim=-60)
    ax2.set_zlim(-1.2, 1.2)

    plt.tight_layout()
    os.makedirs('assets', exist_ok=True)
    plt.savefig('assets/3d_spatiotemporal.png', dpi=300, bbox_inches='tight')
    print("Saved assets/3d_spatiotemporal.png")
    
    print("Generating 3D Loss Landscape Plot...")
    dir1 = create_directions(model)
    dir2 = create_directions(model)
    directions = [dir1, dir2]

    alphas = np.linspace(-1.5, 1.5, 40)
    betas = np.linspace(-1.5, 1.5, 40)
    A, B = np.meshgrid(alphas, betas)
    L_grid = np.zeros_like(A)

    for i in range(len(betas)):
        for j in range(len(alphas)):
            apply_directions(model, directions, A[i,j], B[i,j])
            
            u_pred_ic = model(x_ic, t_ic)
            u_pred_bc = model(x_bc, t_bc)
            u_pred_f, u_t, u_x, u_xx = compute_derivatives(model, x_f, t_f)
            r = u_t + u_pred_f * u_x - nu * u_xx
            
            loss = torch.mean((u_pred_ic - u_ic)**2) + torch.mean((u_pred_bc - u_bc)**2) + torch.mean(r**2)
            L_grid[i,j] = loss.item()
            remove_directions(model, directions, A[i,j], B[i,j])

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    L_log = np.log10(L_grid + 1e-4)

    surf = ax.plot_surface(A, B, L_log, cmap=cm.terrain, linewidth=0, antialiased=True, rstride=1, cstride=1)
    ax.set_title('3D PINN Loss Landscape (Log Scale)', fontsize=16, pad=20)
    ax.set_xlabel('Direction 1 (alpha)', fontsize=12, labelpad=10)
    ax.set_ylabel('Direction 2 (beta)', fontsize=12, labelpad=10)
    ax.set_zlabel('Log10(Loss)', fontsize=12, labelpad=10)
    ax.view_init(elev=45, azim=45) 

    fig.colorbar(surf, shrink=0.5, aspect=10, label='Log10(Loss)')
    plt.savefig('assets/3d_loss_landscape.png', dpi=300, bbox_inches='tight')
    print("Saved assets/3d_loss_landscape.png")

if __name__ == "__main__":
    main()
