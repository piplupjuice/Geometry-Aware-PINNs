"""
Milestone 6: Spectral Bias
Compares the frequency spectrum of the PINN prediction against the analytical 
solution of Burgers' Equation.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
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
    if t == 0:
        return -np.sin(np.pi * x)
    
    x = np.asarray(x)
    u = np.zeros_like(x)
    
    def u_single(x_val, t_val):
        def integrand(xi):
            return np.exp(-np.cos(np.pi * xi) / (2 * np.pi * nu) - (x_val - xi)**2 / (4 * nu * t_val))
        def numerator_integrand(xi):
            return np.sin(np.pi * xi) * integrand(xi)
        
        denom, _ = quad(integrand, -1, 1, limit=100)
        num, _ = quad(numerator_integrand, -1, 1, limit=100)
        
        if abs(denom) < 1e-12:
            return 0.0
        return - num / denom
        
    for i, xi in enumerate(x):
        u[i] = u_single(xi, t)
    return u

def compute_spectrum(u, N):
    fft_vals = np.fft.fft(u)
    ps = np.abs(fft_vals)**2
    if np.sum(ps) > 0:
        ps = ps / np.sum(ps)
    freqs = np.fft.fftfreq(N, d=2/N)
    mask = freqs >= 0
    return freqs[mask], ps[mask]

def main():
    nu = 0.01 / np.pi
    x_ic, t_ic, u_ic, x_bc, t_bc, u_bc, x_f, t_f = generate_data(N_f=1000, device=device)

    model = PINN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print("Training PINN for 5000 epochs...")
    for epoch in range(5000):
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
        if epoch % 1000 == 0:
            print(f"Epoch {epoch} | Loss: {total_loss.item():.4f}")

    print("Computing Frequency Spectra...")
    x_grid = np.linspace(-1, 1, 256)
    t_eval = 0.5

    print("Computing analytical solution (this may take a few seconds)...")
    u_true = analytical_burgers(x_grid, t_eval, nu)

    x_tensor = torch.tensor(x_grid, dtype=torch.float32).reshape(-1, 1).to(device)
    t_tensor = torch.full_like(x_tensor, t_eval)
    with torch.no_grad():
        u_pred = model(x_tensor, t_tensor).cpu().numpy().flatten()

    freqs_true, ps_true = compute_spectrum(u_true, len(x_grid))
    freqs_pred, ps_pred = compute_spectrum(u_pred, len(x_grid))

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    axs[0].plot(x_grid, u_true, label='Analytical Solution', color='blue')
    axs[0].plot(x_grid, u_pred, label='PINN Prediction', color='red', linestyle='dashed')
    axs[0].set_title(f'Spatial Domain at t={t_eval}')
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('u(x,t)')
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)

    axs[1].stem(freqs_true[:15], ps_true[:15], linefmt='blue', markerfmt='bo', basefmt='b-', label='Analytical')
    axs[1].stem(freqs_pred[:15], ps_pred[:15], linefmt='red', markerfmt='rx', basefmt='r-', label='PINN')
    axs[1].set_title('Frequency Domain (Power Spectrum)')
    axs[1].set_xlabel('Frequency (k)')
    axs[1].set_ylabel('Normalized Power')
    axs[1].legend()
    axs[1].grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/milestone6_spectral.png')
    print("Plot saved to results/milestone6_spectral.png")

if __name__ == "__main__":
    main()
