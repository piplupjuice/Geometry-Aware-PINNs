"""
Milestone 8: Optimizer Dynamics
Compares the optimization trajectory of Adam and Adam + L-BFGS, focusing
on the Data-PDE loss gap.
"""

import torch
import torch.nn as nn
import torch.autograd as autograd
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

def flatten_grads(model):
    grads = []
    for p in model.parameters():
        if p.grad is not None:
            grads.append(p.grad.view(-1))
    if len(grads) > 0:
        return torch.cat(grads)
    return torch.tensor([], device=device)

def compute_loss_and_grads(model, x_ic, t_ic, u_ic, x_bc, t_bc, u_bc, x_f, t_f, nu):
    model.zero_grad()
    
    u_pred_ic = model(x_ic, t_ic)
    u_pred_bc = model(x_bc, t_bc)
    loss_data = torch.mean((u_pred_ic - u_ic)**2) + torch.mean((u_pred_bc - u_bc)**2)
    
    u_pred_f, u_t, u_x, u_xx = compute_derivatives(model, x_f, t_f)
    r = u_t + u_pred_f * u_x - nu * u_xx
    loss_pde = torch.mean(r**2)
    
    total_loss = loss_data + loss_pde
    total_loss.backward(retain_graph=True)
    
    g_data = flatten_grads(model).detach().clone()
    
    model.zero_grad()
    loss_pde.backward()
    g_pde = flatten_grads(model).detach().clone()
    
    if len(g_data) > 0 and len(g_pde) > 0:
        cos_sim = torch.dot(g_data, g_pde) / (torch.linalg.norm(g_data) * torch.linalg.norm(g_pde) + 1e-8)
    else:
        cos_sim = torch.tensor(0.0)
    
    return loss_data.item(), loss_pde.item(), cos_sim.item()

def train_model(strategy, x_ic, t_ic, u_ic, x_bc, t_bc, u_bc, x_f, t_f, nu):
    model = PINN().to(device)
    
    if strategy == 'adam_only':
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        epochs = 4000
        history = {'data': [], 'pde': [], 'cos_sim': []}
        
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
            
            if epoch % 100 == 0:
                ld, lp, cs = compute_loss_and_grads(model, x_ic, t_ic, u_ic, x_bc, t_bc, u_bc, x_f, t_f, nu)
                history['data'].append(ld)
                history['pde'].append(lp)
                history['cos_sim'].append(cs)
        return history
        
    elif strategy == 'adam_then_lbfgs':
        optimizer_adam = torch.optim.Adam(model.parameters(), lr=1e-3)
        for epoch in range(1500):
            optimizer_adam.zero_grad()
            u_pred_ic = model(x_ic, t_ic)
            u_pred_bc = model(x_bc, t_bc)
            u_pred_f, u_t, u_x, u_xx = compute_derivatives(model, x_f, t_f)
            loss_data = torch.mean((u_pred_ic - u_ic)**2) + torch.mean((u_pred_bc - u_bc)**2)
            r = u_t + u_pred_f * u_x - nu * u_xx
            loss_pde = torch.mean(r**2)
            (loss_data + loss_pde).backward()
            optimizer_adam.step()
            
        optimizer_lbfgs = torch.optim.LBFGS(model.parameters(), lr=0.5, max_iter=20, 
                                           history_size=50, tolerance_grad=1e-7, 
                                           tolerance_change=1e-9)
        history = {'data': [], 'pde': [], 'cos_sim': []}
        
        def closure():
            optimizer_lbfgs.zero_grad()
            u_pred_ic = model(x_ic, t_ic)
            u_pred_bc = model(x_bc, t_bc)
            u_pred_f, u_t, u_x, u_xx = compute_derivatives(model, x_f, t_f)
            loss_data = torch.mean((u_pred_ic - u_ic)**2) + torch.mean((u_pred_bc - u_bc)**2)
            r = u_t + u_pred_f * u_x - nu * u_xx
            loss_pde = torch.mean(r**2)
            loss = loss_data + loss_pde
            loss.backward()
            return loss
        
        for i in range(100):
            optimizer_lbfgs.step(closure)
            ld, lp, cs = compute_loss_and_grads(model, x_ic, t_ic, u_ic, x_bc, t_bc, u_bc, x_f, t_f, nu)
            history['data'].append(ld)
            history['pde'].append(lp)
            history['cos_sim'].append(cs)
            
        return history

def main():
    nu = 0.01 / np.pi
    x_ic, t_ic, u_ic, x_bc, t_bc, u_bc, x_f, t_f = generate_data(N_f=1000, device=device)

    print("Training Adam only...")
    hist_adam = train_model('adam_only', x_ic, t_ic, u_ic, x_bc, t_bc, u_bc, x_f, t_f, nu)

    print("Training Adam -> L-BFGS...")
    hist_lbfgs = train_model('adam_then_lbfgs', x_ic, t_ic, u_ic, x_bc, t_bc, u_bc, x_f, t_f, nu)

    fig, axs = plt.subplots(3, 1, figsize=(10, 12))

    axs[0].plot(hist_adam['data'], label='Adam: Data Loss', color='blue')
    axs[0].plot(hist_adam['pde'], label='Adam: PDE Loss', color='cyan')
    axs[0].plot(hist_lbfgs['data'], label='L-BFGS: Data Loss', color='red', linestyle='--')
    axs[0].plot(hist_lbfgs['pde'], label='L-BFGS: PDE Loss', color='orange', linestyle='--')
    axs[0].set_yscale('log')
    axs[0].set_title('Loss Curves')
    axs[0].legend()

    axs[1].plot(hist_adam['cos_sim'], label='Adam: Cosine Similarity', color='blue')
    axs[1].plot(hist_lbfgs['cos_sim'], label='L-BFGS: Cosine Similarity', color='red')
    axs[1].axhline(0, color='black', linestyle='--')
    axs[1].set_title('Gradient Interference (Cosine Similarity)')
    axs[1].legend()

    axs[2].plot([d - p for d, p in zip(hist_adam['data'], hist_adam['pde'])], label='Adam: Data - PDE Gap', color='blue')
    axs[2].plot([d - p for d, p in zip(hist_lbfgs['data'], hist_lbfgs['pde'])], label='L-BFGS: Data - PDE Gap', color='red')
    axs[2].set_title('Loss Imbalance (Data - PDE)')
    axs[2].legend()

    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/milestone8_optimizer.png')
    print("Plot saved to results/milestone8_optimizer.png")

if __name__ == "__main__":
    main()
