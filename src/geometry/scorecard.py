"""
Geometry Scorecard diagnostic metrics for PINNs.
"""

import torch
import numpy as np
from src.pinn.model import compute_derivatives

class GeometryScorecard:
    def __init__(self, model, history, x_f, t_f, nu, x_grid, t_eval, device=None):
        self.model = model
        self.history = history
        self.x_f = x_f
        self.t_f = t_f
        self.nu = nu
        self.x_grid = x_grid
        self.t_eval = t_eval
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        # Storage for K-FAC hooks
        self.activations = {}
        self.grad_outputs = {}
        
    def _hook_fn(self, module, input, output):
        self.activations[module] = input[0].detach()
        
    def _grad_hook_fn(self, module, grad_input, grad_output):
        self.grad_outputs[module] = grad_output[0].detach()

    def compute_kfac_condition(self):
        # Register hooks
        self.model.fc1.register_forward_hook(self._hook_fn)
        self.model.fc1.register_full_backward_hook(self._grad_hook_fn)
        
        # Forward and backward pass to trigger hooks
        self.model.zero_grad()
        u_pred_f, u_t, u_x, u_xx = compute_derivatives(self.model, self.x_f, self.t_f)
        r = u_t + u_pred_f * u_x - self.nu * u_xx
        loss = torch.mean(r**2)
        loss.backward()
        
        # Compute B matrix
        layer = self.model.fc1
        delta = self.grad_outputs[layer]
        B = (delta.T @ delta) / delta.shape[0]
        
        eig_B = torch.linalg.eigvalsh(B)
        eig_B = eig_B[eig_B > 1e-8]
        
        if len(eig_B) == 0:
            return float('nan')
        return (torch.max(eig_B) / torch.min(eig_B)).item()

    def compute_spectral_ratio(self):
        x_tensor = torch.tensor(self.x_grid, dtype=torch.float32).reshape(-1, 1).to(self.device)
        t_tensor = torch.full_like(x_tensor, self.t_eval)
        
        with torch.no_grad():
            u_pred = self.model(x_tensor, t_tensor).cpu().numpy().flatten()
            
        fft_vals = np.fft.fft(u_pred)
        ps = np.abs(fft_vals)**2
        
        # Low frequencies: k=0, 1. High frequencies: k > 1
        low_freq_power = np.sum(ps[:2])
        high_freq_power = np.sum(ps[2:])
        
        if high_freq_power < 1e-12:
            return float('inf')
        return low_freq_power / high_freq_power

    def generate_report(self):
        print("="*50)
        print("GEOMETRY SCORECARD FOR PINN")
        print("="*50)
        
        # 1. Final Losses
        final_data = self.history['data'][-1]
        final_pde = self.history['pde'][-1]
        print(f"1. Final Data Loss: {final_data:.4e}")
        print(f"   Final PDE Loss:  {final_pde:.4e}")
        if final_pde > 10 * final_data:
            print("   [!] DIAGNOSIS: Loss Imbalance (PDE >> Data)")
            
        # 2. Gradient Pathology
        min_cos_sim = min(self.history['cos_sim'])
        print(f"\n2. Min Gradient Cosine Similarity: {min_cos_sim:.4f}")
        if min_cos_sim < 0.1:
            print("   [!] DIAGNOSIS: Gradient Pathology (Conflicting gradients)")
            
        # 3. Curvature (K-FAC)
        cond_B = self.compute_kfac_condition()
        print(f"\n3. K-FAC Condition Number (Layer 1, B): {cond_B:.4e}")
        if cond_B > 1e4:
            print("   [!] DIAGNOSIS: Riemannian Pathology (Extreme ill-conditioning)")
            
        # 4. Spectral Bias
        spec_ratio = self.compute_spectral_ratio()
        print(f"\n4. Spectral Ratio (Low/High Freq Power): {spec_ratio:.4e}")
        if spec_ratio > 1e2:
            print("   [!] DIAGNOSIS: Severe Spectral Bias (High frequencies ignored)")
            
        print("="*50)
