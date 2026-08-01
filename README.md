# Geometry-Aware Optimization Diagnostics for Physics-Informed Neural Networks

## Title and Abstract
**Geometry-Aware Optimization Diagnostics for Physics-Informed Neural Networks**

This repository constitutes a rigorous research documentation of my learning journey exploring Physics-Informed Neural Networks (PINNs) through the lens of Riemannian Geometry, Kronecker-Factored Approximate Curvature (K-FAC), and Spectral Bias. The culmination of this study is the **Geometry Scorecard**, a set of diagnostic metrics designed to quantify optimization pathologies in PINNs.

## Motivation & Learning Objectives
- The implementation of PINNs.
- Exploration of Loss Landscapes and Optimizer Dynamics.
- Study of the Hessian and Fisher Information Matrices.
- Construction of a reproducible, research-grade open-source repository.

## Repository Structure
- `src/`: Reusable classes and functions with PEP8 docstrings and typing.
- `experiments/`: Extracted Python scripts mapping to key milestones (e.g., Hessian Spectrum, K-FAC).
- `assets/`: Generated visualizations and loss landscapes.

## Mathematical Background
Physics-Informed Neural Networks frequently suffer from optimization difficulties known as *Gradient Pathology*. This can be modeled by analyzing the local geometry of the loss landscape, specifically utilizing the:
* **Hessian Matrix:** Evaluates curvature.
* **Fisher Information Matrix:** Analyzes parameter influence constraints.
* **Gradient Interference:** Computed via cosine similarity between PDE and Data loss gradients.

## The Geometry Scorecard
To diagnose these issues, I implemented the *Geometry Scorecard*, which evaluates:
1. **Condition Number:** Identifying severe ill-conditioning via Riemannian Geometry.
2. **Cosine Similarity:** Flagging conflicting gradient updates.
3. **Spectral Ratio:** Highlighting Spectral Bias (where low frequencies are prioritized over high frequencies).

## Visualizations
### 3D Spatio-Temporal Comparison
<p align="center">
  <img src="assets/3d_spatiotemporal.png" alt="3D Spatio-Temporal Comparison" width="800"/>
</p>

### 3D Loss Landscape
<p align="center">
  <img src="assets/3d_loss_landscape.png" alt="3D Loss Landscape" width="600"/>
</p>

## Reproducibility
The code within `src/` and `experiments/` is thoroughly tested and runs on a single Kaggle T4 GPU using PyTorch. See `requirements.txt` for dependencies.

## Citation and License
If you find this learning repository helpful, feel free to use it.
Licensed under the MIT License.
