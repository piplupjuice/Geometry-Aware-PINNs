# Learning Log

### Phase 1: PINN Mechanics
- Mastered automatic differentiation in PyTorch (`torch.autograd.grad`).
- Understood the failure modes of standard Adam when applied to multi-objective loss landscapes.

### Phase 2: Diagnostics
- Discovered that spectral bias prevents the network from learning high-frequency components of the PDE.
- Implemented `GeometryScorecard` to rigorously quantify these effects using K-FAC condition numbers.

### Phase 3: Repository Structuring
- Re-architected raw Jupyter experiments into a modular, production-ready codebase.
- Solidified Python packaging and reproducibility best practices.
