
# Complex_Modulation

This repository provides reference implementations and test scripts for complex modulation (amplitude and phase) using a phase-only spatial light modulator (SLM).

## Overview

The codebase focuses on Computer-Generated Holography (CGH) algorithms that enable the encoding of arbitrary complex fields (amplitude and phase) onto a phase-only SLM. The main approach implemented is the **Bolduc exact encoding method**:

- **Bolduc.py**: Demonstrates analytic target generation (e.g., Laguerre-Gaussian modes) and exact encoding.
- **Bolduc_e_holo.py**: Enables encoding of arbitrary far-field intensity patterns from BMP images (e.g., logos, text, experimental data).

All scripts use physical coordinates (meters) for direct experimental relevance and reproducibility.

## Features

- Exact complex-field encoding with a phase-only hologram (Bolduc method)
- Support for both analytic and image-based targets
- Physical coordinate system throughout
- Modular utility functions for grid generation, masking, and visualization

## Getting Started

1. Clone the repository
2. Install requirements (numpy, matplotlib, Pillow)
3. Run the example scripts:
	- `Bolduc.py` for analytic targets
	- `Bolduc_e_holo.py` for BMP image targets

## Citation

If you use this code or the Bolduc method in your research, please cite:

> P. Bolduc et al., "Exact solution to simultaneous intensity and phase encryption with a single phase-only hologram," Opt. Lett. 38, 3546 (2013).

---
For questions or contributions, please contact Marco Astarita.
