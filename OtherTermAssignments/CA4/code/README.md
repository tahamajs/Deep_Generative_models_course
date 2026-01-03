# Deep Generative Models - Homework 4
# Complete Implementation: Diffusion Models, Stable Diffusion, and Flow Matching

This directory contains the **complete** separated Python files extracted from the main_code.ipynb notebook.

## Files:

- `utils.py`: Common imports, device setup, and utility functions
- `ddpm.py`: DDPM and DDIM classes and functions for image generation
- `stable_diffusion.py`: Stable Diffusion and DreamBooth classes and functions
- `flow_matching.py`: **Complete** Flow Matching implementation with all evaluation functions
- `run_ddpm.py`: Runner script for DDPM training and sampling
- `run_stable_diffusion.py`: Runner script for DreamBooth training
- `run_flow_matching.py`: Runner script for Flow Matching training and evaluation

## ✅ Complete Extraction Status:

- **DDPM**: All classes, functions, and visualization code extracted
- **Stable Diffusion**: All DreamBooth classes and functions extracted
- **Flow Matching**: All classes, training, sampling, AND comprehensive evaluation code extracted
- **Results Saving**: All `save_fig()` calls and evaluation reports included
- **Runner Scripts**: Separate execution scripts for clean separation of concerns

## Usage:

The main files contain only class and function definitions. Use the runner scripts to execute:

```bash
python run_ddpm.py              # Run DDPM training and sampling
python run_stable_diffusion.py  # Run DreamBooth (requires diffusers)
python run_flow_matching.py     # Run Flow Matching for time series
```

Or import and use the classes directly:

```python
from ddpm import DDPMScheduler, DDPMTrainer
from stable_diffusion import DreamBoothTrainer
from flow_matching import FlowMatchingTrainer
```

## Requirements:

- PyTorch
- torchvision
- numpy, matplotlib, tqdm
- scipy
- Optional: diffusers, transformers, peft, accelerate (for Stable Diffusion)
- Optional: yfinance, pandas (for Flow Matching data)

## Parts:

1. **DDPM & DDIM**: Denoising Diffusion Probabilistic Models for CIFAR-10 image generation
2. **Stable Diffusion & DreamBooth**: Latent diffusion with fine-tuning for subject-specific generation
3. **Flow Matching**: Continuous generative model for financial time series generation

Each implementation includes training, sampling, and comprehensive evaluation.