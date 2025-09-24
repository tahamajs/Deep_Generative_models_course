# CA2 - GANs and Normalizing Flows (FashionMNIST)

This folder contains an educational project for the "Deep Generative Models" course. The notebook `CA2_DGM.ipynb` implements two main families of generative models on the FashionMNIST dataset:

- RealNVP (normalizing flows) trained on raw images and later on learned latent representations.
- GAN (DCGAN-style) trained to generate 64x64 versions of FashionMNIST images, with FID evaluation.

This README documents the notebook structure, configuration, how to run experiments (without running code here), reproducibility notes, and troubleshooting tips.

## Repository layout

- `code/CA2_DGM.ipynb` — The main notebook (annotated). Edits were made to consolidate imports and add a configuration cell. Each code block should be preceded by explanatory Markdown cells.
- `data/` — (created at runtime) where datasets are downloaded by torchvision.
- `real_images/` — directory used to store a fixed set of real images for FID computation.
- `generated_images/` — directory where per-epoch generated images are saved for FID computation and visualization.

## Notebook Overview

1. Setup and Configuration
   - Consolidated imports and a single configuration cell (`device`, hyperparameters, paths, image size).
2. Data Preparation
   - FashionMNIST is downloaded and preprocessed. For GAN training images are resized to `64x64` and normalized to `[-1, 1]`.
3. RealNVP (Normalizing Flow)
   - RealNVP is implemented with coupling layers and trained by maximizing log-likelihood (minimizing negative log-likelihood).
   - The notebook includes functions to compute log-likelihoods and visualize generated samples.
   - Out-of-distribution detection is evaluated using MNIST and KMNIST datasets.
4. Encoder-Decoder (Latent Space)
   - A small encoder-decoder (MLP) is trained to produce 50-dimensional latent representations of images.
   - RealNVP is also trained in this latent space to improve efficiency.
5. GAN (DCGAN-style)
   - A DCGAN-style generator and discriminator are implemented.
   - Training loop alternates between generator and discriminator updates.
   - Fixed noise is used to generate consistent grids for visualization.
   - FID scores are computed per epoch using `pytorch-fid`.

## How to run (local environment instructions)

1. Create a Python environment and install dependencies (suggested):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install torch torchvision matplotlib numpy scipy pytorch-fid
```

2. Open the notebook in Jupyter or VS Code and review the top `Setup and Configuration` cell before executing any cells. The editorial pass performed by the instructor/assistant did not run code; it only reorganized the notebook.

3. To run the GAN training (example):
   - Ensure you have a CUDA-capable GPU for reasonable training times.
   - Execute the configuration cell and then the GAN training cell blocks in order.
   - Generated images and FID scores will be saved in `./generated_images/`.

## Reproducibility Notes

- Random seeds are not set in the notebook by default. For reproducible runs, set seeds for `numpy`, `torch`, and Python's `random` at the top of the configuration cell:

```python
import random
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
```

- Batch size and learning rates are defined in the configuration cell and should be adjusted according to available compute.

## Editorial changes made

- Consolidated imports into a single top cell.
- Inserted a `Setup and Configuration` code cell with hyperparameters and paths.
- Removed duplicate import statements found later in the notebook.
- Added explanatory Markdown cells and ensured major code blocks are documented. Code comments inside code cells were left for the user to remove if required; the editorial pass prioritized not executing the notebook and organizing cells.

## Troubleshooting

- If `pytorch-fid` fails during FID calculation, ensure `torchvision` and `pytorch-fid` versions are compatible with your PyTorch installation.
- FID calculation may require a reasonable number of real images (suggest 1000+). Use `real_images/` to store a fixed set of samples.

## License & Acknowledgments

This work is an academic project for a course (Deep Generative Models). The code uses standard PyTorch patterns and public datasets (FashionMNIST, MNIST, KMNIST).

For issues or suggestions, open an issue in the repository or contact the notebook owner.
