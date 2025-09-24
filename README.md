# Deep Generative Models — Course Materials (DGM)

This repository collects lecture slides, assignments (CAs), code notebooks, reports, and reference papers used in the "Deep Generative Models" course (University of Tehran). The materials are organized to be reproducible and educational: each assignment contains an annotated Jupyter notebook, supporting code, and a report.

This `README.md` is intentionally long and detailed. It documents the repository layout, the goals of each assignment notebook, recommended environment and reproducibility steps, and specific notes for CA2 (GANs and Normalizing Flows) which contains the notebook `CAs/CA2/code/CA2_DGM.ipynb` and its companion `CAs/CA2/README.md`.

## Table of contents

- Repository structure and purpose
- Quick start (setup & run)
- Notebooks and assignments (CA1..CA4) — summary and status
- Deep dive: CA2 (GANs & Normalizing Flows)
- Data, storage and artifact management
- Reproducibility checklist and recommended configuration
- Testing and lightweight smoke checks
- Common issues and troubleshooting
- References and further reading
- Credits and license

---

## Repository structure (top-level)

- `CAs/` — Course assignments. Each `CA#` typically contains:
  - `code/` — Jupyter notebooks and code used for experiments (e.g. `code.ipynb`, `CA2_DGM.ipynb`).
  - `description/` — Written answers and explanations required by the assignment.
  - `report/` — PDF report and figures.
  - `train/` — checkpoints, small prepared datasets, or saved outputs (if present).
- `Slides/` — Lecture slides and course material used in class.
- `papers/` — Research papers referenced during the course.
- `Exams/` — Past exams and solutions.
- `Extra/` — Misc utilities, templates, or exploratory notebooks.

This repository is primarily an educational resource. Notebooks are annotated for readability and (where possible) reorganized to centralize imports and configuration.

---

## Quick start — environment and running notebooks

Recommended steps to set up a local, reproducible environment (using `venv` and `pip`):

1. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install core dependencies (adjust PyTorch install for your CUDA version):

```bash
pip install -U pip
pip install torch torchvision matplotlib numpy scipy pytorch-fid jupyterlab
```

Notes:

- If you have a CUDA-enabled GPU, install the matching `torch`/`torchvision` binaries using the official instructions at https://pytorch.org.
- `pytorch-fid` is used in CA2 to compute the Frechet Inception Distance (FID). If you encounter installation issues, consider using a system package or an alternative FID implementation.

3. Launch JupyterLab from the project root and open the notebook you'd like to run:

```bash
jupyter lab
```

4. Important safety note: the notebooks under `CAs/` were edited as part of a documentation pass (imports consolidated, configuration cell added). The editorial pass did not execute the notebooks. Before running long training jobs, review the `Setup and Configuration` cell in each notebook and run smoke tests described below.

---

## Notebooks and assignments — short summary and status

This section summarizes the primary assignments and their current status in the repository.

- CA1 (folder: `CAs/CA1`)

  - Focus: Variational Autoencoders (VAE) and experiments exploring latent structure.
  - Key files: `code/code.ipynb`, `report/report.pdf`.
  - Status: Notebook reorganized (imports consolidated, configuration cell added), and a `README.md` was produced describing the experiments. The accompanying PDF report could not be reliably extracted verbatim in the editing environment; a synthesized summary was added to the CA1 README with a note about the limitation.
- CA2 (folder: `CAs/CA2`)

  - Focus: Normalizing flows (RealNVP) and GANs (DCGAN-style) applied to FashionMNIST. Includes OOD detection experiments (MNIST, KMNIST) and FID evaluation for GANs.
  - Key files: `code/CA2_DGM.ipynb`, `README.md` (created to document run instructions and reproducibility).
  - Status: Imports consolidated, configuration cell inserted, duplicate imports removed, explanatory Markdown blocks added. Notebook was not executed as part of the editorial pass.
- CA3, CA4 (folders present)

  - Each contains code, descriptions and reports. They follow a similar structure but may need manual documentation or editorial passes for publication-quality presentation.

Other folders (Slides, papers, Extra, Exams) contain lecture materials, relevant readings, and supporting documents.

---

## Deep dive: CA2 (GANs & Normalizing Flows)

CA2 is both pedagogical and experimental. It demonstrates two complementary approaches to deep generative modeling:

1. RealNVP (normalizing flows): an explicit density model trained by maximum likelihood. The notebook contains:

   - Implementation of coupling layers and RealNVP stacking.
   - Training using negative log-likelihood (NLL).
   - Computation of log-likelihoods for in-distribution and out-of-distribution (OOD) datasets (MNIST, KMNIST).
   - Visualization of generated samples via the inverse mapping.
2. GAN (DCGAN-style): an adversarial generator trained to produce realistic fashion images. The notebook contains:

   - DCGAN-style `Generator` and `Discriminator` classes implemented in PyTorch.
   - A training loop alternating generator and discriminator updates.
   - Fixed noise vectors to produce consistent image grids for visual progress.
   - FID evaluation using `pytorch-fid` computed per-epoch.

Why run CA2?

- RealNVP gives explicit densities and allows for direct OOD detection experiments based on log-likelihood.
- Training RealNVP in a learned latent space (via an encoder-decoder) reduces dimensionality and speeds up flow training.
- GAN training provides qualitative sample generation and a complementary evaluation via FID.

Files of interest in `CAs/CA2`:

- `code/CA2_DGM.ipynb` — the annotated notebook (imports consolidated and a configuration cell added).
- `README.md` — localized instructions, reproducibility notes and quick-start steps.

High-level suggested execution order (no code is run by the editor):

1. Edit the top `Setup and Configuration` cell to set `device`, `latent_dim`, `batch_size`, `epochs`, and `image_size`.
2. Run the data preparation cells to download datasets and build DataLoaders.
3. Train and evaluate RealNVP (or train RealNVP on learned latent representations after training the encoder-decoder).
4. Train the GAN and observe per-epoch outputs and FID metrics.

---

## Data, storage, and artifact management

- Datasets are downloaded by `torchvision` into `./data/` by default. Consider configuring a dataset cache directory if multiple users will run experiments on the same machine.
- Save model checkpoints and run metadata:
  - Save generator/discriminator weights (`.pth`) and RealNVP checkpoints.
  - Save a `run_info.json` for each experiment (hyperparameters, seed, timestamp, Git commit hash).
- Keep `real_images/` fixed when evaluating FID across multiple runs: create a reproducible reference set (e.g., 2k images sampled from the training set with fixed RNG) and reuse it across experiments.

---

## Reproducibility checklist and recommended configuration

- Use a virtual environment and pin package versions.
- Set random seeds for `python`, `numpy`, and `torch` at the top of each notebook's configuration cell.
- Record the Git commit hash used for each experiment. Example:

```bash
git rev-parse --short HEAD
```

- Save checkpoints frequently and store `run_info.json` alongside saved artifacts.
- When computing FID:
  - Use the same preprocessing pipeline (resize, normalization) for real and generated images.
  - Use the same Inception feature extraction dims across runs.

Suggested `run_info.json` schema (example):

```json
{
  "commit": "abc123",
  "seed": 42,
  "latent_dim": 100,
  "lr": 0.0002,
  "batch_size": 64,
  "epochs": 20,
  "image_size": 64,
  "notes": "DCGAN baseline run"
}
```

---

## Testing and smoke-checks

Before committing to long training, run small smoke tests:

- Shapes and forward pass tests

  - Assert the generator output shape and value ranges.
  - Assert that the discriminator produces a scalar per image.
  - Verify RealNVP forward returns `(z, log_det_jacobian)` and `inverse(z)` maps back to image-like shapes.
- Mini-training run

  - `batch_size=16`, `epochs=1`, `N=128` samples; verify logs, saves, and no runtime exceptions.
- FID sanity-check

  - Compute FID on a small set (e.g., 200 real vs 200 generated) to verify the pipeline; expect noisy values but end-to-end correctness.

---

## Common issues and troubleshooting

- `pytorch-fid` errors: ensure the package and its dependencies are installed and compatible with your `torch`/`torchvision` versions. If the provided function `calculate_fid_given_paths` errors, verify that the image directories are non-empty and properly preprocessed.
- CUDA / memory errors: reduce `batch_size` or `image_size`, or use mixed precision with `torch.cuda.amp`.
- FID values unexpectedly high: check that both real and generated images are preprocessed identically. Confirm `real_images/` contains the intended images.

---

## References and further reading

- Dinh, Laurent, Jascha Sohl-Dickstein, and Samy Bengio. "Density estimation using Real NVP." 2017.
- Goodfellow, Ian, et al. "Generative Adversarial Nets." 2014.
- Radford, Alec, Luke Metz, and Soumith Chintala. "Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks." 2015.
- Heusel, Martin, et al. "GANs trained by a two time-scale update rule converge to a local Nash equilibrium." 2017.

Many of the referenced papers are available in the `papers/` directory.

---

## Credits and license

This repository contains course materials for the Deep Generative Models course. The code and notebooks are intended for educational and research use. If you reuse code or figures derived from these materials in publications or public projects, please credit the course author and repository.

---
