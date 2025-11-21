# Figure Requirements for LaTeX Report

This document lists all the figures needed for the complete report compilation.

## Required Figures

Place all figures in the `figures/` directory with the following names:

### Question 1: Probabilistic Graphical Models

1. **bayesian_network.png** - Disease model Bayesian network showing M→I←S, I→D, I→T←F, T→D
2. **complex_network.png** - Complex network with C→O→{S,A}, {A,S}→T, T→{B,M}

### Question 2: VAE Implementation

3. **dsprites_samples.png** - Sample images from dSprites dataset showing shape, scale, orientation variations
4. **training_loss.png** - Total loss curve over epochs
5. **recon_kl_loss.png** - Reconstruction loss vs KL divergence loss over training
6. **reconstructions.png** - Original images (top row) vs reconstructions (bottom row)
7. **latent_space_pca.png** - PCA projection of latent space colored by ground truth factors
8. **beta_comparison.png** - Side-by-side comparison of reconstructions for β=1, β=2, β=5
9. **latent_traversal.png** - Grid showing latent space traversal for each dimension

## Generating Figures from Notebook

Run the notebook `code.ipynb` to generate all figures. The code includes:

- Visualization functions that save figures automatically
- Training plots that export to PNG
- Reconstruction comparisons
- Latent space analysis

## Compilation Instructions

```bash
cd /Users/tahamajs/Documents/uni/DGM/V2/CA1/report
pdflatex report.tex
pdflatex report.tex  # Run twice for references
```

Or use:
```bash
latexmk -pdf report.tex
```

## Note on Missing Figures

If figures are not yet generated:
- Run the notebook cells sequentially
- Figures will be saved in the `figures/` directory
- Alternatively, you can use placeholder images or comment out the `\includegraphics` lines temporarily

## Table of Contents

The report includes:
- Question 1: PGMs (Bayesian & Markov Networks)
- Question 2: VAEs (Theory, Implementation, β-VAE, MIG, Advanced Variants)
- Complete mathematical derivations
- Algorithm pseudocode
- Experimental results and discussion
