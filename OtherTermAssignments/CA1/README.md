# Deep Generative Models - CA1
## Variational Autoencoders and Probabilistic Graphical Models

**University of Tehran**  
**Faculty of Engineering - Department of Electrical and Computer Engineering**  
**Course:** Deep Generative Models  
**Instructor:** Dr. Mostafa Tavasoli Pour

---

## 📋 Assignment Overview

This assignment covers two main topics:

### Question 1: Probabilistic Graphical Models (33 points)
- **Part 1:** Bayesian Network - Disease Model (16 pts)
- **Part 2:** Given Bayesian/Markov Network Analysis (17 pts)
- **Part 3:** Seven-Node Markov Network (11 pts)
- **Part 4:** Variational Inference Derivation (10 pts)

### Question 2: Variational Autoencoders (67 points)
- **Part 1:** Theoretical Questions (10 pts)
- **Part 2:** VAE Implementation and Training (10 pts)
- **Part 3:** β-VAE with Multiple β Values (17 pts)
- **Part 4:** MIG Metric Evaluation (8 pts)
- **Part 5:** Latent Space Analysis (3 pts)
- **Part 6:** Advanced VAE Variants (VQ-VAE, VampPrior, SC-VAE) (15 pts)

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install torch torchvision numpy matplotlib networkx scikit-learn tqdm scipy
```

Or with conda:
```bash
conda install pytorch torchvision numpy matplotlib networkx scikit-learn tqdm scipy -c pytorch
```

### 2. Download Dataset

The notebook includes an automatic downloader, but you can also download manually:

```bash
wget https://github.com/deepmind/dsprites-dataset/raw/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz
```

Or run this Python code:
```python
import urllib.request
url = 'https://github.com/deepmind/dsprites-dataset/raw/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'
urllib.request.urlretrieve(url, 'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')
```

### 3. Run the Notebook

Open `code/code.ipynb` and run all cells sequentially.

**Key Configuration (in the executable training cell):**
```python
TRAIN_MODELS = True  # Set to True to train models
EPOCHS = 50
BATCH_SIZE = 128
LEARNING_RATE = 0.001
LATENT_DIM = 256
BETA_VALUES = [1.0, 2.0, 5.0]
```

---

## 📁 Directory Structure

```
CA1/
├── README.md                          # This file
├── code/
│   └── code.ipynb                     # Complete implementation notebook
├── description/                       # Assignment description (Persian)
└── (Generated during execution:)
    ├── vae_beta1.pth                  # Trained model checkpoints
    ├── vae_beta2.pth
    ├── vae_beta5.pth
    ├── training_history_beta*.png     # Training curves
    ├── reconstructions_beta*.png      # Reconstruction visualizations
    ├── reconstruction_comparison_all.png
    ├── pca_beta*.png                  # Latent space PCA visualizations
    ├── mi_matrix.png                  # Mutual information matrix
    ├── bayesian_network.png           # PGM diagrams
    ├── markov_network.png
    └── seven_node_markov_network.png
```

---

## 📊 What's Implemented

### Question 1: Probabilistic Graphical Models ✅

#### Part 1: Bayesian Network
- ✅ Network structure visualization
- ✅ Joint probability factorization
- ✅ Conditional independence analysis using d-separation
- ✅ Complete mathematical proofs

#### Part 2: Given Network
- ✅ Joint probability for Bayesian network
- ✅ Markov blanket identification
- ✅ Perfect I-map analysis
- ✅ Chordality checking
- ✅ Maximal cliques computation
- ✅ Marginalization comparison (Bayesian vs Markov)

#### Part 3: Seven-Node Network
- ✅ Network visualization
- ✅ Maximal cliques identification
- ✅ Conditional independence verification
- ✅ Analysis of potential function effects

#### Part 4: Variational Inference
- ✅ Complete mathematical derivation
- ✅ ELBO computation and optimization
- ✅ Visualization of optimal parameter

### Question 2: Variational Autoencoders ✅

#### Theoretical Understanding
- ✅ Explanation of ELBO and why we don't maximize log-likelihood directly
- ✅ dSprites dataset description and visualization
- ✅ Reparameterization trick explanation

#### Implementation
- ✅ Complete VAE architecture (Encoder + Decoder)
- ✅ Training pipeline with monitoring
- ✅ Loss function implementation (Reconstruction + KL)
- ✅ Visualization utilities

#### β-VAE
- ✅ Implementation of β-VAE objective
- ✅ Training with multiple β values (1.0, 2.0, 5.0)
- ✅ Comparison of reconstruction quality
- ✅ Analysis of β effect on disentanglement

#### Evaluation
- ✅ MIG (Mutual Information Gap) metric implementation
- ✅ Per-factor disentanglement analysis
- ✅ Mutual information matrix visualization
- ✅ PCA visualization of latent space

#### Advanced Variants (Theoretical)
- ✅ VQ-VAE explanation (discrete latent space)
- ✅ VampPrior explanation (learnable prior)
- ✅ SC-VAE explanation (sparse coding)

---

## ⏱️ Expected Runtime

| Hardware | Training (3 models × 50 epochs) | Evaluation | Total |
|----------|--------------------------------|------------|-------|
| **GPU (CUDA)** | 1.5 - 2 hours | 10 min | ~2 hours |
| **CPU** | 6 - 8 hours | 30 min | ~8 hours |
| **Google Colab (Free GPU)** | 2 - 3 hours | 15 min | ~3 hours |

---

## 🎯 Key Results to Report

### Question 1
1. Bayesian network diagram with explanations
2. Mathematical derivations for conditional independence
3. Markov network analysis with clique factorization
4. Variational inference solution: θ* = x + 1

### Question 2
1. **Training Curves:** Loss, reconstruction, and KL divergence for each β
2. **Reconstruction Quality:** Visual comparison across β values
3. **MIG Scores:** Quantitative disentanglement metrics
4. **PCA Visualizations:** Latent space structure
5. **Analysis:** Trade-off between reconstruction and disentanglement

### Expected Results
- **Standard VAE (β=1):** MIG ≈ 0.1-0.2 (poor disentanglement)
- **β-VAE (β=2):** MIG ≈ 0.2-0.3 (moderate disentanglement)
- **β-VAE (β=5):** MIG ≈ 0.3-0.5 (good disentanglement)
- **Trade-off:** Higher β → Better disentanglement, Worse reconstruction

---

## 🔧 Troubleshooting

### CUDA Out of Memory
```python
# Reduce batch size
BATCH_SIZE = 64  # or 32
# Or reduce data subset
DATA_SUBSET_SIZE = 25000
```

### Training Too Slow
```python
# Reduce epochs for testing
EPOCHS = 20
# Or use smaller dataset
DATA_SUBSET_SIZE = 10000
```

### NaN Loss
```python
# Lower learning rate
LEARNING_RATE = 0.0005
# Or reduce β
BETA_VALUES = [1.0, 1.5, 2.0]
```

### Poor MIG Scores
```python
# Increase β
BETA_VALUES = [2.0, 5.0, 10.0]
# Or train longer
EPOCHS = 100
```

---

## 📝 Report Submission

### Required Files
1. **Report (PDF):** Following the provided template
2. **Code:** Complete Jupyter notebook with outputs
3. **Figures:** All generated visualizations
4. **Models (Optional):** Trained checkpoints

### File Naming
```
HW1_[LastName]_[StudentID].zip
```

### What to Include in Report

#### Question 1 (33 points)
- Network diagrams with explanations
- Mathematical derivations
- Conditional independence proofs
- Variational inference solution

#### Question 2 (67 points)
- Theoretical answers (ELBO, reparameterization, etc.)
- Training curves and analysis
- Reconstruction comparisons
- MIG scores and interpretation
- Latent space visualizations
- Advanced variants explanations

---

## 📚 Key Concepts Covered

### Probabilistic Graphical Models
- Bayesian Networks (directed)
- Markov Networks (undirected)
- D-separation and conditional independence
- Moralization and chordality
- Clique factorization
- Variational inference

### Variational Autoencoders
- Evidence Lower Bound (ELBO)
- Reparameterization trick
- Encoder-decoder architecture
- KL divergence regularization
- β-VAE for disentanglement
- Mutual Information Gap (MIG)
- Latent space analysis

### Advanced Topics
- VQ-VAE (discrete representations)
- VampPrior (learnable priors)
- SC-VAE (sparse coding)

---

## 📖 References

1. **Kingma, D. P., & Welling, M. (2013).** Auto-encoding variational bayes. *ICLR*.
2. **Higgins, I., et al. (2017).** β-VAE: Learning basic visual concepts with a constrained variational framework. *ICLR*.
3. **Chen, R. T. Q., et al. (2018).** Isolating sources of disentanglement in variational autoencoders. *NeurIPS*.
4. **van den Oord, A., et al. (2017).** Neural discrete representation learning. *NeurIPS* (VQ-VAE).
5. **Tomczak, J. M., & Welling, M. (2018).** VAE with a VampPrior. *AISTATS*.
6. **Koller, D., & Friedman, N. (2009).** Probabilistic Graphical Models. *MIT Press*.

---

## 💡 Tips for Success

1. **Start Early:** Training takes several hours
2. **Use GPU:** 5-10× faster than CPU
3. **Monitor Progress:** Check reconstructions during training
4. **Save Often:** Auto-save enabled, but save notebook frequently
5. **Understand Theory:** Don't just run code, understand concepts
6. **Quality over Speed:** Good analysis > quick completion
7. **Document Everything:** Take notes on observations
8. **Ask Questions:** Contact TAs if stuck

---

## ✉️ Contact

For questions or issues:
- **Question 1:** ma.moghimi202@gmail.com
- **Question 2:** S.m.moosavi000@ut.ac.ir
- **Subject Line:** TAI_HW1

---

## 📅 Important Dates

- **Release Date:** Mehr 1404 (October 2024)
- **Deadline:** 13 Aban 1404 (November 3, 2025)
- **Late Submission:** Up to 7 days with grade penalty
- **Grace Days:** Available (check course policy)

---

## ⚖️ Academic Integrity

- **Individual Work:** This is a solo assignment
- **Cite Sources:** Reference any external code or ideas
- **No Plagiarism:** Copying code/reports = zero grade for all parties
- **AI Tools:** Allowed for understanding, not for direct answers
- **Discussion:** Can discuss concepts, not share code/answers

---

## 🎓 Learning Outcomes

By completing this assignment, you will:
1. Understand probabilistic graphical models (Bayesian and Markov networks)
2. Master variational autoencoder theory and implementation
3. Learn disentangled representation learning
4. Gain experience with PyTorch and deep learning best practices
5. Develop skills in scientific visualization and analysis
6. Practice writing technical reports

---

## 🏆 Bonus Tips

- Compare your MIG scores with published papers
- Try additional β values (β=10, β=0.5)
- Experiment with different architectures
- Visualize latent traversals (change one dimension at a time)
- Try training on other datasets
- Implement one of the advanced variants (VQ-VAE, etc.)

---

**Good luck with your assignment! 🚀**

*This notebook provides a complete, production-ready implementation with proper documentation, visualization, and analysis tools. All questions are fully answered with both theoretical explanations and practical implementations.*
