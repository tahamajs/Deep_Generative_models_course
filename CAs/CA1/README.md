

# **Deep Generative Models - Homework 1**

**University of Tehran** | **Department of Electrical and Computer Engineering**

 **Course** : Deep Generative Models |  **Instructor** : Dr. Mostafa Tavasoli |  **Term** : Fall 1403

 **Author** : *Taha Majlesi*

 **Email** : [taha.maj4@gmail.com](mailto:taha.maj4@gmail.com) | [tahamajlesi@ut.ac.ir](mailto:tahamajlesi@ut.ac.ir)

 **Profiles** : [LinkedIn](https://www.linkedin.com/in/tahamajlesi/) | [GitHub](https://github.com/tahamajs) | [Hugging Face](https://huggingface.co/tahamajs/plamma)

---

## **Table of Contents**

* [Introduction](https://chatgpt.com/c/67b0eae8-c268-8007-9df3-33d28ab21913#introduction)
* [Course Information](https://chatgpt.com/c/67b0eae8-c268-8007-9df3-33d28ab21913#course-information)
* [Assignment Details](https://chatgpt.com/c/67b0eae8-c268-8007-9df3-33d28ab21913#assignment-details)
* [Sections Overview](https://chatgpt.com/c/67b0eae8-c268-8007-9df3-33d28ab21913#sections-overview)
  * [Probabilistic Graphical Models (PGM)](https://chatgpt.com/c/67b0eae8-c268-8007-9df3-33d28ab21913#probabilistic-graphical-models-pgm)
  * [Variational Autoencoders (VAE)](https://chatgpt.com/c/67b0eae8-c268-8007-9df3-33d28ab21913#variational-autoencoders-vae)
* [Implementation Details](https://chatgpt.com/c/67b0eae8-c268-8007-9df3-33d28ab21913#implementation-details)
* [Mathematical Derivations](https://chatgpt.com/c/67b0eae8-c268-8007-9df3-33d28ab21913#mathematical-derivations)
* [Training and Experimentation](https://chatgpt.com/c/67b0eae8-c268-8007-9df3-33d28ab21913#training-and-experimentation)
* [Results and Analysis](https://chatgpt.com/c/67b0eae8-c268-8007-9df3-33d28ab21913#results-and-analysis)
* [Submission Guidelines](https://chatgpt.com/c/67b0eae8-c268-8007-9df3-33d28ab21913#submission-guidelines)
* [License](https://chatgpt.com/c/67b0eae8-c268-8007-9df3-33d28ab21913#license)
* [Project Structure](https://chatgpt.com/c/67b0eae8-c268-8007-9df3-33d28ab21913#project-structure)

---

## **Introduction**

This repository contains Homework 1 for the Deep Generative Models course at the University of Tehran. The assignment includes both theoretical derivations and practical implementations in:

* Probabilistic Graphical Models (PGMs): Bayesian Networks, Markov Networks, and inference techniques.
* Variational Autoencoders (VAEs): ELBO, KL Divergence, reparameterization, and deep generative modeling.

The goal is to build a solid theoretical foundation and gain hands-on experience with deep generative models.

---

## **Course Information**

* **University** : University of Tehran
* **Department** : Electrical and Computer Engineering
* **Course** : Deep Generative Models
* **Instructor** : Dr. Mostafa Tavasoli
* **Term** : Fall 1403

---

## **Assignment Details**

The assignment is divided into two sections:

### **1. Probabilistic Graphical Models (PGM)**

* Constructing Bayesian Networks and Markov Networks.
* Deriving conditional independence properties.
* Implementing variational inference for Bayesian Networks.

### **2. Variational Autoencoders (VAE)**

* Theoretical Derivations: ELBO, KL Divergence, reparameterization.
* Implementation: Building and training a VAE.
* Experiments: Latent space visualization, disentanglement, and β-VAE.

---

## **Sections Overview**

### **Probabilistic Graphical Models (PGM)**

* Bayesian Networks: Construct a DAG for a server monitoring system and compute joint probability distributions.
* Conditional Independence: Use d-separation to verify independence properties.
* Markov Networks: Analyze and factorize a given Markov network.
* Variational Inference: Implement a Bayesian inference technique.

### **Variational Autoencoders (VAE)**

* ELBO Derivation: Prove the Evidence Lower Bound (ELBO).
* KL Divergence: Show its role in regularizing latent space.
* Reparameterization Trick: Enable gradient-based optimization.
* VAE Implementation: Encoder-Decoder structure with CNN layers.
* Training and Experimentation: Evaluate reconstruction loss, KL divergence, and latent space visualization.
* β-VAE and Disentanglement: Modify β to analyze factorized latent representations.

---

## **Implementation Details**

### **Dataset**

* Face Image Dataset (preprocessed for training).
* Train/Test Split: 80/20.
* Preprocessing:
  * Convert to grayscale (if needed).
  * Resize to 128×128.
  * Normalize pixel values to `[0,1]`.

### **Model Architecture**

| Component         | Layers Used                                                  |
| ----------------- | ------------------------------------------------------------ |
| **Encoder** | Conv2D (ReLU) → Conv2D → Flatten → Dense (Latent Space)   |
| **Decoder** | Dense → Reshape → Conv2D Transpose (ReLU) → Conv2D Output |

### **Training Parameters**

| Parameter               | Value     |
| ----------------------- | --------- |
| **Image Size**    | (128,128) |
| **Batch Size**    | 128       |
| **Optimizer**     | Adam      |
| **Learning Rate** | 0.0005    |
| **Epochs**        | 1000      |

### **Loss Function**

* ELBO Loss: Combination of Reconstruction Loss and KL Divergence.
* β-VAE Loss: Adds a weight term `β` to KL divergence for better disentanglement.

---

## **Mathematical Derivations**

### **1. ELBO Derivation**

The ELBO (Evidence Lower Bound) is derived from Bayes' Theorem:

$$
\log p(x) = \mathbb{E}_{q(z|x)} \left[ \log p(x|z) \right] - D_{KL} \left( q(z|x) \| p(z) \right)
$$

**2. Reparameterization Trick**

Sampling from q(z∣x) is non-differentiable, so we use:

$$
z = \mu + \sigma \cdot \epsilon, \quad \text{where } \epsilon \sim \mathcal{N}(0,1)
$$

This enables gradient-based training.

---

## **Training and Experimentation**

1. Train the VAE for 1000 epochs.
2. Monitor ELBO loss and KL divergence during training.
3. Visualize latent space using interpolations.
4. Modify latent vectors to control face attributes.

---

## **Results and Analysis**

* Loss Curves: Track ELBO, Reconstruction Loss, and KL Divergence.
* Latent Space Analysis: Visualize interpolation between latent vectors.
* Disentanglement Study: Modify β-VAE parameters for better latent separation.
* Generated Images: Show example outputs.

---

## **Submission Guidelines**

1. **Report** : Submit a well-structured PDF including:

* Theoretical answers
* Mathematical derivations
* Experimental results

1. **Code** :

* Provide fully executable Python scripts.
* Use Jupyter notebooks for analysis and visualization.

1. **Submission Format** :

* Upload the project to GitHub.
* Provide a clear README.md.
* Ensure the code runs without errors.

---

## **License**

This project is licensed under the MIT License.

For more details, see the [LICENSE](https://chatgpt.com/c/LICENSE) file.

---

## **Project Structure**

```
DeepGenModels_HW1/
│── README.md               # Overview of the repository
│── LICENSE                 # License file
│── .gitignore              # Files to ignore in Git tracking
│── requirements.txt        # Dependencies list
│── data/                   # Dataset (or links to download)
│── src/                    # Source code for models
│    ├── model.py           # VAE implementation
│    ├── train.py           # Training script
│    ├── inference.py       # Generating new samples
│    ├── utils.py           # Helper functions
│── notebooks/              # Jupyter notebooks for analysis
│── results/                # Training logs, images, plots
│── docs/                   # Additional documentation
│── report/                 # Final report and derivations
│── submission/             # Packaged submission files
```
