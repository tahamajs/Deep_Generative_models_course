# **Advanced Generative Models - Homework 2**

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
  * [Flow-Based Generative Models](https://chatgpt.com/c/67b0eae8-c268-8007-9df3-33d28ab21913#flow-based-generative-models)
  * [Generative Adversarial Networks (GANs)](https://chatgpt.com/c/67b0eae8-c268-8007-9df3-33d28ab21913#generative-adversarial-networks-gans)
* [Implementation Details](https://chatgpt.com/c/67b0eae8-c268-8007-9df3-33d28ab21913#implementation-details)
* [Mathematical Derivations](https://chatgpt.com/c/67b0eae8-c268-8007-9df3-33d28ab21913#mathematical-derivations)
* [Training and Experimentation](https://chatgpt.com/c/67b0eae8-c268-8007-9df3-33d28ab21913#training-and-experimentation)
* [Results and Analysis](https://chatgpt.com/c/67b0eae8-c268-8007-9df3-33d28ab21913#results-and-analysis)
* [Submission Guidelines](https://chatgpt.com/c/67b0eae8-c268-8007-9df3-33d28ab21913#submission-guidelines)
* [License](https://chatgpt.com/c/67b0eae8-c268-8007-9df3-33d28ab21913#license)
* [Project Structure](https://chatgpt.com/c/67b0eae8-c268-8007-9df3-33d28ab21913#project-structure)

---

## **Introduction**

This repository contains Homework 2 for the Deep Generative Models course at the University of Tehran. This assignment covers  **advanced generative models** , including:

* **Flow-based models** (e.g., RealNVP, Residual Flows)
* **Generative Adversarial Networks (GANs)**
* **Wasserstein GAN (WGAN)** and improvements over standard GANs
* **Likelihood-based out-of-distribution (OOD) detection**
* **Mathematical derivations and deep learning implementations**

By completing this assignment, students will gain hands-on experience with  **modern generative models** , understand their strengths and weaknesses, and implement them in Python.

---

## **Course Information**

* **University** : University of Tehran
* **Department** : Electrical and Computer Engineering
* **Course** : Deep Generative Models
* **Instructor** : Dr. Mostafa Tavasoli
* **Term** : Fall 1403

---

## **Assignment Details**

This assignment consists of two major sections:

### **1. Flow-Based Generative Models**

* Understanding Normalizing Flows (e.g., RealNVP, Residual Flows)
* Implementing RealNVP on FashionMNIST
* Likelihood-based Out-of-Distribution Detection
* Jacobian determinant estimation in residual flows

### **2. Generative Adversarial Networks (GANs)**

* Mathematical derivations of GAN training
* Implementation of GAN and WGAN
* FID (Frechet Inception Distance) for quality evaluation
* Analyzing convergence and stability of GAN training

---

## **Sections Overview**

### **Flow-Based Generative Models**

Flow-based models provide a **reversible transformation** between a simple distribution (e.g., Gaussian) and complex data distributions.

#### **Tasks:**

1. **Understanding RealNVP (Normalizing Flow Model)** :

* Study the  **original RealNVP paper** .
* Understand how **log-likelihood estimation** works in normalizing flows.

1. **Implementing RealNVP on FashionMNIST** :

* Implement **coupling transformations** for RealNVP.
* Train the model on  **FashionMNIST dataset** .
* Evaluate the model by generating  **out-of-distribution samples** .

1. **Likelihood-Based OOD Detection** :

* Analyze how flow models fail in detecting  **out-of-distribution samples** .
* Experiment with  **FashionMNIST and another dataset** .

1. **Jacobian Determinant Estimation in Residual Flows** :

* Study how **Hutchinson trace estimator** is used for Jacobian computation.

---

### **Generative Adversarial Networks (GANs)**

GANs are **powerful generative models** trained using an adversarial framework.

#### **Tasks:**

1. **Mathematical Analysis of GANs** :

* Prove why **vanishing gradients** occur in standard GAN loss.
* Derive  **optimal discriminator function** .

1. **Implementing GAN from Scratch** :

* Define **generator** and **discriminator** models.
* Train a **FashionMNIST GAN** and visualize generated samples.

1. **Understanding Wasserstein GAN (WGAN)** :

* Explain why  **WGAN improves over standard GAN** .
* Study the **role of Lipschitz constraint** and  **critic loss** .

1. **Implementing WGAN** :

* Modify the standard GAN to  **WGAN framework** .
* Train the model and compare  **convergence properties** .

1. **Evaluating GANs using FID** :

* Compute  **Frechet Inception Distance (FID)** .
* Compare FID scores across  **different training epochs** .

---

## **Implementation Details**

### **Dataset**

* The dataset used is  **FashionMNIST** .
* 80/20 train-test split.
* **Preprocessing** :
* Normalize pixel values to `[0,1]`.

### **GAN Model Architecture**

| Layer Type    | Generator           | Discriminator |
| ------------- | ------------------- | ------------- |
| Input         | Noise vector (100D) | 28×28 Image  |
| Convolutional | 4×4 Transpose Conv | 4×4 Conv     |
| Activation    | ReLU                | LeakyReLU     |
| BatchNorm     | Yes                 | Yes           |
| Output        | 28×28 Image        | Sigmoid       |

### **Training Parameters**

| Parameter     | Value  |
| ------------- | ------ |
| Latent Dim    | 100    |
| Learning Rate | 0.0002 |
| Batch Size    | 64     |
| Epochs        | 100    |

### **Loss Functions**

* **GAN Loss** :
* $$
  L_G = -\mathbb{E}[\log D(G(z))]
  $$
* $$
  L_D = -\mathbb{E}[\log D(x)] - \mathbb{E}[\log(1 - D(G(z)))]
  $$

  **WGAN Loss** :
* $$
  L_D = -\mathbb{E}[D(x)] + \mathbb{E}[D(G(z))]
  $$
* $$
  * L_G = -\mathbb{E}[D(G(z))]
  $$

---



## **License**

This project is licensed under the MIT License.

For more details, see the [LICENSE](https://chatgpt.com/c/LICENSE) file.

---
