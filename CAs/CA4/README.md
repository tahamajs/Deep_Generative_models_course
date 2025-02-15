
# **Diffusion and Score-Based Generative Models - Deep Generative Models HW3**

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
  * [Diffusion Models](https://chatgpt.com/c/67b0eae8-c268-8007-9df3-33d28ab21913#diffusion-models)
  * [Score-Based Generative Models](https://chatgpt.com/c/67b0eae8-c268-8007-9df3-33d28ab21913#score-based-generative-models)
* [Implementation Details](https://chatgpt.com/c/67b0eae8-c268-8007-9df3-33d28ab21913#implementation-details)
* [Mathematical Derivations](https://chatgpt.com/c/67b0eae8-c268-8007-9df3-33d28ab21913#mathematical-derivations)
* [Training and Experimentation](https://chatgpt.com/c/67b0eae8-c268-8007-9df3-33d28ab21913#training-and-experimentation)
* [Results and Analysis](https://chatgpt.com/c/67b0eae8-c268-8007-9df3-33d28ab21913#results-and-analysis)
* [Submission Guidelines](https://chatgpt.com/c/67b0eae8-c268-8007-9df3-33d28ab21913#submission-guidelines)
* [License](https://chatgpt.com/c/67b0eae8-c268-8007-9df3-33d28ab21913#license)
* [Project Structure](https://chatgpt.com/c/67b0eae8-c268-8007-9df3-33d28ab21913#project-structure)

---

## **Introduction**

This repository contains **Homework 3** for the **Deep Generative Models** course at the  **University of Tehran** . The assignment covers  **two major generative modeling techniques** :

* **Diffusion-Based Models** :
* **DDPM (Denoising Diffusion Probabilistic Models)**
* **DDIM (Denoising Diffusion Implicit Models)**
* **Forward and Backward Diffusion Processes**
* **Score-Based Generative Models** :
* Learning probability densities using **score functions**
* **Sliced Score Matching (SSM)**
* **Langevin Dynamics for Sampling**

This assignment provides **both theoretical and practical** components to help students gain hands-on experience with  **modern generative techniques** .

---

## **Course Information**

* **University** : University of Tehran
* **Department** : Electrical and Computer Engineering
* **Course** : Deep Generative Models
* **Instructor** : Dr. Mostafa Tavasoli
* **Term** : Fall 1403

---

## **Assignment Details**

This homework consists of  **two main sections** :

### **1. Diffusion Models**

* Understanding **Forward and Backward Diffusion**
* Implementing **DDPM and DDIM**
* Training a **Diffusion Model on the Sprites Dataset**
* Comparing **DDPM and DDIM sampling efficiency**
* Implementing **FID (Frechet Inception Distance) for evaluation**

### **2. Score-Based Generative Models**

* Understanding **Score Matching**
* Implementing **Sliced Score Matching (SSM)**
* **Noise Perturbation Techniques** for learning the score function
* Implementing **Langevin Dynamics Sampling**

---

## **Sections Overview**

### **Diffusion Models**

Diffusion models progressively **add noise** to data in a **forward process** and learn to **reverse this process** to generate new samples.

#### **Tasks:**

1. **Understanding Diffusion Models**
   * Explain the **Markov process** in forward diffusion.
   * Prove that adding Gaussian noise results in a normal distribution.
2. **Implementing DDPM and DDIM**
   * Implement **DDPM training** on the  **Sprites dataset** .
   * Implement **DDIM sampling** and compare results with DDPM.
3. **Training a Noise Scheduler**
   * Implement a **linear noise scheduler** to control diffusion.
4. **Sampling from a Trained Model**
   * Compare **DDPM and DDIM** in terms of **sampling speed** and quality.
   * Use **FID score** for quantitative evaluation.

---

### **Score-Based Generative Models**

Score-based models learn to **model probability distributions** using **score functions** instead of explicit density functions.

#### **Tasks:**

1. **Understanding Score-Based Models**
   * Explain how **Score Matching** differs from standard generative models.
   * Compare  **Diffusion Models vs. Score-Based Models** .
2. **Implementing Score Matching**
   * Implement **Sliced Score Matching (SSM)** to learn score functions.
   * Train a model using  **noise perturbation techniques** .
3. **Sampling Using Langevin Dynamics**
   * Implement **Langevin Dynamics Sampling** to generate samples.
   * Compare **Langevin Sampling** with Diffusion Sampling.
4. **Evaluating Model Performance**
   * Visualize results using **scatter plots** and  **density estimates** .
   * Compare different  **perturbation noise levels** .

---

## **Implementation Details**

### **Dataset**

* **Diffusion Models** : **Sprites dataset** (16x16 pixel game character images)
* **Score-Based Models** : **Mixture of Gaussians**

### **DDPM/DDIM Training Parameters**

| Parameter      | Value  |
| -------------- | ------ |
| Learning Rate  | 0.0002 |
| Batch Size     | 64     |
| Training Steps | 50,000 |

### **Score Matching Model Parameters**

| Parameter     | Value     |
| ------------- | --------- |
| Noise Levels  | [1, 3, 7] |
| Learning Rate | 0.001     |
| Batch Size    | 128       |

---

## **Mathematical Derivations**

1. **Forward Diffusion Process**
   * Derive the  **Markov transition probabilities** .
   * Prove that **adding Gaussian noise** leads to a normal distribution.
2. **Score Matching Equations**
   * Explain how **Sliced Score Matching (SSM)** reduces complexity.
   * Compare  **SSM vs. Vanilla Score Matching** .
3. **Langevin Dynamics**
   * Show why **Langevin Sampling** converges to the true distribution.

---

## **Training and Experimentation**

1. **Train DDPM and DDIM** on the Sprites dataset.
2. **Compare DDPM vs. DDIM Sampling Time** .
3. **Train a Score-Based Model** on a  **Mixture of Gaussians** .
4. **Visualize Langevin Sampling Convergence** .

## **License**

This project is licensed under the  **MIT License** .

For more details, see the [LICENSE](https://chatgpt.com/c/LICENSE) file.

---
