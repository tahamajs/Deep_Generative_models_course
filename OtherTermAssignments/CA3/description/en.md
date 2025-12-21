# In the Name of God

## University of Tehran

**College of Engineering**
**Faculty of Electrical and Computer Engineering**

**Course:** Trustworthy AI
**Instructor:** Dr. Mostafa Tavassoli
**Assignment:** Homework 3
**Date:** Azar 1404 (December 2025)

---

## Question 1: Energy-Based Models (EBM)

**Context:**
Energy-Based Models are a flexible approach in machine learning. Instead of explicitly defining a normalized probability distribution, they define an energy function over the input space that acts as a measure of "compatibility" of the input with the learned structure. Such a model only needs to assign low energy to observed data and high energy to unobserved data, without the need for the integral of the probability to sum to one (normalization). This flexibility converts them into a powerful conceptual tool for modeling in complex spaces. From denoising and reconstruction to generating samples, EBMs allow for analyzing data behavior without being constrained by the strict form of probability distributions.

### Part 1: Theory Questions

**Subsection 1 (4 Points):**
If we want to generate a face of a young man using an Energy-Based Model trained on faces, what method can we use? (Easier model).

**Subsection 2 (5 Points):**
First, explain the **Rejection Sampling** method.
If we want to sample from a distribution  corresponding to an Energy-Based Model trained on the MNIST dataset using Rejection Sampling, let the optimal proposal distribution be  and the target unnormalized distribution be .
If , what will be the **acceptance rate**?

**Subsection 3 (6 Points):**
Generative models from the EBM family define a flexible function  as the energy function:



Where  is the partition function. To train these models using the **Maximum Likelihood** method, we calculate the gradient of the log-likelihood.
It can be proven that this gradient equals:


**a)** Based on the relation above, what change does the training process try to make to the value of  for the training data () and the generated samples ()?
**b)** What is the computational challenge of the second term (Expectation over the model) in practice? How does the **Contrastive Divergence (CD)** method attempt to resolve this challenge?

---

### Part 2: Implementation Questions

**Goal:** In this section, we intend to train an Energy-Based Model on the **MNIST** dataset. After training, we will use the **Langevin Dynamics** algorithm to perform inference and generate samples to better understand the concept.

**Subsection 1: Data Preparation (5 Points)**

* Load the **MNIST** dataset using `torchvision.datasets.MNIST` (Train and Test separately).
* Apply necessary transforms so that each image is a tensor of shape `1 x 28 x 28` in the range `[0, 1]`.
* Define DataLoaders (e.g., Batch size 64).
* Write a simple function to display a batch of images (e.g., an 8x2 grid).
* *Expected Output:* One cell displaying random training images, one cell displaying random test images.



**Subsection 2: Architecture and Training (15 Points)**
We want to define a convolutional model that takes an image as input and outputs a single number (Energy).

* Create the model based on **Table 1**.
* Write the **Langevin Sampling** function. (Suggestion: Write it so you can provide a starting `x`).
* Train the model based on **Algorithm 1** (Training Pseudocode).
* *Suggested Hyperparameters:* Epochs = 10, Lambda = 1e-3.
* During Langevin sampling in training, start from noise (Uniform [0,1]).
* **Logging:** In every epoch (or every few hundred steps), print the Loss, Mean Energy of Real Data (), and Mean Energy of Fake Data ().
* **Visualization:** At the start, middle, and end of training, display images generated via Langevin sampling.


* **Post-Training Sampling:** After the model is trained, take a few images from the `train` set. Use these images as the *starting point* for Langevin sampling and display the resulting "generated" images.

**Table 1: Model Specifications (EBM)**
| Row | Configuration | Input Shape | Output Shape |
| :--- | :--- | :--- | :--- |
| 0 | — | (B, 1, 28, 28) | (B, 1, 28, 28) |
| 1 | Conv2d(1→32, k=3, s=1, p=1) + LeakyReLU(0.2) | (B, 1, 28, 28) | (B, 32, 28, 28) |
| 2 | Conv2d(32→32, k=4, s=2, p=1) + LeakyReLU(0.2) | (B, 32, 28, 28) | (B, 32, 14, 14) |
| 3 | Conv2d(32→64, k=3, s=1, p=1) + LeakyReLU(0.2) | (B, 32, 14, 14) | (B, 64, 14, 14) |
| 4 | Conv2d(64→64, k=4, s=2, p=1) + LeakyReLU(0.2) | (B, 64, 14, 14) | (B, 64, 7, 7) |
| 5 | Conv2d(64→128, k=3, s=1, p=1) + LeakyReLU(0.2) | (B, 64, 7, 7) | (B, 128, 7, 7) |
| 6 | AdaptiveAvgPool2d(output_size=1) | (B, 128, 7, 7) | (B, 128, 1, 1) |
| 7 | Reshape to vector (Flatten) | (B, 128, 1, 1) | (B, 128) |
| 8 | Linear(128→1) (FullyConnected) | (B, 128) | (B, 1) |

**Algorithm 1: One-step Training Pseudocode**

```python
B := number of images in x_real (Batch size)
# 1. Compute energy of real data
E_real := E_theta(x_real) # vector of size B

# 2. Generate fake samples using Langevin dynamics
x_fake := LANGEVIN_SAMPLING(...) 

# 3. Compute energy of fake samples
E_fake := E_theta(x_fake) # vector of size B

# 4. Data term of the loss:
# - push energies of real data down
# - push energies of fake data up
data_term := mean(E_real) - mean(E_fake)

# 5. Regularization on the magnitude of energies
# (to avoid energies growing to very large values)
reg_term := lambda * (mean(E_real^2) + mean(E_fake^2))

# 6. Total loss
loss := data_term + reg_term

```

**Subsection 3: Image Generation (10 Points)**

* Use Langevin sampling starting from **Uniform Noise [0,1]**.
* Run for a sufficient number of steps to generate 16 images.
* Display the 16 generated images.
* **Report:** Analyze the quality of the generated images. If the results are not good, explain the main reason for the failure. (The next section on denoising might help improve sampling).

**Subsection 4: Denoising Noisy Images (15 Points)**

* Take a small batch of 16 images from the `test` set.
* Add **Gaussian Noise** to these images (using noise levels mentioned below) and `clamp` them to `[0,1]`. Show these noisy images next to the original ones.
* Use these **noisy images** as the starting point for Langevin Sampling.
* Run sampling for a sufficient number of steps.
* Display the output images next to the real and noisy images.
* Perform this for three noise levels: **0.2, 0.4, and 0.6**.
* **Report:**
* Explain how close the noisy image became to the original image.
* Identify where the model succeeded and where it failed among the specified noise levels.



---

## Question 2: Score-Based Models

**Context:**
Score-Based Models have emerged in recent years as a powerful and distinct approach in Deep Generative Learning. Unlike traditional models that try to directly estimate the probability density function (PDF), these models focus on the **vector field of the gradient of the log-probability density** (Score Function). This framework provides flexibility for learning and removes computational limitations related to the normalization constant ().

### Part 1: Theory Questions

**Subsection 1 (5 Points):**
Assume an Energy-Based Model is defined as  for a probability distribution.
Define the **Score Function** for this model. Show that the Score Function is independent of the Partition Function . What advantage does this feature create during model training?

**Subsection 2 (6 Points):**
Explain why calculating the term  (divergence of the score) in the original **Score Matching** loss function is difficult and costly for high-dimensional data (like images).
Prove that minimizing the **Denoising Score Matching (DSM)** objective function is equivalent to minimizing the original Score Matching objective on the noisy distribution (precise mathematical proof is not required; explaining the logic and relationship between the gradient of the optimal score matching and the score of the noisy distribution is sufficient).

**Subsection 3 (4 Points):**
Consider a mixture of two Gaussian distributions with **Disjoint Supports** (Region A and Region B) with weights  and :
.
Show that the true score function  in any region depends only on the density of that region ( or ) and effectively ignores the mixing weights ().
**b)** Explain why this property causes the **Langevin Dynamics** algorithm to fail to sample correctly from the mixture (unable to respect the weights  and ) and why the samples cannot cross between the disjoint modes (mixing problem).

**Subsection 4 (5 Points):**
**a)** Name the problems that arise when applying Score Matching directly to real data (without noise). Explain why these problems prevent correct learning.
**b)** Referring to the **NCSN** (Noise Conditional Score Networks) paper, explain how **Multi-scale Noise Perturbation** solves these problems. Explain the role of  (largest noise) and  (smallest noise) in **Annealed Langevin Dynamics** sampling.

---

### Part 2: Implementation Questions

**Goal:** Implement an **NCSN** model. You will generate images of handwritten digits using **Annealed Langevin Dynamics**. Then, you will upgrade this to a **Conditional Model**.

**Subsection 1: Base Model (20 Points)**

1. **Data Preparation:** Download MNIST. Use a U-Net structure. Normalize images to `[-1, 1]`. Create geometric sequence of noise levels .
2. **Network Architecture:**
* Use **Table 2**.
* To achieve desired quality, use **AdaptiveResBlock**.
* Instead of simple addition, inject noise information into the network using **FiLM** (Feature-wise Linear Modulation). The noise embedding is converted into `scale` and `shift` parameters via a Dense layer and applied to the normalized output:



* **Input:** Divide the input image by .


3. **Training:**
* Train using the **Weighted Denoising Score Matching** loss function.
* Plot training loss vs. epoch.


4. **Sampling:**
* Implement **Annealed Langevin Dynamics**.
* Display a final grid of 16 generated digits.
* Visualize the process of transforming pure noise into the final number for 3 different samples (plot the evolution from ). Analyze the results.



**Table 2: Proposed ScoreNet Architecture**
| Stage | Layer / Block Type | Input Channels | Output Channels |
| :--- | :--- | :--- | :--- |
| **Embedding** | GaussianFourierProjection + Dense | Scalar () | 256 |
| **Input** | Conv2d (3x3) | 1 | 64 |
| **Encoder 1** | AdaptiveResBlock | 64 | 64 |
| **Down 1** | AvgPool2d (2x2) | 64 | 64 |
| **Encoder 2** | AdaptiveResBlock | 64 | 128 |
| **Down 2** | AvgPool2d (2x2) | 128 | 128 |
| **Encoder 3** | AdaptiveResBlock | 128 | 256 |
| **Up 1** | Interpolate + Concat | 256 | 384 |
| **Decoder 1** | AdaptiveResBlock | 384 | 128 |
| **Up 2** | Interpolate + Concat | 128 | 192 |
| **Decoder 2** | AdaptiveResBlock | 192 | 64 |
| **Output** | Conv2d (3x3) | 64 | 1 |

**Subsection 2: Conditional Model (15 Points)**
We want to control the generation process (e.g., generate only the number '5').

* Learn the conditional distribution .
* Refer to the NCSN paper for training techniques for conditional models.
* **Implementation:** Add an `nn.Embedding` layer for the 10 classes (digits 0-9). Combine the Class Embedding vector with the Noise Embedding vector at the input of the **AdaptiveResBlocks**.
* **Train:** Retrain the model using dataset labels. Plot loss vs epoch.
* **Generate:** Create a Grid where each **row** corresponds to a specific number (0 to 9).
* **Analysis:** Compare results with the unconditional section.

**Table 3: Suggested Hyperparameters**
| Parameter | Value | Description |
| :--- | :--- | :--- |
|  | 30 | Initial noise level |
|  | 0.01 | Final noise level |
| Num Levels (L) | 10 | Number of noise scales |
| Batch Size | 64 | - |
| Optimizer | Adam | - |
| Learning Rate | 2e-4 | - |
| Epochs | 30 | Minimum recommended |
| Langevin Steps (T)| 150 | Steps per noise level |
| Step LR () | 2e-5 | Langevin step size |

---

### Submission Guidelines

* **Deadline:** Thursday, 4th of Dey (December 25th).
* **Grace Time:** This deadline is strict. Use grace time if enabled on the portal.
* **Platform:** Upload to the course portal. Portal closes 7 days after the deadline.
* **Format:** Python code (executable) + Report (PDF).
* **Individual Work:** This is an individual assignment.
* **Plagiarism:** Similarity in code or reports will be considered cheating for both parties. Using ready-made code without citation/modification is cheating (Grade 0).
* **Report:** Must be typed (Handwritten not accepted). Images/Tables must have captions. A large portion of the grade depends on the report and problem-solving process explanation.
* **Naming Convention:** `HW3_[Lastname]_[StudentNumber].zip`
* **TA Contact:**
* Q1: `sinaprocomp@gmail.com`
* Q2: `farhadnasri999@gmail.com`



