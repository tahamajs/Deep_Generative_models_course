
# University of Tehran

**College of Engineering**
**Faculty of Electrical and Computer Engineering**

**Course:** Deep Generative Models
**Instructor:** Dr. Mostafa Tavassoli
**Assignment:** Homework 2
**Date:** Aban 1404 (November 2025)

---

## Question 1: Normalizing Flows

### Part 1: Theory

**Subsection 1 (5 Points):**
Using the **change of variables** formula, find the closed-form expression for  in each of the following cases:

1. 
2. 

**Subsection 2 (5 Points):**
In each of the following transformations, find the closed-form expression for the **determinant of the Jacobian matrix**:

1.  such that  and , where  is a neural network.
2.  such that  and , where  and  are neural networks.

---

### Part 2: Implementation (10 Points)

**Subsection 1: Implementation**
In this section, you are expected to implement a **Masked Autoregressive Flow (MAF)**. As you know, this is a type of Normalizing Flow where each transformation block is an **autoregressive** transformation; that is, each dimension of the output depends only on previous dimensions. To achieve this, the MAF paper utilizes **MADE** (Masked Autoencoder for Distribution Estimation) layers.

A suggested architecture is provided below; however, you are free to use more complex architectures if you have access to a GPU.

* **Image dimensions:** 
* **MADE Architecture:**
* `Linear(input: 128*128*3, output: 512)`
* `Linear(input: 512, output: 512)`
* `Linear(input: 512, output: 2*128*128*3)`
* *Note:* These are not standard linear layers; they must maintain the **autoregressive property** via masking.


* **Number of Transformation Blocks:** 7-8
* **Batch Size:** 3
* **Number of Epochs:** 100
* **Learning Rate:** 0.0001
* **Deliverable:** Explain the details of your implementation for MADE and a single MAF block in your report.

**Subsection 2: Model Training and Image Generation (15 Points)**
After implementation, train your model on the **MVTec AD** dataset, specifically the **capsule** class. Download the dataset using the following commands:

```bash
wget https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937454-1629951595/capsule.tar.xz
tar -xf capsule.tar.xz

```

1. Compare the training time vs. the image generation time.
2. Explain why image generation is time-consuming in **autoregressive** models.
3. Include generated images in your report.
4. As you observed, MAF is slow at generating images. Explain how the **Inverse Autoregressive Flow (IAF)** model improves the generation speed and compare the two models in terms of training vs. generation time.

---

### Part 3: Anomaly Detection (20 Points)

In this section, we explore a key application of Normalizing Flows: **Anomaly Detection**. In general, models must learn to identify "anomalies" after seeing only "normal" images during training.

1. Why is the "Accuracy" metric not used for evaluating anomaly detection models? What metrics are used instead?
2. Explain the concept of an **Anomaly Score**.
3. Explain how a Normalizing Flow trained on a normal dataset can be used for the task of anomaly detection.
4. Now, evaluate your trained model from the previous section for Anomaly Detection. Use the data in the `test` folder of the capsule dataset.
* Use **Negative Log Likelihood (NLL)** as the anomaly score for each image.
* Report the metrics you identified in step 1. You are expected to reach an **AUROC** of at least **0.6**.


5. As you know, exact likelihood calculation is not possible in **VAEs**, but they are still used for anomaly detection. In the test phase, the encoder and decoder are used to reconstruct the image. It is expected that the model reconstructs anomalies as "normal" images. By calculating the difference between the original and reconstructed image, we reach an anomaly score. Explain, with reasons, whether this method can also be used with Normalizing Flows.

---

## Question 2: CycleGAN

### Part 1: Theory (10 Points)

**Section 1: Loss Function**

1. Supervised image-to-image translation methods (like **pix2pix**) face a fundamental challenge in real-world data collection. What is this challenge, and why is it not easily resolved?
2. What is the core idea of **CycleGAN** to bypass this data limitation? Describe it along with its mathematical formulation.
3. Why is enforcing the **Cycle Consistency Constraint** vital? What undesirable behavior is observed if we rely solely on **Adversarial Training**?
4. Write the full objective function (Loss Function) of CycleGAN. What role does the  coefficient play? What happens to the output if  is chosen to be excessively large?
5. The paper mentions **Identity Loss**. What is the concept behind this loss? Provide an example of what kind of unwanted changes it prevents.

**Section 2: Architecture and Training (10 Points)**

1. Why does CycleGAN use two separate generators instead of a single unit for bidirectional translation? Explain from both theoretical and practical perspectives.
2. Detail the generator architecture chosen for this model and compare it with the pix2pix generator.
3. The **PatchGAN** discriminator is used here. What is the objective of this choice and what is its nature?
4. Why is an **Image History Buffer** (storing the last 50 generated images) used during discriminator training instead of using only the latest generated image? What problem in the training process does this mechanism solve?

**Section 3: Model Limitations (2 Points)**

1. CycleGAN performs poorly on tasks requiring large **geometric changes**. What does this limitation mean? Why does the cycle consistency constraint hinder learning these types of changes?

---

### Part 2: Implementation (30 Points Total)

**Subsection 1: Data Preparation (2 Points)**
Prepare one of the following datasets: `horse2zebra`, `apple2orange`, `summer2winter_yosemite`, or `monet2photo`. Display a few random images from both domains (classes).

**Subsection 2: Architecture Implementation (11 Points)**
Implement the **Generator** and **Discriminator** classes as described in the paper. Assume an image dimension of .

**Subsection 3: Loss Function and Training (15 Points)**
Implement the full loss function and train the model for **20 epochs**.

* Use the hyperparameter values mentioned in the paper (explain your reasoning if you choose different values).
* Plot the loss curves (Generators, Discriminators, and Cycle Consistency).
* Evaluate model output: Select random images from both domains at the beginning, middle, and end of training. Display input images alongside their translations and analyze the learning quality.

**Subsection 4: Bonus (5 Points)**
Implement the **Image History Buffer** and train the model on a different dataset. Analyze the training process and output quality as in the previous section.

---

## Submission Guidelines

* **Deadline:** End of Monday, **3rd of Azar** (November 24, 2025).
* **Grace Time:** Up to 7 days after the deadline with the specified penalty; the portal will close after that.
* **Format:** Python implementation (executable) + typed PDF Report.
* **Language:** Python.
* **Collaboration:** Assignments must be done **individually**. Plagiarism results in a grade of **zero** for all parties involved.
* **Naming:** `HW2_[Lastname]_[StudentNumber].zip`
* **Contact:**
* Q1: `javadkavian8@gmail.com`
* Q2: `alireza.zamani@outlook.com`



