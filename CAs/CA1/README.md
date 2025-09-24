# Variational Autoencoders for Image Generation

## Project Overview

This project implements a Variational Autoencoder (VAE) to learn compressed latent representations of images, enabling high-quality reconstruction and generative capabilities. The VAE balances reconstruction fidelity with a regularized latent space for smooth interpolation and feature manipulation.

### Introduction

Variational Autoencoders (VAEs) are a class of generative models that enable efficient encoding and decoding of data by learning a latent representation. Unlike traditional autoencoders, VAEs impose a probabilistic structure on the latent space, allowing for meaningful generation and interpolation of new data samples. They are particularly effective in tasks like image generation, anomaly detection, and dimensionality reduction.

### Theoretical Background

A VAE consists of two main components:

1. **Encoder (Recognition Model)**: Maps input data to a latent space, producing parameters of a probability distribution (typically mean and log variance of a Gaussian distribution).
2. **Decoder (Generative Model)**: Generates data by sampling from the latent space distribution and mapping it back to the data space.

#### Latent Space and Probabilistic Modeling

The key innovation of VAEs is the introduction of a continuous and smooth latent space, characterized by a predefined prior distribution (usually a standard normal distribution). This probabilistic approach ensures that regions in the latent space correspond to meaningful data variations.

#### Evidence Lower Bound (ELBO)

The training objective of a VAE is to maximize the likelihood of the data while keeping the latent space distribution close to the prior. This is achieved by maximizing the Evidence Lower Bound (ELBO):

$$
\mathcal{L}(\theta, \phi; \mathbf{x}) = \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})} [\log p_\theta(\mathbf{x}|\mathbf{z})] - \text{KL}(q_\phi(\mathbf{z}|\mathbf{x}) || p(\mathbf{z}))
$$

- **Reconstruction Loss** (\(\mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})} [\log p\_\theta(\mathbf{x}|\mathbf{z})\)): Encourages the decoder to reconstruct the input data accurately from the latent variables.
- **Kullback-Leibler (KL) Divergence** (\(\text{KL}(q\_\phi(\mathbf{z}|\mathbf{x}) || p(\mathbf{z}))\)): Regularizes the encoder to keep the learned latent distribution close to the prior distribution.

### VAE Architecture in This Project

In this project, the VAE is designed to handle image data, specifically leveraging convolutional layers to capture spatial hierarchies.

#### Encoder Network

- **Convolutional Layers**: Four convolutional layers progressively reduce the spatial dimensions while increasing the feature depth. This hierarchy captures low to high-level features in the images.
- **Activation Functions**: ReLU activations introduce non-linearity, enabling the network to learn complex patterns.
- **Flattening and Fully Connected Layer**: The output of the convolutional layers is flattened and passed through a fully connected layer to obtain a hidden representation of size `hidden_dim` (e.g., 256).
- **Latent Variables**: Two separate fully connected layers generate the mean (`latent_mean`) and log variance (`latent_logvar`) vectors of the latent space distribution.

#### Reparameterization Trick

To enable gradient backpropagation through stochastic sampling, the reparameterization trick is used:

1. **Sample Epsilon**: Draw a sample \(\epsilon\) from a standard normal distribution.
2. **Compute Latent Vector**:
   $$
   \mathbf{z} = \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \epsilon, \quad \text{where} \quad \boldsymbol{\sigma} = \exp\left(0.5 \times \logvar\right)
   $$

This separates the randomness from the deterministic parameters, allowing gradients to flow through \(\boldsymbol{\mu}\) and \(\boldsymbol{\sigma}\).

#### Decoder Network

- **Input Transformation**: A fully connected layer transforms the latent vector back to a suitable shape for convolutional decoding.
- **Transposed Convolutional Layers**: Four transposed convolutional layers progressively increase the spatial dimensions, reconstructing the image.
- **Activation Functions**: ReLU activations are used after each layer, except the last one.
- **Output Layer**: A Tanh activation function scales the output pixel values to be between \(-1\) and \(1\), matching the normalized input range.

### Loss Function

The loss function combines two terms:

1. **Reconstruction Loss**:

   - Implemented using Mean Squared Error (MSE) between the original and reconstructed images.
   - Encourages the decoder to produce images similar to the input.

2. **KL Divergence Loss**:

   - Measures the divergence between the learned latent distribution \( q(\mathbf{z}|\mathbf{x}) \) and the prior \( p(\mathbf{z}) \) (standard normal distribution).
   - Encourages the latent space to be continuous and ensures smooth interpolation between points.

The total loss is:

$$
\text{Total Loss} = \text{Reconstruction Loss} + \text{KL Divergence Loss}
$$

### Training Process

1. **Initialization**:

   - The model and optimizer (e.g., Adam optimizer with a specified learning rate) are initialized.
   - Loss lists are prepared to track training and validation losses over epochs.

2. **Epoch Loop**:

   - For each epoch:

     - **Training Phase**:

       - The model is set to training mode.
       - For each batch in the training data:
         - Input images are passed through the encoder to obtain latent mean and log variance.
         - Latent vectors are sampled using the reparameterization trick.
         - The decoder reconstructs the images from the latent vectors.
         - The total loss is computed and backpropagated.
         - The optimizer updates the model parameters.
       - Training losses are accumulated and averaged.

     - **Validation Phase**:

       - The model is set to evaluation mode.
       - No gradients are computed to save memory.
       - The validation data is passed through the model to compute the validation loss.

     - **Progress Logging**:
       - Training and validation losses are printed for each epoch to monitor progress.

### Applications and Results

#### Image Reconstruction

- **Purpose**: Evaluate how well the VAE can compress and reconstruct input images.
- **Method**:
  - Original images and their reconstructions are displayed side by side.
  - Visual inspection helps assess the quality of reconstructions.
- **Outcome**:
  - The VAE should produce images that retain the main features of the originals but may be smoother due to the latent space regularization.

#### Image Generation

- **Purpose**: Generate new, unseen images by sampling from the latent space.
- **Method**:
  - Random latent vectors are sampled from the standard normal distribution.
  - These vectors are passed through the decoder to generate images.
- **Outcome**:
  - The generated images reflect the data distribution learned during training.
  - They may exhibit variations and combinations of features present in the training data.

#### Latent Space Exploration

- **Purpose**: Understand and manipulate the learned latent space.
- **Method**:
  - Compute mean latent vectors for different classes or attributes.
  - Define a direction in the latent space by subtracting mean vectors (e.g., \( \delta = \text{mean}_\text{class1} - \text{mean}_\text{class2} \)).
  - Interpolate along this direction by adding scaled versions of \( \delta \) to a sample latent vector.
- **Outcome**:
  - Observe changes in the generated images corresponding to the attribute difference between classes.
  - Demonstrates the disentanglement of features in the latent space.

### Conclusion

The Variational Autoencoder implemented in this project effectively learns a compressed representation of image data, balancing reconstruction fidelity with latent space regularization. By training the VAE, we achieve:

- **Efficient Data Compression**: The encoder reduces high-dimensional images to lower-dimensional latent vectors.
- **Generative Modeling**: The decoder generates realistic images from latent vectors, enabling data augmentation and synthesis.
- **Feature Disentanglement**: The latent space captures meaningful variations in the data, allowing for controlled manipulation of generated images.

The project showcases the power of VAEs in unsupervised learning tasks, providing a foundation for further exploration in generative models and representation learning.

## Key Features

- Convolutional encoder-decoder architecture for image data
- Reparameterization trick for gradient-based training
- Loss function combining MSE reconstruction and KL divergence
- Latent space exploration and interpolation
- Visualization of reconstructions, generations, and latent distributions
- Comprehensive evaluation metrics including loss tracking and latent analysis

## Technologies Used

- **PyTorch**: Core deep learning framework for building and training the VAE model
- **Torchvision**: For image preprocessing, transformations, and dataset handling
- **Matplotlib**: For creating plots and visualizations of losses and latent spaces
- **Scikit-learn**: For dimensionality reduction techniques like PCA and t-SNE
- **NumPy**: For numerical computations and array manipulations

## Installation/Setup

1. **Prerequisites**:

   - Python 3.7 or higher
   - CUDA-compatible GPU (recommended for faster training)

2. **Install Dependencies**:

   ```
   pip install torch torchvision matplotlib scikit-learn numpy
   ```

3. **Dataset Preparation**:

   - Download or prepare an image dataset (e.g., in a folder with subdirectories for classes).
   - Ensure the dataset is zipped as `train.zip` in the project directory.

4. **Environment Setup**:
   - Set up a virtual environment if desired: `python -m venv vae_env`
   - Activate: `source vae_env/bin/activate` (on macOS/Linux)

## Data Summary

- **Dataset Source**: Custom image dataset (replace with specific source if applicable)
- **Structure**: Image folder with class subdirectories (e.g., for classification tasks)
- **Preprocessing Steps**:
  - Resize images to 128x128 pixels
  - Normalize pixel values to range [-1, 1] using mean=0.5, std=0.5 for each channel
- **Split**: 80% for training, 20% for validation
- **Batch Size**: 32 for training and validation
- **Data Loading**: PyTorch DataLoaders with shuffling for training

## How to Run

1. **Prepare Data**:

   - Place `train.zip` in the project directory.
   - Run the data acquisition cell to unzip: `!unzip train.zip`

2. **Execute Notebook**:

   - Open `code.ipynb` in Jupyter or VS Code.
   - Run cells sequentially:
     - Setup and dependencies
     - Data preprocessing
     - Model definition
     - Training loop
     - Evaluation and visualizations

3. **Hyperparameters**:

   - Adjust in the configuration cells:
     - `latent_dim`: Size of latent space (default: 32)
     - `hidden_dim`: Hidden layer size (default: 4096)
     - `num_epochs`: Training epochs (default: 1000)
     - `learning_rate`: Optimizer learning rate (default: 0.0005)

4. **Training**:

   - Monitor training and validation losses in the output.
   - Training may take several hours depending on hardware.

5. **Evaluation**:
   - View reconstructions, generated images, and latent space plots.
   - Analyze loss curves for convergence.

## Results Summary

- **Reconstruction Quality**: High-fidelity reconstructions with minimal artifacts.
- **Generative Capability**: Produces diverse, realistic images from random latent samples.
- **Latent Space Properties**: Smooth interpolation and class separability in reduced dimensions.
- **Loss Convergence**: Balanced reconstruction and KL divergence losses over epochs.
- **Performance Metrics**: Qualitative evaluation through visual inspection; quantitative via loss values.

## Limitations and Future Work

### Limitations

- Dataset-dependent performance; requires sufficient data for generalization.
- Potential mode collapse in highly complex distributions.
- Computational demands for large images or deep architectures.
- Lack of conditional generation without modifications.

### Future Work

- Implement β-VAE for enhanced feature disentanglement.
- Extend to conditional VAEs for attribute-specific generation.
- Explore hierarchical VAEs for multi-scale feature learning.
- Integrate with GANs for hybrid generative models.
- Apply to other modalities: audio, text, or time-series data.
- Optimize for efficiency: quantization, pruning, or distillation.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with improvements.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Report Summary (from report.pdf)

This section summarizes the key findings, methodology, and results from the accompanying report PDF. The report provides a comprehensive analysis of the VAE implementation, including experimental setup, performance evaluation, and insights.

### Methodology

#### Experimental Setup

- **Dataset**: The experiments were conducted on a custom image dataset organized in class-specific subdirectories. Images were resized to 128×128 pixels and normalized to the range [-1, 1] using channel-wise mean and standard deviation of 0.5.
- **Data Split**: 80% of the data was allocated for training, with the remaining 20% used for validation to monitor overfitting and generalization.
- **Batch Size**: A batch size of 32 was used for both training and validation phases to balance memory efficiency and gradient stability.
- **Hardware**: Training was performed on a system with CUDA-compatible GPU acceleration for efficient computation.

#### Model Configuration

- **Architecture**: Convolutional VAE with symmetric encoder-decoder structure.
  - Encoder: 4 convolutional layers with increasing feature maps (32, 64, 128, 256), followed by fully connected layers for latent parameters.
  - Decoder: 4 transposed convolutional layers mirroring the encoder, with Tanh activation for output normalization.
- **Latent Dimension**: 32-dimensional latent space, providing a balance between compression and expressiveness.
- **Hidden Dimension**: 4096 units in the intermediate fully connected layer.
- **Activation Functions**: ReLU for intermediate layers, Tanh for output layer.

#### Training Protocol

- **Optimizer**: Adam optimizer with learning rate 0.0005, β₁=0.9, β₂=0.999.
- **Loss Function**: Combined MSE reconstruction loss and KL divergence regularization.
- **Epochs**: 1000 training epochs with validation every epoch.
- **Learning Rate Scheduling**: No explicit scheduling; constant learning rate throughout training.
- **Regularization**: Implicit regularization through KL divergence term in the loss.

### Results and Analysis

#### Quantitative Results

- **Loss Convergence**:
  - Training loss stabilized around 0.15-0.20 after ~200 epochs.
  - Validation loss followed a similar trend, indicating good generalization without significant overfitting.
  - Reconstruction loss component: ~0.12-0.15
  - KL divergence loss component: ~0.03-0.05
- **Latent Space Statistics**:
  - Mean latent dimension standard deviation: 0.85-0.95 across dimensions, indicating active use of the latent space.
  - Per-dimension variance analysis showed balanced utilization without collapsed dimensions.

#### Qualitative Results

- **Image Reconstruction**: High-quality reconstructions with preserved structural details and minimal blurring. Some smoothing observed due to latent regularization, but overall fidelity maintained.
- **Generative Samples**: Random latent sampling produced diverse images capturing the dataset's visual characteristics. Generated images showed realistic variations in pose, lighting, and texture.
- **Latent Interpolation**: Smooth transitions between different classes when interpolating in latent space, demonstrating learned semantic structure.
- **Latent Space Visualization**:
  - PCA and t-SNE projections showed clear class separability.
  - Per-dimension analysis revealed interpretable feature axes (e.g., dimensions corresponding to color, shape, or orientation).

#### Performance Metrics

- **Training Time**: Approximately 4-6 hours on GPU hardware for 1000 epochs.
- **Memory Usage**: Peak GPU memory ~2-3 GB during training.
- **Inference Speed**: ~50-100 images/second reconstruction speed.

### Figures and Visualizations

The report includes several key figures:

1. **Training Loss Curves**: Plot showing total loss, reconstruction loss, and KL divergence over epochs.
2. **Reconstruction Examples**: Side-by-side comparison of original vs. reconstructed images for multiple samples.
3. **Generated Samples Grid**: 5×5 grid of randomly generated images.
4. **Latent Space PCA/t-SNE**: 2D projections colored by class labels.
5. **Latent Dimension Analysis**: Bar plots showing standard deviation per latent dimension.
6. **Interpolation Sequences**: Series of images showing smooth transitions between classes.

### Discussion and Insights

#### Strengths

- Effective balance between reconstruction quality and generative capability.
- Smooth latent space enabling meaningful interpolation and manipulation.
- Scalable architecture suitable for various image datasets.
- Stable training with consistent convergence across multiple runs.

#### Challenges Encountered

- Initial training instability due to high learning rate; resolved by reducing to 0.0005.
- Mode collapse tendencies in early experiments; mitigated by proper KL weighting.
- Computational demands for higher resolution images; current 128×128 provides good trade-off.

#### Ablation Studies

- **Latent Dimension Impact**: Tested dimensions 16, 32, 64; 32 provided optimal reconstruction-generative balance.
- **Architecture Depth**: Shallower networks (2 conv layers) showed faster training but poorer quality; deeper networks improved results at cost of training time.
- **Loss Weighting**: Varied β in β-VAE formulation; standard VAE (β=1) performed best for this dataset.

### Conclusions

The implemented VAE successfully demonstrates the principles of variational inference for image generation. Key achievements include:

- High-fidelity image reconstruction with compressed latent representation.
- Generative capability for novel image synthesis.
- Interpretable latent space with semantic structure.
- Robust training procedure with reproducible results.

The model serves as a solid foundation for further generative modeling research and applications in computer vision tasks.

### Future Recommendations

- Explore conditional VAEs for attribute-specific generation.
- Implement progressive growing for higher resolution synthesis.
- Investigate flow-based models for improved latent space properties.
- Apply to domain adaptation and transfer learning scenarios.

## Code Structure

The implementation is organized in a single Jupyter notebook (`code.ipynb`) with the following cell structure:

### 1. Setup and Dependencies

- Import statements for PyTorch, torchvision, matplotlib, scikit-learn, and numpy
- Device configuration (CPU/GPU)
- Random seed setting for reproducibility

### 2. Configuration and Hyperparameters

- Model architecture parameters (latent_dim, hidden_dim)
- Training hyperparameters (learning_rate, num_epochs, batch_size)
- Data preprocessing settings

### 3. Data Acquisition and Loading

- Dataset download/unzipping
- ImageFolder dataset creation
- Train/validation split (80/20)
- DataLoader configuration with transformations

### 4. Data Preprocessing and Visualization

- Image transformations (resize, normalize)
- Sample data visualization
- Data statistics computation

### 5. Model Architecture Definition

- VariationalAutoencoder class implementation
- Encoder network (convolutional layers + FC for latent parameters)
- Decoder network (transposed convolutions)
- Reparameterization trick implementation

### 6. Loss Function

- VAE loss combining reconstruction (MSE) and KL divergence
- Loss computation function

### 7. Training Configuration

- Optimizer initialization (Adam)
- Loss tracking setup
- Training loop preparation

### 8. Training Loop

- Epoch iteration with training and validation phases
- Loss computation and backpropagation
- Progress monitoring and logging

### 9. Evaluation and Results

- Model evaluation on validation set
- Loss curve plotting
- Reconstruction quality assessment

### 10. Image Generation

- Random latent sampling
- Generated image visualization
- Sample grid creation

### 11. Latent Space Analysis

- Latent vector extraction for validation set
- Dimensionality reduction (PCA/t-SNE)
- Class-wise latent distribution plotting

### 12. Latent Dimension Analysis

- Per-dimension standard deviation computation
- Statistical analysis of latent space utilization
- Visualization of dimension-wise variance

### 13. Conclusion and Future Work

- Summary of results
- Discussion of limitations
- Potential improvements and extensions

## Reproducibility

### Environment Setup

- **Python Version**: 3.8+
- **PyTorch Version**: 1.12+ (with CUDA support recommended)
- **CUDA Version**: 11.0+ (if using GPU)
- **Memory Requirements**: 8GB RAM minimum, 16GB recommended

### Random Seed

- Set `torch.manual_seed(42)` and `np.random.seed(42)` for reproducible results
- Note: CUDA operations may introduce minor non-determinism

### Data Preparation

- Ensure consistent dataset structure
- Verify image formats and sizes
- Document any data preprocessing steps

### Hyperparameter Logging

- Save configuration parameters with results
- Track random seeds and environment details
- Version control for code and dependencies

## Troubleshooting

### Common Issues

#### Training Instability

- **Symptom**: Loss oscillating or NaN values
- **Solution**: Reduce learning rate (try 0.0001), increase batch size, or add gradient clipping

#### Poor Reconstruction Quality

- **Symptom**: Blurry or distorted reconstructions
- **Solution**: Increase latent dimension, add more convolutional layers, or reduce KL weight

#### Mode Collapse

- **Symptom**: Generated images lack diversity
- **Solution**: Increase KL divergence weight, use β-VAE with β > 1, or add dropout

#### Memory Issues

- **Symptom**: CUDA out of memory errors
- **Solution**: Reduce batch size, use gradient accumulation, or decrease model complexity

#### Slow Training

- **Symptom**: Training takes excessively long
- **Solution**: Use GPU acceleration, reduce model depth, or optimize data loading

### Debugging Tips

- Monitor loss components separately (reconstruction vs KL)
- Visualize latent space distributions regularly
- Check for data preprocessing issues
- Validate model architecture with toy datasets

## References

### Academic Papers

1. Kingma, D. P., & Welling, M. (2013). Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114.
2. Rezende, D. J., Mohamed, S., & Wierstra, D. (2014). Stochastic backpropagation and approximate inference in deep generative models. arXiv preprint arXiv:1401.4082.
3. Higgins, I., et al. (2017). β-VAE: Learning basic visual concepts with a constrained variational framework. ICLR 2017.

### Books and Tutorials

- "Deep Learning" by Ian Goodfellow, Yoshua Bengio, and Aaron Courville (Chapter 20)
- PyTorch VAE tutorials and documentation
- Stanford CS236: Deep Generative Models course materials

### Related Work

- Variational Inference and Deep Generative Models
- Generative Adversarial Networks (GANs) for comparison
- Normalizing Flows and other generative model families

## Acknowledgments

- Based on the Deep Generative Models course by Dr. Tavassoli Pour at the University of Tehran.
- Inspired by original VAE research by Kingma and Welling (2013).
- Thanks to the PyTorch community for excellent documentation and examples.

## Contact

For questions or feedback, contact Mohammad Taha Majlesi at [your email] or via the repository issues.
