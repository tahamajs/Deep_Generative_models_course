# Project Rules and Conventions

This document contains all the rules, conventions, and best practices used in the Deep Generative Models (DGM) course project.

---

## 🎯 Core Development Rules

### Code Quality

1. **No Code Duplication**: Make the codes not duplicated and fully structured

   - Reuse functions and classes across notebooks when possible
   - Extract common utilities into shared modules
   - Maintain DRY (Don't Repeat Yourself) principle

2. **Full Structure**: Code must be fully structured with:
   - Clear organization and modularity
   - Proper separation of concerns
   - Well-defined interfaces between components

### Version Control

3. **Git Commit After Features**: After adding features to the code, you must commit that on git
   - Make atomic commits for each feature
   - Include meaningful commit messages
   - Do not commit work-in-progress features

---

## 📁 Repository Structure Conventions

### Standard Folder Structure

Each course assignment (CA1-CA4) follows this structure:

```
CA[#]_[Name]/
├── code/          # Jupyter notebooks and code files
├── description/   # Assignment description PDF
├── report/        # PDF reports and LaTeX sources
├── images/        # Generated images and visualizations
├── train/         # Training datasets (if applicable)
└── README.md      # Detailed documentation for the assignment
```

### Naming Conventions

- **Folders**: Use underscores and PascalCase (e.g., `CA1_Variational_Autoencoders/`)
- **Notebooks**: Use descriptive names (e.g., `code.ipynb`, `CA2_DGM.ipynb`)
- **Reports**: Use consistent naming (e.g., `DGM_CA1_final_EN.pdf`)
- **Images**: Follow pattern `output_cell_[#]_img_[#].png` or descriptive names

---

## 📓 Notebook Organization Rules

### Notebook Structure

All Jupyter notebooks must follow this structure:

1. **Overview/Introduction Cell** (Markdown)

   - Brief description of the notebook
   - Learning objectives
   - What will be covered

2. **Setup and Configuration Cell** (Code)

   - Consolidated imports at the top
   - Configuration parameters (hyperparameters, paths, etc.)
   - Random seed setup for reproducibility
   - Device configuration (CPU/GPU)

3. **Data Loading and Preprocessing** (Code + Markdown)

   - Dataset loading
   - Data preprocessing steps
   - Data visualization if applicable

4. **Model Definition** (Code + Markdown)

   - Architecture definitions
   - Explanatory markdown blocks
   - Clear separation of components

5. **Training** (Code + Markdown)

   - Training loops
   - Progress monitoring
   - Checkpoint saving

6. **Evaluation and Results** (Code + Markdown)
   - Metrics computation
   - Visualizations
   - Analysis and interpretation

### Code Organization in Notebooks

- **Consolidate Imports**: All imports should be in a single cell at the top
- **Configuration Cell**: Centralize all hyperparameters and settings
- **Explanatory Markdown**: Add markdown cells between major sections
- **Educational Clarity**: Include overview cells for educational purposes

### Example Configuration Cell Structure:

```python
# Setup and Configuration
import random
import numpy as np
import torch

# Set random seeds for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 64
learning_rate = 1e-3
epochs = 50
latent_dim = 128
```

---

## 🔬 Reproducibility Requirements

### Environment Management

1. **Always Use Virtual Environments**

   - Use `venv` or `conda` to isolate dependencies
   - Never install packages globally

2. **Package Version Pinning**

   - Create `requirements.txt` or `environment.yml`
   - Pin all package versions (e.g., `torch==2.0.1`)
   - Commit dependency files to repository

3. **Python Version Specification**
   - Specify consistent Python versions (e.g., Python 3.10)
   - Document in README

### Randomness Control

4. **Set Seeds for All Random Sources**

   - Python `random` module
   - NumPy random number generator
   - PyTorch random number generator
   - CUDA random number generator (if applicable)

5. **Deterministic Operations**
   - Set `torch.backends.cudnn.deterministic = True`
   - Set `torch.backends.cudnn.benchmark = False`
   - Use fixed seeds in data splitting

### Experiment Tracking

6. **Record Git Commit Hash**

   - Use `git rev-parse --short HEAD` for each experiment
   - Include in run metadata

7. **Save Comprehensive Metadata**

   - Create `run_info.json` for each experiment
   - Include: timestamp, commit hash, hyperparameters, model config, dataset info
   - Save checkpoints with descriptive names

8. **Document Hardware**
   - Note GPU/CPU used
   - Document batch sizes that fit hardware
   - Record memory requirements

### Data Consistency

9. **Fixed Data Splits**

   - Use fixed train/val/test splits
   - Use seeded random splits
   - Document split methodology

10. **Consistent Preprocessing**
    - Apply identical preprocessing across experiments
    - Document normalization statistics
    - Save preprocessing parameters

---

## 🧪 Testing and Validation Rules

### Before Long Training Runs

1. **Smoke Tests**

   - Test with small batch sizes (e.g., `batch_size=16`)
   - Run for 1 epoch only
   - Verify forward/backward passes work
   - Check loss values are finite (not NaN/inf)

2. **Shape Verification**

   - Assert correct tensor shapes throughout
   - Test encoder/decoder output shapes
   - Verify inverse transformations (for flows)

3. **Checkpoint Testing**
   - Verify models save correctly
   - Test model loading
   - Ensure checkpoint compatibility

### Evaluation Checks

4. **Sanity Checks**
   - Visual inspection of generated samples
   - Verify metrics are reasonable
   - Check for mode collapse (in GANs)
   - Monitor training curves

---

## 💾 Data and Artifact Management

### Dataset Handling

1. **Download and Caching**

   - Use `torchvision` default cache or set `TORCH_HOME` environment variable
   - Document dataset locations
   - Verify dataset integrity

2. **Storage Organization**
   - Keep large datasets out of repository (use `.gitignore`)
   - Consider Git LFS for large checkpoints
   - Use shared cache directories when appropriate

### Model Checkpoints

3. **Saving Conventions**

   - Save PyTorch `state_dict()` (`.pth` files)
   - Use descriptive filenames: `model_name_epoch_[#].pth`
   - Save optimizer states if resuming training

4. **Checkpoint Organization**
   ```
   experiments/
   ├── run_[date]_[name]/
   │   ├── checkpoints/
   │   ├── samples/
   │   ├── logs/
   │   └── run_info.json
   ```

### Generated Artifacts

5. **Image Management**
   - Save image grids as PNG/JPG
   - Use consistent naming conventions
   - Clean up intermediate results to save space
   - For FID evaluation: keep fixed reference set of real images

---

## 📝 Documentation Standards

### README Files

1. **Each CA folder must have a README.md** containing:
   - Assignment overview
   - Quick start instructions
   - Key concepts explained
   - File structure
   - Usage instructions
   - Reproducibility notes
   - References

### Code Documentation

2. **Code Comments**

   - Explain complex logic
   - Document function parameters and returns
   - Add inline comments for non-obvious operations

3. **Markdown Cells**
   - Explain theoretical concepts
   - Provide context for experiments
   - Document observations and findings

### Report Standards

4. **Reports Must Include**
   - Abstract and introduction
   - Methodology
   - Experimental setup
   - Results and analysis
   - Discussion and conclusions
   - References

---

## 🔧 Technical Best Practices

### Performance

1. **GPU Usage**

   - Always use GPU when available (5-10× faster)
   - Check CUDA availability: `torch.cuda.is_available()`
   - Monitor GPU memory usage

2. **Memory Management**

   - Use appropriate batch sizes
   - Enable gradient accumulation for large models
   - Use mixed precision training when applicable
   - Clear unnecessary variables: `del variable`

3. **Optimization**
   - Use `torch.compile()` for speedups (PyTorch 2.0+)
   - Profile code to identify bottlenecks
   - Optimize data loading (use multiple workers)

### Error Handling

4. **Common Issues Prevention**
   - Check for CUDA out of memory errors
   - Add gradient clipping to prevent exploding gradients
   - Monitor for NaN losses
   - Verify input normalization

### Model-Specific Guidelines

#### VAEs (CA1)

- Monitor KL divergence to prevent collapse
- Use KL annealing if needed
- Test reparameterization trick gradient flow

#### Normalizing Flows (CA2)

- Verify invertibility: test inverse mapping
- Check log-determinant of Jacobian is finite
- Add small epsilon to prevent division by zero

#### GANs (CA2)

- Monitor for mode collapse
- Balance generator/discriminator learning rates
- Use fixed noise vectors for consistent evaluation

#### Diffusion Models (CA3)

- Use appropriate noise schedules
- Monitor sampling quality
- Consider DDIM for faster sampling

---

## 🚫 Things to Avoid

1. **Don't Commit Large Files**

   - Use `.gitignore` for datasets, checkpoints, large images
   - Consider Git LFS if necessary

2. **Don't Run Long Training Without Testing**

   - Always run smoke tests first
   - Verify code works on small batches

3. **Don't Skip Documentation**

   - Document all experiments
   - Explain non-obvious choices
   - Maintain README files

4. **Don't Ignore Reproducibility**

   - Always set random seeds
   - Document all hyperparameters
   - Save experiment metadata

5. **Don't Duplicate Code**
   - Extract common functionality
   - Reuse utilities across notebooks

---

## 📚 Additional Guidelines

### Academic Integrity

- Individual work required for assignments
- Cite all external sources and code
- No plagiarism or code sharing
- AI tools allowed for understanding, not direct answers

### Workflow

1. **Start Early**: Training takes several hours
2. **Monitor Progress**: Check reconstructions/generations during training
3. **Save Often**: Enable auto-save, but save notebooks frequently
4. **Understand Theory**: Don't just run code, understand concepts
5. **Quality over Speed**: Good analysis > quick completion
6. **Document Everything**: Take notes on observations

### Environment Setup

- Follow the Quick Start guide in main README
- Use virtual environments (venv or conda)
- Install dependencies from `requirements.txt` or `environment.yml`
- Verify CUDA version matches PyTorch installation

---

## 🎓 Learning Outcomes Focus

When working on assignments, focus on:

1. Understanding the theoretical foundations
2. Implementing models correctly
3. Analyzing results critically
4. Drawing meaningful conclusions
5. Comparing different approaches
6. Documenting findings clearly

---

## 📋 Checklist for New Features/Experiments

When adding new features or experiments:

- [ ] Code is structured and not duplicated
- [ ] Configuration is centralized
- [ ] Random seeds are set
- [ ] Smoke tests pass
- [ ] Documentation is updated
- [ ] README is updated if needed
- [ ] Changes are committed to git
- [ ] Commit message is descriptive
- [ ] Metadata is saved for experiments
- [ ] Checkpoints are saved appropriately

---

## 🔗 References

For detailed information, refer to:

- Main `README.md` for project overview
- Individual CA README files for assignment-specific details
- Course slides in `Slides/` directory
- Assignment PDFs in `description/` folders

---

_Last updated: Based on project structure and conventions as of project creation_
_This document should be updated as new conventions are established_
