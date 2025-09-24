# CA4: Fine-Tuning Paligemma Vision-Language Model on CLEVR Dataset

## Overview

This project demonstrates the fine-tuning of Google's Paligemma Vision-Language Model (VLM) on the CLEVR dataset using Low-Rank Adaptation (LoRA) for parameter-efficient fine-tuning. The goal is to enhance the model's visual reasoning capabilities by training it to answer complex questions about synthetic scenes containing multiple objects with varying attributes.

### Key Components

- **Model**: Paligemma-3B (pre-trained vision-language model)
- **Dataset**: CLEVR (Compositional Language and Elementary Visual Reasoning)
- **Technique**: LoRA (Low-Rank Adaptation) for efficient fine-tuning
- **Quantization**: 8-bit quantization using BitsAndBytes
- **Evaluation**: ROUGE metrics for answer quality assessment

## Prerequisites

- Python 3.8+
- Google Colab with GPU (recommended) or local GPU
- Hugging Face account (for dataset access)
- Google Drive (for data persistence)

## Installation

```bash
pip install --upgrade pip
pip install transformers datasets peft evaluate bitsandbytes rouge_score
```

## Data Preparation

The notebook uses a 1% subset of the CLEVR dataset for demonstration purposes. The full dataset contains:

- Training: ~70,000 images with questions
- Validation: ~15,000 images
- Test: ~15,000 images

Data is automatically downloaded and cached to Google Drive for persistence.

## Model Configuration

- **Base Model**: google/paligemma-3b-pt-224
- **LoRA Config**: r=8, alpha=32, target modules: q_proj, o_proj
- **Quantization**: 8-bit loading for memory efficiency
- **Input Processing**: Images resized to 224x224, text tokenized with max length 128
- **Output**: Generated answers with max length 384

## Training Setup

- **Batch Size**: 2 (train), 1 (eval)
- **Learning Rate**: 2e-5
- **Epochs**: 2
- **Gradient Accumulation**: 2 steps
- **Optimizer**: AdamW with 8-bit precision
- **Evaluation**: ROUGE-1, ROUGE-2, ROUGE-L scores

## How to Run

1. **Environment Setup**: Run the first cell to install dependencies
2. **Configuration**: Set random seeds and hyperparameters
3. **Data Loading**: Load and preprocess CLEVR dataset subset
4. **Model Preparation**: Load Paligemma with LoRA and quantization
5. **Training**: Execute `trainer.train()` (may take 2-3 hours)
6. **Evaluation**: Assess model performance with ROUGE metrics
7. **Inference**: Test the model on new images

### Expected Training Time

- With Google Colab GPU: ~2-3 hours for 2 epochs
- Memory Requirements: ~8GB GPU RAM with 8-bit quantization

## Results

After fine-tuning, the model should show improved performance on CLEVR visual reasoning tasks. Evaluate using:

- **Quantitative**: ROUGE scores on validation set
- **Qualitative**: Sample question-answer pairs
- **Visualization**: Image-question analysis

## Reproducibility

- Set random seeds: `torch.manual_seed(42)`, `np.random.seed(42)`, `random.seed(42)`
- Use fixed dataset subset ratio: 1%
- Save model checkpoints to `./results`
- Log training progress with TensorBoard

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch size or use smaller subset
2. **Dataset Loading**: Ensure Hugging Face authentication for CLEVR dataset
3. **GPU Availability**: Verify CUDA installation and GPU memory
4. **Package Conflicts**: Use the specified pip installs in order

### Colab-Specific

- Mount Google Drive for data persistence
- Enable GPU runtime in Colab settings
- Monitor GPU memory usage during training

## File Structure

```
CA4/
├── code/
│   └── CA4.ipynb          # Main notebook
├── description/
│   └── DGM_HW4.pdf        # Assignment description
└── report/
    └── [report files]     # Student report
```

## References

- [Paligemma Paper](https://arxiv.org/abs/2407.07726)
- [CLEVR Dataset](https://cs.stanford.edu/people/jcjohns/clevr/)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
- [PEFT Library](https://huggingface.co/docs/peft/index)

## Acknowledgments

This implementation is based on the Deep Generative Models course assignment, utilizing state-of-the-art vision-language models and efficient fine-tuning techniques.
