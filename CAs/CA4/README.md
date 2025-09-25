# CA4: Fine-Tuning Paligemma Vision-Language Model on CLEVR Dataset

## Overview

This project demonstrates the fine-tuning of Google's **Paligemma Vision-Language Model (VLM)** on the **CLEVR dataset** using **Low-Rank Adaptation (LoRA)** for parameter-efficient fine-tuning. The goal is to enhance the model's visual reasoning capabilities by training it to answer complex questions about synthetic scenes containing multiple objects with varying attributes.

### Key Components

- **Model**: Paligemma-3B (pre-trained vision-language model)
- **Dataset**: CLEVR (Compositional Language and Elementary Visual Reasoning)
- **Technique**: LoRA (Low-Rank Adaptation) for efficient fine-tuning
- **Quantization**: 8-bit quantization using BitsAndBytes
- **Evaluation**: ROUGE metrics and exact-match accuracy for answer quality assessment

## Prerequisites

- Python 3.8+
- Google Colab with GPU (recommended) or local GPU
- Hugging Face account (for dataset access)
- Local storage for data caching (notebook uses `./data/` directory)

## Installation

```bash
pip install --upgrade pip
pip install transformers datasets peft evaluate bitsandbytes rouge_score huggingface_hub
```

## Core Concepts Explained

### Vision-Language Models (VLMs)

Vision-Language Models are AI systems that can process and understand both visual (images) and textual (language) inputs simultaneously. They bridge the gap between computer vision and natural language processing, enabling tasks like:

- **Image Captioning**: Generating descriptive text for images
- **Visual Question Answering (VQA)**: Answering questions about image content
- **Visual Reasoning**: Understanding relationships and logic in visual scenes

**Paligemma** is Google's VLM that combines a vision encoder (SigLIP) with a language decoder (Gemma). It can process images and generate coherent text responses, making it ideal for tasks requiring both visual understanding and linguistic reasoning.

### CLEVR Dataset

CLEVR (Compositional Language and Elementary Visual Reasoning) is a synthetic dataset designed to test visual reasoning abilities. Key features:

- **Synthetic Scenes**: Computer-generated images with 3D objects (spheres, cubes, cylinders) in various colors, sizes, and materials
- **Structured Questions**: Questions that require understanding object properties, spatial relationships, and logical operations
- **Ground Truth Answers**: Deterministic answers based on scene composition
- **Compositionality**: Questions test different reasoning skills (counting, comparison, spatial reasoning, logical operations)

**Why CLEVR?**

- **Controlled Environment**: Eliminates real-world image variability (lighting, occlusion, etc.)
- **Reasoning Focus**: Emphasizes logical reasoning over pattern recognition
- **Evaluation**: Enables precise measurement of reasoning capabilities

### Parameter-Efficient Fine-Tuning (PEFT)

Traditional fine-tuning updates all model parameters, requiring significant computational resources. PEFT methods update only a small subset of parameters while achieving comparable performance.

**Low-Rank Adaptation (LoRA)**:

- **How it Works**: Instead of updating full weight matrices, LoRA adds trainable low-rank matrices (A and B) to frozen original weights
- **Mathematical Formulation**: For a weight matrix W, LoRA computes: W' = W + BA, where B ∈ ℝ^(d×r), A ∈ ℝ^(r×k), and r << min(d,k)
- **Benefits**:
  - **Memory Efficiency**: Only train ~1-2% of parameters
  - **Faster Training**: Reduced gradient computation and memory usage
  - **Modular Adaptation**: Easy to switch between different tasks
  - **Maintained Performance**: Often achieves 95-100% of full fine-tuning performance

### Quantization

Quantization reduces the precision of model weights to save memory and computation.

**8-bit Quantization (BitsAndBytes)**:

- **Process**: Converts 32-bit floating-point weights to 8-bit integers
- **Benefits**: ~4x memory reduction, faster inference, enables larger models on limited hardware
- **Trade-offs**: Slight accuracy loss, requires compatible hardware (modern GPUs)
- **Implementation**: Uses `BitsAndBytesConfig` with `load_in_8bit=True`

### Evaluation Metrics

#### ROUGE (Recall-Oriented Understudy for Gisting Evaluation)

ROUGE measures the overlap between generated and reference text:

- **ROUGE-1**: Unigram overlap (individual words)
- **ROUGE-2**: Bigram overlap (word pairs)
- **ROUGE-L**: Longest Common Subsequence (word order preservation)

**Interpretation**:

- Higher scores (0-1) indicate better overlap
- ROUGE-1: Measures basic word matching
- ROUGE-2: Measures phrase-level coherence
- ROUGE-L: Measures sentence structure preservation

#### Exact-Match Accuracy

- **Definition**: Percentage of predictions that exactly match reference answers
- **Calculation**: (Number of exact matches) / (Total predictions)
- **Normalization**: Case-insensitive, whitespace-stripped comparison
- **Use Case**: Suitable for CLEVR's deterministic, short answers

## Data Preparation

The notebook uses a 1% subset of the CLEVR dataset for demonstration purposes. The full dataset contains:

- Training: ~70,000 images with questions
- Validation: ~15,000 images
- Test: ~15,000 images

Data is automatically downloaded from Hugging Face and cached locally in `./data/` directory for persistence.

**Preprocessing Steps**:

1. **Image Processing**: Resize to 224×224, normalize pixel values
2. **Text Processing**: Tokenize questions, prepare answer labels
3. **Format Conversion**: Convert to model-compatible tensors

## Model Configuration

- **Base Model**: `google/paligemma-3b-pt-224`
- **LoRA Config**: r=8, alpha=32, target modules: `q_proj`, `o_proj`
- **Quantization**: 8-bit loading for memory efficiency
- **Input Processing**: Images resized to 224×224, text tokenized with max length 128
- **Output**: Generated answers with max length 384

## Training Setup

- **Batch Size**: 2 (train), 1 (eval)
- **Learning Rate**: 2e-5
- **Epochs**: 2
- **Gradient Accumulation**: 2 steps
- **Optimizer**: AdamW with 8-bit precision
- **Evaluation**: ROUGE-1, ROUGE-2, ROUGE-L scores + exact-match accuracy
- **Hardware**: Requires GPU with CUDA support

**Training Process**:

1. **Forward Pass**: Model processes image-question pairs
2. **Loss Computation**: Cross-entropy loss on generated tokens
3. **LoRA Update**: Only low-rank adapters updated
4. **Evaluation**: Periodic assessment on validation set

## How to Run

1. **Environment Setup**: Run the first cell to install dependencies
2. **Configuration**: Set random seeds and hyperparameters
3. **Data Loading**: Load and preprocess CLEVR dataset subset (caches to `./data/`)
4. **Model Preparation**: Load Paligemma with LoRA and quantization
5. **Training**: Execute `trainer.train()` (may take 2-3 hours)
6. **Evaluation**: Assess model performance with ROUGE metrics and exact-match
7. **Inference**: Test the model on new images

### Expected Training Time

- With Google Colab GPU: ~2-3 hours for 2 epochs
- Memory Requirements: ~8GB GPU RAM with 8-bit quantization

## Results Interpretation

After fine-tuning, evaluate performance using multiple metrics:

### Quantitative Metrics

- **ROUGE Scores**:

  - ROUGE-1 > 0.8: Good word-level overlap
  - ROUGE-2 > 0.6: Reasonable phrase matching
  - ROUGE-L > 0.7: Preserved sentence structure

- **Exact-Match Accuracy**:
  - > 0.7: Strong performance on CLEVR tasks
  - > 0.8: Excellent reasoning capability

### Qualitative Analysis

- **Sample Predictions**: Review generated answers for logical consistency
- **Error Analysis**: Identify common failure modes (counting, spatial reasoning, etc.)
- **Visualization**: Examine image-question pairs where model succeeds/fails

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

- Enable GPU runtime in Colab settings
- Monitor GPU memory usage during training
- Use local `./data/` caching instead of Drive for portability

## File Structure

```
CA4/
├── code/
│   └── CA4.ipynb          # Main notebook with complete implementation
├── description/
│   └── DGM_HW4.pdf        # Assignment description
├── report/
│   └── DGM_CA4_report.pdf # Student report
└── README.md              # This file
```

## Advanced Topics

### Generation-Based Evaluation

Unlike traditional classification metrics, VLM evaluation requires text generation:

- **Process**: Generate answers using `model.generate()`, decode to text
- **Comparison**: Compare generated text to reference answers
- **Metrics**: ROUGE for overlap, exact-match for precision

### Memory Management

- **Gradient Checkpointing**: Trade computation for memory
- **Mixed Precision**: FP16 training with BF16 weights
- **Quantization**: 8-bit weights reduce memory footprint
- **Batch Size Tuning**: Balance speed vs. memory constraints

### Hyperparameter Tuning

- **LoRA Rank (r)**: Higher r = more parameters = better performance (but slower)
- **Learning Rate**: 1e-5 to 5e-5 typically works well
- **Batch Size**: Limited by GPU memory, use gradient accumulation
- **Max Length**: Balance between answer completeness and computation

## References

- [Paligemma Paper](https://arxiv.org/abs/2407.07726)
- [CLEVR Dataset](https://cs.stanford.edu/people/jcjohns/clevr/)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
- [PEFT Library](https://huggingface.co/docs/peft/index)
- [BitsAndBytes Quantization](https://huggingface.co/docs/bitsandbytes/main/en/index)

## Acknowledgments

This implementation is based on the Deep Generative Models course assignment, utilizing state-of-the-art vision-language models and efficient fine-tuning techniques.
