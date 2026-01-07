# Math Reasoning Diffusion Model

A **discrete diffusion language model** for mathematical reasoning with chain-of-thought generation. Based on state-of-the-art research including MDLM, Diffusion-of-Thought, and SEDD.

## Key Features

- **Discrete Masked Diffusion**: More stable than continuous diffusion for text generation
- **Chain-of-Thought Reasoning**: Generates step-by-step solutions, not just answers
- **Self-Conditioning**: Improves generation quality significantly
- **Bidirectional Attention**: Full attention (not causal) allows global planning
- **Multi-GPU Support**: Distributed training with DDP for 2x T4 GPUs
- **Memory Efficient**: FP16 training, gradient checkpointing, optimized for 16GB GPUs

## Architecture

```
Question → Encoder → [Condition Embeddings]
                            ↓
[Masked Tokens] → Transformer Denoiser → Predicted Tokens
                            ↑
                    [Time Embeddings]
```

The model learns to "unmask" the solution given the question as a condition. During inference, it starts from fully masked tokens and iteratively predicts the solution.

## Installation

```bash
# Clone or download the code
cd math_diffusion

# Install dependencies
pip install -r requirements.txt

# For distributed training
pip install torch>=2.0 --index-url https://download.pytorch.org/whl/cu118
```

## Quick Start

### 1. Test Setup

```bash
python main.py test
```

This verifies your environment, CUDA availability, and runs a quick model test.

### 2. Train the Model

```bash
# Single GPU training
python main.py train --dataset gsm8k --config default

# Multi-GPU training (2x T4)
torchrun --nproc_per_node=2 main.py train --dataset gsm8k

# Quick debug run
python main.py train --config debug --max_steps 1000
```

### 3. Run Inference

```bash
# Interactive mode
python main.py infer --checkpoint outputs/checkpoint_final.pt

# Batch inference
python main.py infer --checkpoint outputs/checkpoint_final.pt \
    --mode batch --input questions.json --output results.json

# Visualize diffusion process
python main.py infer --checkpoint outputs/checkpoint_final.pt \
    --mode visualize --question "What is 5 + 3 * 2?"
```

## Configuration Presets

| Preset | Parameters | Memory | Use Case |
|--------|------------|--------|----------|
| `debug` | ~2M | ~2GB | Quick testing |
| `small` | ~15M | ~4GB | Fast experiments |
| `default` | ~45M | ~8GB | Standard training |
| `large` | ~100M | ~14GB | Best quality |

```bash
python main.py train --config small --dataset gsm8k
```

## Datasets

### GSM8K (Recommended for starting)
- 8,500 grade-school math problems
- High-quality human-written chain-of-thought solutions
- Format: Question → Step-by-step reasoning → Final answer

### MATH
- 12,500 competition math problems
- LaTeX solutions with `\boxed{}` answers
- More challenging than GSM8K

### Combined
- Both datasets together for more diverse training

```bash
python main.py train --dataset combined
```

## Training Tips for 2x T4 GPUs

1. **Effective Batch Size**: With default settings, effective batch = 4 × 2 × 8 = 64
2. **Memory**: If OOM, reduce `batch_size_per_gpu` or use `--config small`
3. **Training Time**: 
   - GSM8K with default config: ~4-6 hours for 50K steps
   - Expect good results around 30K-50K steps
4. **Checkpoints**: Saved every 5000 steps in `outputs/`

### Monitor Training

```bash
# Watch training progress
tail -f outputs/training.log

# Check GPU usage
watch -n 1 nvidia-smi
```

## Model Architecture Details

### Discrete Diffusion (MDLM-style)

Unlike continuous diffusion that adds Gaussian noise to embeddings, we use **masked diffusion**:

- **Forward Process**: Gradually replace tokens with [MASK]
- **Reverse Process**: Predict original tokens from masked input
- **Loss**: Cross-entropy on masked positions only

This is more stable and produces better text than continuous approaches.

### Self-Conditioning

The model optionally receives its previous prediction as input:

```python
# During training (50% of the time)
prev_pred = model(noisy_input, t)
final_pred = model(noisy_input, t, self_cond=prev_pred)

# During inference
for t in timesteps:
    pred = model(x, t, self_cond=prev_pred)
    prev_pred = pred
```

This significantly improves generation quality.

### Time Conditioning

Uses **Adaptive Layer Normalization (adaLN)** from DiT:

```python
# Instead of: LayerNorm(x)
# We use: gamma(t) * LayerNorm(x) + beta(t)
```

This modulates the network based on the diffusion timestep.

## File Structure

```
math_diffusion/
├── main.py          # Entry point
├── config.py        # Configuration classes
├── model.py         # Diffusion model architecture
├── dataset.py       # Data loading (GSM8K, MATH)
├── train.py         # Training loop with DDP
├── inference.py     # Sampling and evaluation
├── requirements.txt # Dependencies
└── README.md        # This file
```

## Example Output

**Input**: "I have 5 apples. I buy 3 more apples, then give away 2. How many apples do I have?"

**Generated Solution**:
```
[SOLUTION] Let me solve this step by step.
1. Start with 5 apples
2. Buy 3 more: 5 + 3 = 8 apples
3. Give away 2: 8 - 2 = 6 apples
[ANSWER] 6
```

## Comparison with Your Original Code

| Aspect | Original | New Implementation |
|--------|----------|-------------------|
| Diffusion Type | Continuous (embedding space) | Discrete (masked tokens) |
| Encoder | Frozen BERT | Learned embeddings (GPT-2 tokenizer) |
| Architecture | TransformerDecoder | Full bidirectional Transformer |
| Time Conditioning | Simple addition | Adaptive Layer Norm (adaLN) |
| Self-Conditioning | No | Yes |
| Multi-GPU | DataParallel | DistributedDataParallel |
| Mixed Precision | No | Yes (FP16) |

## References

- [MDLM: Simple and Effective Masked Diffusion Language Models](https://arxiv.org/abs/2406.07524)
- [Diffusion-of-Thought: Chain-of-Thought Reasoning in Diffusion LMs](https://arxiv.org/abs/2402.07754)
- [SEDD: Score Entropy Discrete Diffusion](https://arxiv.org/abs/2310.16834)
- [Diffusion-LM: Controllable Text Generation](https://arxiv.org/abs/2205.14217)

## Troubleshooting

### CUDA Out of Memory
```bash
# Use smaller config
python main.py train --config small

# Or reduce batch size
python main.py train --batch_size 2
```

### Slow Training
```bash
# Ensure you're using both GPUs
torchrun --nproc_per_node=2 main.py train ...

# Check GPU utilization
nvidia-smi
```

### Poor Generation Quality
- Train for more steps (aim for 50K+)
- Use more sampling steps at inference (1000+)
- Lower temperature (0.7-0.9)
- Ensure self-conditioning is enabled

## License

MIT License - Feel free to use and modify for your research!
