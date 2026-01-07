# Diffusion models for math reasoning: a practitioner's guide

**Diffusion language models have reached a critical inflection point for mathematical reasoning.** Recent advances—particularly SEDD, MDLM, and the Diffusion-of-Thought (DoT) framework—demonstrate that discrete diffusion can now match GPT-2-scale autoregressive models while offering unique advantages for chain-of-thought reasoning: bidirectional self-correction, flexible compute-accuracy tradeoffs, and immunity to the "reversal curse" that plagues left-to-right generation. For building a working math reasoning system on 2x T4 GPUs, the optimal path combines **MDLM's masked diffusion architecture** with **learned embeddings**, trained on **GSM8K or MATH** with their high-quality CoT annotations.

---

## Discrete diffusion now dominates continuous approaches for text

The field has decisively shifted from continuous embedding-space diffusion (Diffusion-LM, 2022) toward discrete token-space methods. **SEDD** (Score Entropy Discrete Diffusion, ICML 2024 Best Paper) introduced a novel score-entropy loss that models probability ratios between neighboring sequences, achieving **25-75% perplexity reduction** over prior methods and outperforming GPT-2 at equivalent scale. **MDLM** (Masked Diffusion Language Models, NeurIPS 2024) simplified this further with SUBS parameterization—essentially treating diffusion as a weighted mixture of masked language modeling losses—delivering **17% better perplexity** than SEDD (23.00 vs 32.79 on LM1B) with 3-4x faster sampling via ddpm_cache.

The architectural choice between discrete and continuous diffusion has significant practical implications:

| Approach | Key Models | Pros | Cons |
|----------|-----------|------|------|
| **Discrete (masked/absorbing)** | SEDD, MDLM, LLaDA | Better perplexity, no rounding errors, direct connection to MLM | Harder gradient-based control |
| **Continuous (embedding-space)** | Diffusion-LM, PLAID | Enables gradient-based steering, smooth interpolation | Rounding errors, higher perplexity |

For math reasoning specifically, **discrete masked diffusion is recommended** because: (1) it avoids the embedding collapse and rounding errors that plague continuous methods, (2) MDLM's simplified loss function is equivalent to cross-entropy making training stable, and (3) the masking process naturally supports the "reveal step-by-step" nature of mathematical derivations.

**Transformer architecture modifications** for diffusion are surprisingly minimal. Unlike autoregressive models with causal attention masks, diffusion denoisers use **bidirectional (full) attention**—all tokens attend to all others for global context. Time conditioning can be handled via adaptive layer normalization (adaLN) where timestep embeddings modulate scale and shift parameters, though MDLM's RADD theory shows that for absorbing diffusion, explicit time conditioning may be unnecessary since the model implicitly infers noise level from mask ratio.

---

## Learned embeddings outperform pre-trained encoders for smaller models

The question of which embeddings to use has a nuanced answer that depends on model scale. For smaller models (under 1B parameters), **training embeddings end-to-end with the diffusion model consistently outperforms using frozen pre-trained embeddings** from BERT, GPT-2, or T5. Diffusion-LM's authors note: "We found that fixed embeddings are suboptimal compared to end-to-end training."

The problems with BERT embeddings specifically are threefold:

- **Anisotropic embedding space**: BERT produces non-smooth, frequency-biased embeddings where common tokens cluster differently than rare ones—problematic for diffusion's iterative denoising
- **MLM training mismatch**: BERT learns from only ~15% of tokens per pass (the masked positions), making it less sample-efficient than causal models that learn from 100% of tokens
- **Representation mismatch**: BERT's bidirectional encodings are optimized for understanding, not generation

If you must use pre-trained models, **use encoder outputs (contextualized representations) rather than raw embedding tables**. TEncDM (AAAI 2025) found that BERT/RoBERTa/T5 *encodings* (post-transformer layers) significantly outperform raw embeddings because they incorporate contextual information critical for diffusion denoising.

For continuous diffusion in embedding space, critical engineering practices include:

- **Anchor loss** to prevent embedding collapse during joint training—without this, embeddings can trivialize the MSE loss by collapsing to similar values
- **Layer normalization** on embeddings before forward diffusion (Difformer approach)
- **Low embedding dimensions** (16-128) often work better than high-dimensional pre-trained spaces (768+)
- **Noise rescaling** to ensure sufficient corruption in high-dimensional spaces

The decoder strategy for mapping continuous embeddings back to tokens matters significantly. Simple nearest-neighbor rounding accumulates errors; **context-dependent learned projection heads** (TEncDM approach) dramatically improve quality by considering surrounding context when selecting final tokens. Self-conditioning—feeding the model's previous prediction back as input—provides near-universal quality improvements at minimal cost.

---

## Diffusion-of-Thought enables self-correcting mathematical reasoning

**The key paper for math reasoning with diffusion is Diffusion-of-Thought (DoT)** (NeurIPS 2024), which integrates chain-of-thought generation into the diffusion process itself. Rather than generating reasoning steps left-to-right, DoT allows reasoning to "diffuse over time"—emerging globally through iterative denoising. This enables a remarkable capability: **self-correction of intermediate reasoning errors**.

In DoT experiments, the model initially produced wrong intermediate calculations (e.g., `<2*3=4>`) but arrived at the correct final answer. Subsequent denoising steps then corrected the intermediate error. Both prior and later reasoning steps inform current predictions—impossible in autoregressive generation where errors propagate irreversibly.

The advantages of diffusion over autoregressive CoT for reasoning are substantial:

- **Global planning**: Considers entire solution simultaneously rather than committing token-by-token
- **Error correction**: Self-correction through iterative refinement; early mistakes don't compound
- **Subgoal learning**: Effectively learns difficult subgoals that challenge AR models (MGDM achieves **91.5%** on Countdown vs 45.8% for AR; **100%** on Sudoku vs 20.7%)
- **Computation flexibility**: Trade sampling steps for accuracy—more denoising = better reasoning

However, DoT found that **pre-trained language understanding is vital**: training from scratch achieved only ~5% GSM8K accuracy, while fine-tuning pre-trained diffusion models (SEDD, PLAID) yielded strong performance. This suggests initializing from a pre-trained diffusion LM before fine-tuning on math reasoning data.

For maintaining coherence in long reasoning chains, several techniques help:

- **Segment-level diffusion** (SLD): Breaks long outputs into multiple latent representations with separate encoding, using adversarial and contrastive learning for robust representation
- **Block diffusion**: Generate text block-by-block semi-autoregressively, allowing flexible length while maintaining local coherence
- **EOS prediction**: DoT-MP appends special `<EOS>` tokens allowing the model to dynamically determine reasoning length

---

## GSM8K and MATH provide gold-standard reasoning annotations

For training a diffusion model on math reasoning, **GSM8K** (8,500 grade-school problems) and **MATH** (12,500 competition problems) are the essential datasets, distinguished by their high-quality human-written chain-of-thought solutions.

| Dataset | Size | Difficulty | CoT Quality | Key Features |
|---------|------|------------|-------------|--------------|
| **GSM8K** | 8.5K | Grade school | ★★★★★ | Human-annotated by STEM degree holders; `<<calculation=result>>` notation |
| **MATH** | 12.5K | Competition | ★★★★★ | Full LaTeX solutions; `\boxed{}` answers; 5 difficulty levels, 7 categories |
| **AQuA-RAT** | 100K | GRE/GMAT | ★★★☆☆ | Multiple-choice with rationales; crowdsourced (some noise) |
| **MathQA** | 37K | Mixed | ★★★★☆ | Cleaned AQuA with operational programs |
| **OpenMathReasoning** | 306K | Advanced | ★★★★★ | DeepSeek-R1/QwQ-generated; won AIMO-2 competition |

For **preprocessing diffusion training data**, the recommended format concatenates question and solution:

```
Question: [problem text]
Solution: [step-by-step reasoning]  
Answer: [final answer]
```

For partial noising (DiffuSeq-style conditional generation), keep question embeddings fixed while applying diffusion noise only to the solution portion. Fixed-length padding to 256-512 tokens works well; for variable-length handling, bucket sequences by similar length or use block diffusion for length-agnostic generation.

**Synthetic data augmentation** significantly improves performance. The MathScale approach extracts concepts from seed problems, builds a concept graph, and generates new problems via LLM guided by the graph—producing 2M problem-answer pairs. For quality filtering, the FLAMES framework found that **higher coverage beats strictly reliable solutions** given a fixed budget, and mixing data from multiple generation strategies provides robustness. Use execution-based verification for code-integrated solutions, or LLM-as-judge for reasoning quality assessment.

---

## Practical training: loss functions, noise schedules, and T4 optimization

The training objective for discrete masked diffusion (MDLM/SEDD style) is straightforward cross-entropy on masked positions, weighted by noise level:

```python
# MDLM-style training step
t = torch.rand(batch_size)  # Sample noise levels
noisy_batch = mask_tokens(batch, mask_prob=t)
logits = model(noisy_batch, t)
loss = cross_entropy(logits[masked_positions], batch[masked_positions])
```

For continuous diffusion, use **MSE loss** between predicted and actual clean embeddings, plus a **decoder NLL term** for token prediction, and optionally a **KL term** ensuring noisy embeddings approach standard Gaussian.

**Noise schedules** significantly impact quality. Cosine schedules (slower noise at extremes) or sqrt schedules generally outperform linear. For discrete diffusion, SEDD uses log-linear schedules. Training typically uses **1000-2000 diffusion steps**; inference can use fewer (500-1000 for speed, 4000+ for quality).

Key hyperparameters for text diffusion:

| Parameter | Recommended | Notes |
|-----------|-------------|-------|
| Learning rate | 1e-4 | With 500-5000 step warmup |
| Batch size | 128-256 effective | Via gradient accumulation |
| Diffusion steps (T) | 1000-2000 | More = better quality |
| EMA decay | 0.9999 | For stable generation |
| Sequence length | 256-512 | Task dependent |

**For 2x T4 GPUs (16GB each)**, the practical constraints are:

- **Model size**: ~130M parameters (MDLM small) fits comfortably; ~350M possible with heavy optimization; 770M+ requires different hardware
- **Enable mixed precision (fp16)** and **gradient checkpointing** to reduce memory 60% (at 20-30% speed cost)
- **Batch size**: 4-8 per GPU with 4-8 gradient accumulation steps for effective batch of 32-64
- **Use DDP** (DistributedDataParallel) for multi-GPU training:

```bash
torchrun --nproc_per_node=2 train.py
```

The **recommended starting configuration** for 2x T4:

```yaml
model: small  # ~130M params
model.length: 512
loader.batch_size: 4
training.accum: 4  # Effective: 4*2*4=32
fp16: true
gradient_checkpointing: true
sampling.predictor: ddpm_cache  # 3-4x faster inference
```

---

## GitHub repositories and code to start immediately

The best starting points for implementation, ranked by production-readiness:

**Tier 1 (Recommended)**:
- **[kuleshov-group/mdlm](https://github.com/kuleshov-group/mdlm)** (577 stars): MDLM official implementation with clean codebase, full training/inference, pretrained models on HuggingFace
- **[louaaron/Score-Entropy-Discrete-Diffusion](https://github.com/louaaron/Score-Entropy-Discrete-Diffusion)** (672 stars): SEDD official code with modular design
- **[ML-GSAI/LLaDA](https://github.com/ML-GSAI/LLaDA)**: 8B scale diffusion LLM codebase including pretraining and SFT

**Tier 2 (Research/Extended)**:
- **[kuleshov-group/bd3lms](https://github.com/kuleshov-group/bd3lms)**: Block diffusion (ICLR 2025) for variable-length generation
- **[XiangLi1999/Diffusion-LM](https://github.com/XiangLi1999/Diffusion-LM)**: Original continuous diffusion for text, controllable generation
- **[ZHZisZZ/dllm](https://github.com/ZHZisZZ/dllm)**: Unified library supporting MDLM, BD3LM, Edit Flows with HuggingFace Trainer integration

Quick start with MDLM:

```bash
conda env create -f requirements.yaml && conda activate mdlm

# Generate samples with pretrained model
python main.py mode=sample_eval \
  eval.checkpoint_path=kuleshov-group/mdlm-owt \
  sampling.predictor=ddpm_cache sampling.steps=1000

# Train from scratch  
python main.py model=small data=openwebtext-split \
  parameterization=subs model.length=512 sampling.steps=1000
```

Common pitfalls and solutions:

- **Embedding collapse** (continuous diffusion): Use anchor loss + layer normalization, or switch to discrete diffusion
- **Training instability**: Gradient clipping (max_norm=1.0), sqrt noise schedule, learning rate warmup
- **Poor generation quality**: Increase sampling steps to 4000+, check train/inference noise schedule alignment, enable self-conditioning
- **KV-cache incompatibility**: Use Block Diffusion (BD3-LMs) for semi-autoregressive generation with caching

---

## Conclusion: the optimal architecture for math chain-of-thought

The research points to a clear implementation strategy for building a diffusion-based math reasoning system:

**Architecture**: Use **MDLM's masked diffusion** as the foundation—it's simpler than SEDD while achieving better perplexity, and the absorbing-state (masking) process aligns naturally with revealing reasoning steps. The SUBS parameterization makes training equivalent to weighted cross-entropy, ensuring stability.

**Embeddings**: For 2x T4 scale, **train embeddings from scratch** jointly with the model using embedding dimension 128-256. If fine-tuning a larger pre-trained model, use T5 encoder outputs rather than raw BERT embeddings.

**Data pipeline**: Start with **GSM8K** (high-quality CoT, manageable size) for initial experiments, then scale to **MATH** + **synthetic augmentation** via LLM-generated solutions verified by answer matching. Format as `Question: [problem] Solution: [CoT] Answer: [result]` with partial noising on the solution portion.

**Training**: Cosine noise schedule, 1000 training steps, lr=1e-4 with warmup, effective batch size 32-64 via gradient accumulation on 2x T4. Enable fp16 and gradient checkpointing. Train for 200K-500K steps on math data.

**Inference**: Use ddpm_cache sampler with 1000-2000 steps for balanced speed/quality. For highest quality reasoning, increase to 4000+ steps. Consider fine-tuning on Diffusion-of-Thought style data to enable self-correction of intermediate reasoning errors.

The field is advancing rapidly—LLaDA recently demonstrated that diffusion can scale to 8B parameters and match LLaMA3's capabilities. While current 2x T4 constraints limit model size, the core architectural patterns established by MDLM and DoT provide a solid foundation that will transfer as hardware scales.