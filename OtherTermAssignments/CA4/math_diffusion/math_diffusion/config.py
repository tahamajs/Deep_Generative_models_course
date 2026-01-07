"""
Configuration for Math Reasoning Diffusion Model
Optimized for 2x T4 GPUs (16GB each)
"""

import torch
from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class ModelConfig:
    """Model architecture configuration"""
    # Embedding dimensions
    hidden_dim: int = 512  # Reduced from 768 for T4 memory
    intermediate_dim: int = 2048
    
    # Transformer architecture
    num_layers: int = 8
    num_heads: int = 8
    dropout: float = 0.1
    
    # Vocabulary (will use GPT-2 tokenizer)
    vocab_size: int = 50257  # GPT-2 vocabulary size
    max_seq_len: int = 512
    
    # Diffusion settings
    diffusion_type: str = "discrete"  # "discrete" (MDLM-style) or "continuous"
    timesteps: int = 1000
    noise_schedule: str = "cosine"  # "linear", "cosine", "sqrt"
    
    # Self-conditioning (improves quality significantly)
    use_self_conditioning: bool = True
    self_cond_prob: float = 0.5


@dataclass
class TrainingConfig:
    """Training configuration optimized for 2x T4"""
    # Batch sizes
    batch_size_per_gpu: int = 4
    gradient_accumulation_steps: int = 8
    # Effective batch = 4 * 2 GPUs * 8 accum = 64
    
    # Learning rate
    learning_rate: float = 1e-4
    min_learning_rate: float = 1e-6
    weight_decay: float = 0.01
    
    # Schedule
    warmup_steps: int = 1000
    max_steps: int = 100000
    
    # Optimization
    max_grad_norm: float = 1.0
    use_amp: bool = True  # Mixed precision
    use_gradient_checkpointing: bool = True
    
    # EMA for stable generation
    use_ema: bool = True
    ema_decay: float = 0.9999
    
    # Logging
    log_every: int = 50
    eval_every: int = 1000
    save_every: int = 5000
    
    # Paths
    output_dir: str = "./outputs"
    checkpoint_path: Optional[str] = None


@dataclass
class DataConfig:
    """Dataset configuration"""
    dataset_name: str = "gsm8k"  # "gsm8k", "math", "combined"
    max_question_len: int = 128
    max_answer_len: int = 384  # CoT can be long
    
    # Preprocessing
    num_workers: int = 4
    prefetch_factor: int = 2
    
    # Data augmentation
    use_augmentation: bool = False


@dataclass
class SamplingConfig:
    """Inference/sampling configuration"""
    num_steps: int = 1000  # More steps = better quality
    sampler: str = "ddpm"  # "ddpm", "ddim", "ddpm_cache"
    
    # Temperature for discrete diffusion
    temperature: float = 1.0
    top_k: Optional[int] = None
    top_p: Optional[float] = 0.95
    
    # For morphing (question -> answer)
    morph_strength: float = 0.0  # 0 = start from mask, 1 = start from noised question


@dataclass
class Config:
    """Master configuration"""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    
    # Hardware
    seed: int = 42
    device: str = "cuda"
    num_gpus: int = 2
    
    def __post_init__(self):
        # Auto-detect GPUs
        if torch.cuda.is_available():
            self.num_gpus = torch.cuda.device_count()
            print(f"Detected {self.num_gpus} GPU(s)")
        else:
            self.device = "cpu"
            self.num_gpus = 0
            self.training.use_amp = False
            print("No GPU detected, using CPU")


def get_config(preset: str = "default") -> Config:
    """Get configuration preset"""
    config = Config()
    
    if preset == "small":
        # Smaller model for faster iteration
        config.model.hidden_dim = 256
        config.model.num_layers = 4
        config.model.intermediate_dim = 1024
        config.training.max_steps = 50000
        
    elif preset == "debug":
        # Minimal config for debugging
        config.model.hidden_dim = 128
        config.model.num_layers = 2
        config.model.intermediate_dim = 512
        config.model.max_seq_len = 128
        config.training.batch_size_per_gpu = 2
        config.training.max_steps = 1000
        config.training.log_every = 10
        config.training.eval_every = 100
        config.sampling.num_steps = 100
        
    elif preset == "large":
        # Larger model (may need gradient checkpointing)
        config.model.hidden_dim = 768
        config.model.num_layers = 12
        config.model.intermediate_dim = 3072
        config.training.batch_size_per_gpu = 2
        config.training.gradient_accumulation_steps = 16
        
    return config
