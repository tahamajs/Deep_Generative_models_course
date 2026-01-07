"""
Training Script for Math Reasoning Diffusion Model

Features:
- Distributed training (DDP) for multi-GPU
- Mixed precision (fp16) training
- Gradient checkpointing for memory efficiency
- EMA model for stable generation
- Wandb logging (optional)
- Checkpointing and resumption
"""

import os
import sys
import math
import time
import argparse
from pathlib import Path
from typing import Optional, Dict
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

# Local imports
from config import get_config, Config
from model import create_model, DiscreteDiffusion
from dataset import create_dataloaders


class EMA:
    """Exponential Moving Average for model weights"""
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = (
                    self.decay * self.shadow[name] + 
                    (1 - self.decay) * param.data
                )
    
    def apply_shadow(self):
        """Apply EMA weights for inference"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        """Restore original weights after inference"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


def setup_distributed():
    """Initialize distributed training"""
    if 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        
        return rank, local_rank, world_size
    else:
        return 0, 0, 1


def cleanup_distributed():
    """Clean up distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()


def get_lr_scheduler(optimizer, config: Config, num_training_steps: int):
    """Create learning rate scheduler with warmup"""
    from torch.optim.lr_scheduler import LambdaLR
    
    warmup_steps = config.training.warmup_steps
    
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, warmup_steps))
        else:
            # Cosine decay
            progress = float(current_step - warmup_steps) / float(
                max(1, num_training_steps - warmup_steps)
            )
            return max(
                config.training.min_learning_rate / config.training.learning_rate,
                0.5 * (1.0 + math.cos(math.pi * progress))
            )
    
    return LambdaLR(optimizer, lr_lambda)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: Optional[GradScaler],
    ema: Optional[EMA],
    step: int,
    config: Config,
    path: str
):
    """Save training checkpoint"""
    # Get raw model (unwrap DDP if needed)
    raw_model = model.module if hasattr(model, 'module') else model
    
    checkpoint = {
        'step': step,
        'model_state_dict': raw_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'scaler_state_dict': scaler.state_dict() if scaler else None,
        'ema_shadow': ema.shadow if ema else None,
        'config': config,
    }
    
    torch.save(checkpoint, path)
    print(f"Saved checkpoint to {path}")


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler = None,
    scaler: Optional[GradScaler] = None,
    ema: Optional[EMA] = None
) -> int:
    """Load training checkpoint"""
    checkpoint = torch.load(path, map_location='cpu')
    
    # Get raw model
    raw_model = model.module if hasattr(model, 'module') else model
    raw_model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and checkpoint.get('scheduler_state_dict'):
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    if scaler and checkpoint.get('scaler_state_dict'):
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    if ema and checkpoint.get('ema_shadow'):
        ema.shadow = checkpoint['ema_shadow']
    
    print(f"Loaded checkpoint from {path} at step {checkpoint['step']}")
    return checkpoint['step']


@torch.no_grad()
def evaluate(
    model: DiscreteDiffusion,
    eval_loader,
    tokenizer,
    config: Config,
    device: torch.device,
    num_samples: int = 5
) -> Dict[str, float]:
    """Evaluate model on validation set"""
    model.eval()
    raw_model = model.module if hasattr(model, 'module') else model
    
    total_loss = 0
    total_acc = 0
    num_batches = 0
    
    for batch in eval_loader:
        question_ids = batch['question_ids'].to(device)
        question_mask = batch['question_mask'].to(device)
        solution_ids = batch['solution_ids'].to(device)
        solution_mask = batch['solution_mask'].to(device)
        
        metrics = raw_model.compute_loss(
            solution_ids,
            condition_ids=question_ids,
            condition_mask=question_mask,
            attention_mask=solution_mask,
            self_cond_prob=0.0  # No self-conditioning during eval
        )
        
        total_loss += metrics['loss'].item()
        total_acc += metrics['accuracy'].item()
        num_batches += 1
        
        if num_batches >= 50:  # Limit eval batches
            break
    
    # Generate some samples
    print("\n--- Sample Generations ---")
    batch = next(iter(eval_loader))
    question_ids = batch['question_ids'][:num_samples].to(device)
    question_mask = batch['question_mask'][:num_samples].to(device)
    
    samples, _ = raw_model.sample(
        condition_ids=question_ids,
        condition_mask=question_mask,
        seq_len=config.data.max_answer_len,
        num_steps=min(100, config.sampling.num_steps),  # Fewer steps for quick eval
        temperature=config.sampling.temperature,
        top_p=config.sampling.top_p,
        verbose=False
    )
    
    for i in range(min(num_samples, len(samples))):
        question = tokenizer.decode(question_ids[i], skip_special_tokens=True)
        generated = tokenizer.decode(samples[i], skip_special_tokens=True)
        if 'question_text' in batch:
            target = batch['solution_text'][i]
        else:
            target = "N/A"
        
        print(f"\nQ: {question[:100]}...")
        print(f"Generated: {generated[:200]}...")
        print(f"Target: {target[:200]}...")
    
    model.train()
    
    return {
        'eval_loss': total_loss / num_batches,
        'eval_accuracy': total_acc / num_batches
    }


def train(config: Config):
    """Main training loop"""
    
    # Setup distributed training
    rank, local_rank, world_size = setup_distributed()
    is_main = rank == 0
    
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    
    if is_main:
        print(f"Training on {world_size} GPU(s)")
        print(f"Config: {config}")
    
    # Set seed for reproducibility
    torch.manual_seed(config.seed + rank)
    
    # Create output directory
    output_dir = Path(config.training.output_dir)
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create dataloaders
    train_loader, eval_loader, tokenizer = create_dataloaders(config)
    
    # Update vocab size based on tokenizer
    config.model.vocab_size = len(tokenizer)
    
    if is_main:
        print(f"Vocabulary size: {config.model.vocab_size}")
        print(f"Training samples: {len(train_loader.dataset)}")
    
    # Create model
    model = create_model(config)
    model = model.to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if is_main:
        print(f"Model parameters: {num_params:,} ({num_params/1e6:.1f}M)")
    
    # Enable gradient checkpointing
    if config.training.use_gradient_checkpointing:
        # Apply to transformer layers
        for layer in model.model.layers:
            layer.gradient_checkpointing = True
    
    # Wrap with DDP for distributed training
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        betas=(0.9, 0.999)
    )
    
    # Create scheduler
    num_training_steps = config.training.max_steps
    scheduler = get_lr_scheduler(optimizer, config, num_training_steps)
    
    # Create gradient scaler for mixed precision
    scaler = GradScaler() if config.training.use_amp else None
    
    # Create EMA
    raw_model = model.module if hasattr(model, 'module') else model
    ema = EMA(raw_model, decay=config.training.ema_decay) if config.training.use_ema else None
    
    # Load checkpoint if resuming
    start_step = 0
    if config.training.checkpoint_path:
        start_step = load_checkpoint(
            config.training.checkpoint_path,
            model, optimizer, scheduler, scaler, ema
        )
    
    # Training loop
    model.train()
    step = start_step
    epoch = 0
    
    # For distributed training, create sampler
    train_sampler = DistributedSampler(
        train_loader.dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    ) if world_size > 1 else None
    
    if train_sampler:
        train_loader = torch.utils.data.DataLoader(
            train_loader.dataset,
            batch_size=config.training.batch_size_per_gpu,
            sampler=train_sampler,
            num_workers=config.data.num_workers,
            pin_memory=True,
            collate_fn=train_loader.collate_fn
        )
    
    # Metrics tracking
    running_loss = 0
    running_acc = 0
    log_steps = 0
    start_time = time.time()
    
    if is_main:
        pbar = tqdm(total=config.training.max_steps - start_step, desc="Training")
    
    while step < config.training.max_steps:
        if train_sampler:
            train_sampler.set_epoch(epoch)
        
        for batch in train_loader:
            if step >= config.training.max_steps:
                break
            
            # Move to device
            question_ids = batch['question_ids'].to(device)
            question_mask = batch['question_mask'].to(device)
            solution_ids = batch['solution_ids'].to(device)
            solution_mask = batch['solution_mask'].to(device)
            
            # Forward pass with mixed precision
            amp_context = autocast() if config.training.use_amp else nullcontext()
            
            with amp_context:
                metrics = (model.module if hasattr(model, 'module') else model).compute_loss(
                    solution_ids,
                    condition_ids=question_ids,
                    condition_mask=question_mask,
                    attention_mask=solution_mask,
                    self_cond_prob=config.model.self_cond_prob
                )
                loss = metrics['loss']
                
                # Scale loss for gradient accumulation
                loss = loss / config.training.gradient_accumulation_steps
            
            # Backward pass
            if config.training.use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Track metrics
            running_loss += metrics['loss'].item()
            running_acc += metrics['accuracy'].item()
            log_steps += 1
            
            # Gradient accumulation step
            if (step + 1) % config.training.gradient_accumulation_steps == 0:
                # Gradient clipping
                if config.training.use_amp:
                    scaler.unscale_(optimizer)
                
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 
                    config.training.max_grad_norm
                )
                
                # Optimizer step
                if config.training.use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                
                scheduler.step()
                optimizer.zero_grad()
                
                # Update EMA
                if ema:
                    ema.update()
            
            step += 1
            
            # Logging
            if is_main and step % config.training.log_every == 0:
                avg_loss = running_loss / log_steps
                avg_acc = running_acc / log_steps
                elapsed = time.time() - start_time
                steps_per_sec = log_steps / elapsed
                
                pbar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'acc': f'{avg_acc:.3f}',
                    'lr': f'{scheduler.get_last_lr()[0]:.2e}',
                    'steps/s': f'{steps_per_sec:.2f}'
                })
                pbar.update(config.training.log_every)
                
                # Reset metrics
                running_loss = 0
                running_acc = 0
                log_steps = 0
                start_time = time.time()
            
            # Evaluation
            if is_main and step % config.training.eval_every == 0:
                # Use EMA weights for evaluation
                if ema:
                    ema.apply_shadow()
                
                eval_metrics = evaluate(
                    model, eval_loader, tokenizer, config, device
                )
                
                print(f"\nStep {step} - Eval Loss: {eval_metrics['eval_loss']:.4f}, "
                      f"Eval Acc: {eval_metrics['eval_accuracy']:.3f}")
                
                if ema:
                    ema.restore()
            
            # Save checkpoint
            if is_main and step % config.training.save_every == 0:
                save_checkpoint(
                    model, optimizer, scheduler, scaler, ema, step, config,
                    str(output_dir / f'checkpoint_{step}.pt')
                )
        
        epoch += 1
    
    # Final save
    if is_main:
        save_checkpoint(
            model, optimizer, scheduler, scaler, ema, step, config,
            str(output_dir / 'checkpoint_final.pt')
        )
        pbar.close()
    
    cleanup_distributed()
    print("Training complete!")


def main():
    parser = argparse.ArgumentParser(description='Train Math Reasoning Diffusion Model')
    parser.add_argument('--config', type=str, default='default',
                       choices=['default', 'small', 'debug', 'large'],
                       help='Configuration preset')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                       help='Output directory')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Resume from checkpoint')
    parser.add_argument('--max_steps', type=int, default=None,
                       help='Override max training steps')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Override batch size per GPU')
    parser.add_argument('--dataset', type=str, default='gsm8k',
                       choices=['gsm8k', 'math', 'combined'],
                       help='Dataset to use')
    
    args = parser.parse_args()
    
    # Get config
    config = get_config(args.config)
    
    # Override with command line args
    config.training.output_dir = args.output_dir
    if args.checkpoint:
        config.training.checkpoint_path = args.checkpoint
    if args.max_steps:
        config.training.max_steps = args.max_steps
    if args.batch_size:
        config.training.batch_size_per_gpu = args.batch_size
    config.data.dataset_name = args.dataset
    
    # Train!
    train(config)


if __name__ == '__main__':
    main()
