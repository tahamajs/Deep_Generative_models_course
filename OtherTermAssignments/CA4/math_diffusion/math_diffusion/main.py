#!/usr/bin/env python3
"""
Math Reasoning Diffusion Model - Main Entry Point

A discrete diffusion language model for mathematical reasoning
with chain-of-thought generation.

Usage:
    # Train on GSM8K with default settings (2x T4 GPUs)
    python main.py train --dataset gsm8k
    
    # Train with custom settings
    python main.py train --config small --max_steps 50000
    
    # Interactive inference
    python main.py infer --checkpoint outputs/checkpoint_final.pt
    
    # Batch inference
    python main.py infer --checkpoint outputs/checkpoint_final.pt --mode batch --input questions.json
    
    # Quick test
    python main.py test
"""

import os
import sys
import argparse


def train_command(args):
    """Run training"""
    from train import main as train_main
    
    # Build sys.argv for train.py
    train_args = ['train.py']
    train_args.extend(['--config', args.config])
    train_args.extend(['--output_dir', args.output_dir])
    train_args.extend(['--dataset', args.dataset])
    
    if args.checkpoint:
        train_args.extend(['--checkpoint', args.checkpoint])
    if args.max_steps:
        train_args.extend(['--max_steps', str(args.max_steps)])
    if args.batch_size:
        train_args.extend(['--batch_size', str(args.batch_size)])
    
    sys.argv = train_args
    train_main()


def infer_command(args):
    """Run inference"""
    from inference import main as infer_main
    
    # Build sys.argv for inference.py
    infer_args = ['inference.py']
    infer_args.extend(['--checkpoint', args.checkpoint])
    infer_args.extend(['--mode', args.mode])
    
    if args.input:
        infer_args.extend(['--input', args.input])
    if args.output:
        infer_args.extend(['--output', args.output])
    if args.question:
        infer_args.extend(['--question', args.question])
    if args.num_steps:
        infer_args.extend(['--num_steps', str(args.num_steps)])
    
    infer_args.extend(['--temperature', str(args.temperature)])
    infer_args.extend(['--top_p', str(args.top_p)])
    infer_args.extend(['--device', args.device])
    
    sys.argv = infer_args
    infer_main()


def test_command(args):
    """Run quick test to verify setup"""
    print("="*60)
    print("Running Quick Test")
    print("="*60)
    
    import torch
    print(f"\nPyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA devices: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            mem = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"         Memory: {mem:.1f} GB")
    
    print("\n" + "-"*40)
    print("Testing imports...")
    
    try:
        from transformers import GPT2Tokenizer
        print("✓ transformers")
    except ImportError as e:
        print(f"✗ transformers: {e}")
    
    try:
        from datasets import load_dataset
        print("✓ datasets")
    except ImportError as e:
        print(f"✗ datasets: {e}")
    
    try:
        from config import get_config
        print("✓ config")
    except ImportError as e:
        print(f"✗ config: {e}")
    
    try:
        from model import create_model
        print("✓ model")
    except ImportError as e:
        print(f"✗ model: {e}")
    
    try:
        from dataset import MathReasoningDataset
        print("✓ dataset")
    except ImportError as e:
        print(f"✗ dataset: {e}")
    
    print("\n" + "-"*40)
    print("Testing model creation...")
    
    try:
        from config import get_config
        from model import create_model
        
        config = get_config('debug')  # Small config for quick test
        model = create_model(config)
        
        num_params = sum(p.numel() for p in model.parameters())
        print(f"✓ Model created: {num_params:,} parameters")
        
        # Test forward pass
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        batch_size = 2
        seq_len = 64
        cond_len = 32
        
        x = torch.randint(0, 1000, (batch_size, seq_len), device=device)
        t = torch.randint(1, 100, (batch_size,), device=device)
        cond = torch.randint(0, 1000, (batch_size, cond_len), device=device)
        
        with torch.no_grad():
            loss_dict = model.compute_loss(x, condition_ids=cond)
        
        print(f"✓ Forward pass: loss = {loss_dict['loss'].item():.4f}")
        
        # Test sampling
        samples, _ = model.sample(
            condition_ids=cond,
            seq_len=32,
            num_steps=10,
            verbose=False
        )
        print(f"✓ Sampling: shape = {samples.shape}")
        
    except Exception as e:
        print(f"✗ Model test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "-"*40)
    print("Testing dataset loading...")
    
    try:
        from dataset import MathReasoningDataset
        from transformers import GPT2Tokenizer
        
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
        
        # Try synthetic data first (always works)
        dataset = MathReasoningDataset(
            dataset_name='gsm8k',
            split='train',
            tokenizer=tokenizer,
            max_question_len=64,
            max_answer_len=128
        )
        
        sample = dataset[0]
        print(f"✓ Dataset loaded: {len(dataset)} samples")
        print(f"  Question shape: {sample['question_ids'].shape}")
        print(f"  Solution shape: {sample['solution_ids'].shape}")
        
    except Exception as e:
        print(f"✗ Dataset test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("Test Complete!")
    print("="*60)
    
    print("\nQuick Start:")
    print("  Train:  python main.py train --config debug --max_steps 1000")
    print("  Infer:  python main.py infer --checkpoint outputs/checkpoint_1000.pt")


def main():
    parser = argparse.ArgumentParser(
        description='Math Reasoning Diffusion Model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py test                              # Verify setup
  python main.py train --dataset gsm8k             # Train on GSM8K
  python main.py train --config debug              # Quick debug training
  python main.py infer --checkpoint model.pt       # Interactive inference
  
For distributed training (2 GPUs):
  torchrun --nproc_per_node=2 main.py train --dataset gsm8k
"""
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--config', type=str, default='default',
                             choices=['default', 'small', 'debug', 'large'],
                             help='Configuration preset')
    train_parser.add_argument('--output_dir', type=str, default='./outputs',
                             help='Output directory')
    train_parser.add_argument('--checkpoint', type=str, default=None,
                             help='Resume from checkpoint')
    train_parser.add_argument('--max_steps', type=int, default=None,
                             help='Max training steps')
    train_parser.add_argument('--batch_size', type=int, default=None,
                             help='Batch size per GPU')
    train_parser.add_argument('--dataset', type=str, default='gsm8k',
                             choices=['gsm8k', 'math', 'combined'],
                             help='Dataset to use')
    
    # Inference command
    infer_parser = subparsers.add_parser('infer', help='Run inference')
    infer_parser.add_argument('--checkpoint', type=str, required=True,
                             help='Path to checkpoint')
    infer_parser.add_argument('--mode', type=str, default='interactive',
                             choices=['interactive', 'batch', 'visualize'],
                             help='Inference mode')
    infer_parser.add_argument('--input', type=str, default=None,
                             help='Input file for batch mode')
    infer_parser.add_argument('--output', type=str, default='results.json',
                             help='Output file')
    infer_parser.add_argument('--question', type=str, default=None,
                             help='Question for single inference')
    infer_parser.add_argument('--num_steps', type=int, default=None,
                             help='Sampling steps')
    infer_parser.add_argument('--temperature', type=float, default=1.0,
                             help='Sampling temperature')
    infer_parser.add_argument('--top_p', type=float, default=0.95,
                             help='Top-p sampling')
    infer_parser.add_argument('--device', type=str, default='cuda',
                             help='Device (cuda/cpu)')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Run quick test')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train_command(args)
    elif args.command == 'infer':
        infer_command(args)
    elif args.command == 'test':
        test_command(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
