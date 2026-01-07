"""
Inference Script for Math Reasoning Diffusion Model

Features:
- Interactive mode for testing
- Batch inference
- Multiple sampling strategies (DDPM, DDIM)
- Visualization of diffusion process
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional, List, Dict
import json

import torch
from transformers import GPT2Tokenizer
from tqdm import tqdm

from config import get_config, Config
from model import create_model, DiscreteDiffusion


def load_model(checkpoint_path: str, device: torch.device) -> tuple:
    """Load trained model from checkpoint"""
    print(f"Loading model from {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    config = checkpoint.get('config', get_config('default'))
    
    # Initialize tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({
        'additional_special_tokens': ['[QUESTION]', '[SOLUTION]', '[ANSWER]', '[MASK]']
    })
    
    # Update config with tokenizer vocab size
    config.model.vocab_size = len(tokenizer)
    
    # Create model
    model = create_model(config)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Load EMA weights if available
    if checkpoint.get('ema_shadow'):
        print("Using EMA weights")
        for name, param in model.named_parameters():
            if name in checkpoint['ema_shadow']:
                param.data = checkpoint['ema_shadow'][name].to(device)
    
    print(f"Model loaded successfully")
    return model, tokenizer, config


@torch.no_grad()
def generate(
    model: DiscreteDiffusion,
    tokenizer: GPT2Tokenizer,
    question: str,
    config: Config,
    device: torch.device,
    num_steps: Optional[int] = None,
    temperature: float = 1.0,
    top_p: float = 0.95,
    verbose: bool = True
) -> Dict:
    """Generate solution for a math question"""
    
    model.eval()
    
    # Format question
    question_text = f"[QUESTION] {question}"
    
    # Tokenize
    question_enc = tokenizer(
        question_text,
        max_length=config.data.max_question_len,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    question_ids = question_enc['input_ids'].to(device)
    question_mask = question_enc['attention_mask'].to(device)
    
    # Generate
    samples, history = model.sample(
        condition_ids=question_ids,
        condition_mask=question_mask,
        seq_len=config.data.max_answer_len,
        num_steps=num_steps or config.sampling.num_steps,
        temperature=temperature,
        top_p=top_p,
        verbose=verbose
    )
    
    # Decode
    generated_text = tokenizer.decode(samples[0], skip_special_tokens=True)
    
    # Parse solution and answer
    solution = generated_text
    answer = ""
    
    if "[ANSWER]" in generated_text:
        parts = generated_text.split("[ANSWER]")
        solution = parts[0].replace("[SOLUTION]", "").strip()
        if len(parts) > 1:
            answer = parts[1].strip()
    elif "[SOLUTION]" in generated_text:
        solution = generated_text.replace("[SOLUTION]", "").strip()
    
    # Decode history for visualization
    decoded_history = []
    for h in history:
        decoded_history.append(tokenizer.decode(h[0], skip_special_tokens=True)[:100])
    
    return {
        'question': question,
        'solution': solution,
        'answer': answer,
        'raw_output': generated_text,
        'history': decoded_history
    }


def interactive_mode(
    model: DiscreteDiffusion,
    tokenizer: GPT2Tokenizer,
    config: Config,
    device: torch.device
):
    """Interactive mode for testing"""
    print("\n" + "="*60)
    print("Math Reasoning Diffusion Model - Interactive Mode")
    print("="*60)
    print("Enter a math question to get a step-by-step solution.")
    print("Type 'quit' or 'exit' to stop.")
    print("Type 'settings' to change generation settings.")
    print("="*60 + "\n")
    
    # Default settings
    num_steps = config.sampling.num_steps
    temperature = config.sampling.temperature
    top_p = config.sampling.top_p or 0.95
    
    while True:
        try:
            question = input("\n📝 Question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if question.lower() == 'settings':
                print(f"\nCurrent settings:")
                print(f"  num_steps: {num_steps}")
                print(f"  temperature: {temperature}")
                print(f"  top_p: {top_p}")
                
                try:
                    new_steps = input(f"  new num_steps [{num_steps}]: ").strip()
                    if new_steps:
                        num_steps = int(new_steps)
                    
                    new_temp = input(f"  new temperature [{temperature}]: ").strip()
                    if new_temp:
                        temperature = float(new_temp)
                    
                    new_top_p = input(f"  new top_p [{top_p}]: ").strip()
                    if new_top_p:
                        top_p = float(new_top_p)
                        
                    print("Settings updated!")
                except ValueError as e:
                    print(f"Invalid input: {e}")
                continue
            
            if not question:
                continue
            
            print("\n🔄 Generating solution...")
            
            result = generate(
                model, tokenizer, question, config, device,
                num_steps=num_steps,
                temperature=temperature,
                top_p=top_p,
                verbose=True
            )
            
            print("\n" + "-"*40)
            print("💡 Solution:")
            print(result['solution'])
            if result['answer']:
                print(f"\n✅ Answer: {result['answer']}")
            print("-"*40)
            
            # Show diffusion process
            if result['history']:
                show_process = input("\nShow diffusion process? [y/N]: ").strip().lower()
                if show_process == 'y':
                    print("\n📊 Diffusion Process:")
                    for i, step in enumerate(result['history']):
                        print(f"  Step {i}: {step}...")
            
        except KeyboardInterrupt:
            print("\nInterrupted. Type 'quit' to exit.")
        except Exception as e:
            print(f"Error: {e}")


def batch_inference(
    model: DiscreteDiffusion,
    tokenizer: GPT2Tokenizer,
    config: Config,
    device: torch.device,
    input_file: str,
    output_file: str,
    num_steps: Optional[int] = None
):
    """Run batch inference on a file of questions"""
    print(f"Loading questions from {input_file}")
    
    # Load questions
    with open(input_file, 'r') as f:
        if input_file.endswith('.json'):
            data = json.load(f)
            if isinstance(data, list):
                questions = [item.get('question', item) if isinstance(item, dict) else item 
                           for item in data]
            else:
                questions = [data.get('question', str(data))]
        else:
            questions = [line.strip() for line in f if line.strip()]
    
    print(f"Processing {len(questions)} questions...")
    
    results = []
    for question in tqdm(questions, desc="Generating"):
        result = generate(
            model, tokenizer, question, config, device,
            num_steps=num_steps,
            verbose=False
        )
        results.append(result)
    
    # Save results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_file}")
    
    # Print summary
    print("\n" + "="*40)
    print("Sample Results:")
    print("="*40)
    for i, result in enumerate(results[:5]):
        print(f"\nQ{i+1}: {result['question'][:80]}...")
        print(f"A{i+1}: {result['answer']}")


def visualize_diffusion(
    model: DiscreteDiffusion,
    tokenizer: GPT2Tokenizer,
    config: Config,
    device: torch.device,
    question: str,
    num_steps: int = 100,
    save_path: Optional[str] = None
):
    """Visualize the diffusion process step by step"""
    print(f"Visualizing diffusion for: {question}")
    
    # Format question
    question_text = f"[QUESTION] {question}"
    question_enc = tokenizer(
        question_text,
        max_length=config.data.max_question_len,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    question_ids = question_enc['input_ids'].to(device)
    question_mask = question_enc['attention_mask'].to(device)
    
    # Modified sampling to capture all steps
    model.eval()
    batch_size = 1
    seq_len = config.data.max_answer_len
    
    # Start from fully masked
    x = torch.full((batch_size, seq_len), model.mask_token_id, device=device)
    
    steps_data = []
    step_size = model.timesteps // num_steps
    
    for i, t in enumerate(tqdm(range(model.timesteps, 0, -step_size), desc="Diffusion")):
        t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)
        
        # Get prediction
        logits = model.model(x, t_tensor, question_ids, question_mask)
        probs = torch.softmax(logits / config.sampling.temperature, dim=-1)
        samples = torch.multinomial(probs.view(-1, model.vocab_size), 1).view(batch_size, seq_len)
        
        # Unmask some tokens
        is_masked = (x == model.mask_token_id)
        current_mask_prob = model.mask_schedule[t]
        next_mask_prob = model.mask_schedule[max(0, t - step_size)]
        unmask_prob = (current_mask_prob - next_mask_prob) / (current_mask_prob + 1e-8)
        unmask = torch.rand_like(x.float()) < unmask_prob
        update_mask = is_masked & unmask
        x = torch.where(update_mask, samples, x)
        
        # Record state
        decoded = tokenizer.decode(x[0], skip_special_tokens=True)
        num_masked = is_masked.sum().item()
        steps_data.append({
            'step': i,
            'timestep': t,
            'num_masked': num_masked,
            'text': decoded[:200]
        })
    
    # Final prediction
    t_tensor = torch.ones((batch_size,), device=device, dtype=torch.long)
    logits = model.model(x, t_tensor, question_ids, question_mask)
    final = logits.argmax(dim=-1)
    is_masked = (x == model.mask_token_id)
    x = torch.where(is_masked, final, x)
    
    final_text = tokenizer.decode(x[0], skip_special_tokens=True)
    
    # Print visualization
    print("\n" + "="*60)
    print("DIFFUSION PROCESS VISUALIZATION")
    print("="*60)
    
    for data in steps_data[::max(1, len(steps_data)//10)]:  # Show 10 snapshots
        print(f"\nStep {data['step']} (t={data['timestep']}, masked={data['num_masked']}):")
        print(f"  {data['text'][:100]}...")
    
    print("\n" + "-"*60)
    print("FINAL OUTPUT:")
    print("-"*60)
    print(final_text)
    
    if save_path:
        with open(save_path, 'w') as f:
            json.dump({
                'question': question,
                'steps': steps_data,
                'final': final_text
            }, f, indent=2)
        print(f"\nVisualization saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Math Reasoning Diffusion Model Inference')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--mode', type=str, default='interactive',
                       choices=['interactive', 'batch', 'visualize'],
                       help='Inference mode')
    parser.add_argument('--input', type=str, default=None,
                       help='Input file for batch mode')
    parser.add_argument('--output', type=str, default='results.json',
                       help='Output file for batch mode')
    parser.add_argument('--question', type=str, default=None,
                       help='Question for single inference or visualization')
    parser.add_argument('--num_steps', type=int, default=None,
                       help='Number of sampling steps')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='Sampling temperature')
    parser.add_argument('--top_p', type=float, default=0.95,
                       help='Top-p sampling threshold')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    device = torch.device(args.device)
    
    # Load model
    model, tokenizer, config = load_model(args.checkpoint, device)
    
    # Override config with args
    if args.num_steps:
        config.sampling.num_steps = args.num_steps
    config.sampling.temperature = args.temperature
    config.sampling.top_p = args.top_p
    
    # Run inference
    if args.mode == 'interactive':
        interactive_mode(model, tokenizer, config, device)
    
    elif args.mode == 'batch':
        if not args.input:
            print("Error: --input required for batch mode")
            sys.exit(1)
        batch_inference(model, tokenizer, config, device, args.input, args.output)
    
    elif args.mode == 'visualize':
        if not args.question:
            args.question = "I have 5 apples. I buy 3 more apples. How many apples do I have now?"
        visualize_diffusion(model, tokenizer, config, device, args.question)


if __name__ == '__main__':
    main()
