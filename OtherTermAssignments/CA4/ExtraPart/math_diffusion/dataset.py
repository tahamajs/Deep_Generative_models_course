"""
Dataset Module for Math Reasoning Diffusion

Supports:
- GSM8K (grade school math with CoT)
- MATH (competition math with CoT)
- Combined datasets
- Data augmentation

Format: Question -> Chain of Thought -> Answer
"""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer
from datasets import load_dataset, concatenate_datasets
from typing import Optional, Dict, List, Tuple
import re
import random


class MathReasoningDataset(Dataset):
    """
    Dataset for math reasoning with chain-of-thought
    
    Each sample contains:
    - question: The math problem
    - solution: Step-by-step reasoning (CoT)
    - answer: Final numerical answer
    
    The model learns to generate: solution + answer given question
    """
    
    def __init__(
        self,
        dataset_name: str = "gsm8k",
        split: str = "train",
        tokenizer: Optional[GPT2Tokenizer] = None,
        max_question_len: int = 128,
        max_answer_len: int = 384,
        use_augmentation: bool = False
    ):
        self.dataset_name = dataset_name
        self.split = split
        self.max_question_len = max_question_len
        self.max_answer_len = max_answer_len
        self.use_augmentation = use_augmentation
        
        # Initialize tokenizer
        if tokenizer is None:
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            # Add special tokens
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.add_special_tokens({
                'additional_special_tokens': ['[QUESTION]', '[SOLUTION]', '[ANSWER]', '[MASK]']
            })
        else:
            self.tokenizer = tokenizer
        
        self.mask_token_id = self.tokenizer.convert_tokens_to_ids('[MASK]')
        if self.mask_token_id == self.tokenizer.unk_token_id:
            # Fallback if [MASK] wasn't added properly
            self.mask_token_id = self.tokenizer.eos_token_id
        
        # Load dataset
        self.data = self._load_dataset()
        print(f"Loaded {len(self.data)} samples from {dataset_name} ({split})")
        
    def _load_dataset(self) -> List[Dict]:
        """Load and preprocess dataset"""
        if self.dataset_name == "gsm8k":
            return self._load_gsm8k()
        elif self.dataset_name == "math":
            return self._load_math()
        elif self.dataset_name == "combined":
            gsm8k = self._load_gsm8k()
            math_data = self._load_math()
            return gsm8k + math_data
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
    
    def _load_gsm8k(self) -> List[Dict]:
        """Load GSM8K dataset"""
        try:
            dataset = load_dataset("gsm8k", "main", split=self.split)
        except Exception as e:
            print(f"Error loading GSM8K: {e}")
            print("Using synthetic data for testing...")
            return self._create_synthetic_data(100)
        
        processed = []
        for item in dataset:
            question = item['question'].strip()
            answer_text = item['answer']
            
            # GSM8K format: reasoning with <<calc=result>> then #### final_answer
            # Extract the reasoning and final answer
            parts = answer_text.split('####')
            if len(parts) == 2:
                solution = parts[0].strip()
                final_answer = parts[1].strip()
            else:
                solution = answer_text
                final_answer = ""
            
            # Clean up the <<calculation>> format
            solution = re.sub(r'<<[^>]+>>', '', solution).strip()
            solution = ' '.join(solution.split())  # Normalize whitespace
            
            processed.append({
                'question': question,
                'solution': solution,
                'answer': final_answer
            })
        
        return processed
    
    def _load_math(self) -> List[Dict]:
        """Load MATH dataset"""
        try:
            dataset = load_dataset("hendrycks/competition_math", split=self.split)
        except Exception as e:
            print(f"Error loading MATH dataset: {e}")
            return []
        
        processed = []
        for item in dataset:
            question = item['problem'].strip()
            solution = item['solution'].strip()
            
            # Extract answer from \boxed{}
            answer_match = re.search(r'\\boxed\{([^}]+)\}', solution)
            if answer_match:
                final_answer = answer_match.group(1)
            else:
                final_answer = ""
            
            # Clean LaTeX for simpler processing
            solution = self._clean_latex(solution)
            
            processed.append({
                'question': question,
                'solution': solution,
                'answer': final_answer
            })
        
        return processed
    
    def _clean_latex(self, text: str) -> str:
        """Clean LaTeX formatting for simpler tokenization"""
        # Remove \boxed{} but keep content
        text = re.sub(r'\\boxed\{([^}]+)\}', r'\1', text)
        # Simplify common LaTeX
        text = text.replace('\\frac', '/')
        text = text.replace('\\cdot', '*')
        text = text.replace('\\times', '*')
        text = re.sub(r'\$+', '', text)  # Remove dollar signs
        return text.strip()
    
    def _create_synthetic_data(self, n: int) -> List[Dict]:
        """Create synthetic math data for testing"""
        data = []
        operations = [
            ("addition", lambda a, b: (f"{a} + {b}", a + b)),
            ("subtraction", lambda a, b: (f"{a} - {b}", a - b)),
            ("multiplication", lambda a, b: (f"{a} × {b}", a * b)),
        ]
        
        templates = [
            "What is {expr}?",
            "Calculate {expr}.",
            "Find the result of {expr}.",
            "Compute {expr}.",
        ]
        
        solution_templates = [
            "Let me solve this step by step. We need to calculate {expr}. The result is {ans}.",
            "To find {expr}, I'll compute it directly. {a} and {b} gives us {ans}.",
            "Step 1: Identify the operation. Step 2: Calculate {expr} = {ans}.",
        ]
        
        for i in range(n):
            a = random.randint(1, 100)
            b = random.randint(1, 100)
            op_name, op_func = random.choice(operations)
            expr, result = op_func(a, b)
            
            question = random.choice(templates).format(expr=expr)
            solution = random.choice(solution_templates).format(
                expr=expr, a=a, b=b, ans=result
            )
            
            data.append({
                'question': question,
                'solution': solution,
                'answer': str(result)
            })
        
        return data
    
    def _augment(self, item: Dict) -> Dict:
        """Apply data augmentation"""
        if not self.use_augmentation:
            return item
        
        # Random number substitution (keep structure, change values)
        # This is a simple form of augmentation
        question = item['question']
        solution = item['solution']
        answer = item['answer']
        
        # 50% chance to paraphrase question slightly
        if random.random() < 0.5:
            prefixes = ["", "Question: ", "Problem: ", "Solve: "]
            question = random.choice(prefixes) + question
        
        return {
            'question': question,
            'solution': solution,
            'answer': answer
        }
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        if self.use_augmentation:
            item = self._augment(item)
        
        # Format: [QUESTION] question [SOLUTION] solution [ANSWER] answer
        question_text = f"[QUESTION] {item['question']}"
        solution_text = f"[SOLUTION] {item['solution']} [ANSWER] {item['answer']}"
        
        # Tokenize question (condition)
        question_enc = self.tokenizer(
            question_text,
            max_length=self.max_question_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize solution (target)
        solution_enc = self.tokenizer(
            solution_text,
            max_length=self.max_answer_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'question_ids': question_enc['input_ids'].squeeze(0),
            'question_mask': question_enc['attention_mask'].squeeze(0),
            'solution_ids': solution_enc['input_ids'].squeeze(0),
            'solution_mask': solution_enc['attention_mask'].squeeze(0),
            # Keep raw text for debugging
            'question_text': item['question'],
            'solution_text': f"{item['solution']} Answer: {item['answer']}"
        }


def create_dataloaders(
    config,
    tokenizer: Optional[GPT2Tokenizer] = None
) -> Tuple[DataLoader, DataLoader, GPT2Tokenizer]:
    """Create train and eval dataloaders"""
    
    # Create datasets
    train_dataset = MathReasoningDataset(
        dataset_name=config.data.dataset_name,
        split="train",
        tokenizer=tokenizer,
        max_question_len=config.data.max_question_len,
        max_answer_len=config.data.max_answer_len,
        use_augmentation=config.data.use_augmentation
    )
    
    # Get tokenizer from dataset if not provided
    tokenizer = train_dataset.tokenizer
    
    # Try to load test split, fall back to train subset
    try:
        eval_dataset = MathReasoningDataset(
            dataset_name=config.data.dataset_name,
            split="test",
            tokenizer=tokenizer,
            max_question_len=config.data.max_question_len,
            max_answer_len=config.data.max_answer_len,
            use_augmentation=False
        )
    except Exception:
        print("No test split available, using subset of train for eval")
        # Use last 10% of train as eval
        eval_size = max(100, len(train_dataset) // 10)
        train_dataset.data = train_dataset.data[:-eval_size]
        eval_data = train_dataset.data[-eval_size:]
        
        eval_dataset = MathReasoningDataset(
            dataset_name=config.data.dataset_name,
            split="train",
            tokenizer=tokenizer,
            max_question_len=config.data.max_question_len,
            max_answer_len=config.data.max_answer_len,
            use_augmentation=False
        )
        eval_dataset.data = eval_data
    
    # Custom collate function to handle text fields
    def collate_fn(batch):
        result = {
            'question_ids': torch.stack([b['question_ids'] for b in batch]),
            'question_mask': torch.stack([b['question_mask'] for b in batch]),
            'solution_ids': torch.stack([b['solution_ids'] for b in batch]),
            'solution_mask': torch.stack([b['solution_mask'] for b in batch]),
        }
        # Keep text for debugging (only in eval)
        if 'question_text' in batch[0]:
            result['question_text'] = [b['question_text'] for b in batch]
            result['solution_text'] = [b['solution_text'] for b in batch]
        return result
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size_per_gpu,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=config.training.batch_size_per_gpu,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    return train_loader, eval_loader, tokenizer


class MetaMathDataset(Dataset):
    """
    MetaMathQA dataset - high quality synthetic math data
    Much larger than GSM8K with good CoT annotations
    """
    
    def __init__(
        self,
        split: str = "train",
        tokenizer: Optional[GPT2Tokenizer] = None,
        max_question_len: int = 128,
        max_answer_len: int = 384,
        max_samples: Optional[int] = None
    ):
        self.max_question_len = max_question_len
        self.max_answer_len = max_answer_len
        
        # Initialize tokenizer
        if tokenizer is None:
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.add_special_tokens({
                'additional_special_tokens': ['[QUESTION]', '[SOLUTION]', '[ANSWER]', '[MASK]']
            })
        else:
            self.tokenizer = tokenizer
        
        self.mask_token_id = self.tokenizer.eos_token_id
        
        # Load MetaMathQA
        try:
            dataset = load_dataset("meta-math/MetaMathQA", split=split)
            if max_samples:
                dataset = dataset.select(range(min(max_samples, len(dataset))))
            self.data = self._process_metamath(dataset)
            print(f"Loaded {len(self.data)} samples from MetaMathQA")
        except Exception as e:
            print(f"Error loading MetaMathQA: {e}")
            self.data = []
    
    def _process_metamath(self, dataset) -> List[Dict]:
        """Process MetaMathQA format"""
        processed = []
        for item in dataset:
            question = item.get('query', item.get('question', ''))
            response = item.get('response', item.get('answer', ''))
            
            # Extract final answer if present
            answer_match = re.search(r'(?:answer is|=)\s*([0-9,.-]+)', response, re.IGNORECASE)
            final_answer = answer_match.group(1) if answer_match else ""
            
            processed.append({
                'question': question.strip(),
                'solution': response.strip(),
                'answer': final_answer
            })
        
        return processed
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        
        question_text = f"[QUESTION] {item['question']}"
        solution_text = f"[SOLUTION] {item['solution']} [ANSWER] {item['answer']}"
        
        question_enc = self.tokenizer(
            question_text,
            max_length=self.max_question_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        solution_enc = self.tokenizer(
            solution_text,
            max_length=self.max_answer_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'question_ids': question_enc['input_ids'].squeeze(0),
            'question_mask': question_enc['attention_mask'].squeeze(0),
            'solution_ids': solution_enc['input_ids'].squeeze(0),
            'solution_mask': solution_enc['attention_mask'].squeeze(0),
        }
