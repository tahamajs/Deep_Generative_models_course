"""
Discrete Diffusion Language Model for Math Reasoning
Based on MDLM (Masked Diffusion Language Models) architecture

Key innovations:
1. Discrete masked diffusion (more stable than continuous)
2. Bidirectional transformer (full attention, not causal)
3. Self-conditioning for better quality
4. Adaptive layer normalization for time conditioning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict
from dataclasses import dataclass


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal embeddings for diffusion timesteps"""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None].float() * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class AdaptiveLayerNorm(nn.Module):
    """
    Adaptive Layer Normalization (adaLN)
    Modulates layer norm with scale and shift from time embeddings
    Used in DiT (Diffusion Transformer) architecture
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.gamma_proj = nn.Linear(hidden_dim, hidden_dim)
        self.beta_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq, hidden)
        # time_emb: (batch, hidden)
        normalized = self.norm(x)
        gamma = self.gamma_proj(time_emb).unsqueeze(1)  # (batch, 1, hidden)
        beta = self.beta_proj(time_emb).unsqueeze(1)
        return gamma * normalized + beta


class TransformerBlock(nn.Module):
    """
    Transformer block with adaptive layer norm for time conditioning
    Uses full bidirectional attention (not causal)
    """
    def __init__(
        self, 
        hidden_dim: int, 
        num_heads: int, 
        intermediate_dim: int,
        dropout: float = 0.1,
        use_flash_attention: bool = True
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # Attention
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, intermediate_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_dim, hidden_dim),
            nn.Dropout(dropout)
        )
        
        # Adaptive layer norms
        self.adaln1 = AdaptiveLayerNorm(hidden_dim)
        self.adaln2 = AdaptiveLayerNorm(hidden_dim)
        
        self.attn_dropout = nn.Dropout(dropout)
        self.use_flash = use_flash_attention and hasattr(F, 'scaled_dot_product_attention')
        
    def forward(
        self, 
        x: torch.Tensor, 
        time_emb: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        condition: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Pre-norm with time conditioning
        normed = self.adaln1(x, time_emb)
        
        # Self-attention (bidirectional)
        q = self.q_proj(normed).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(normed).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(normed).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Optional cross-attention with condition
        if condition is not None:
            cond_len = condition.shape[1]
            k_cond = self.k_proj(condition).view(batch_size, cond_len, self.num_heads, self.head_dim).transpose(1, 2)
            v_cond = self.v_proj(condition).view(batch_size, cond_len, self.num_heads, self.head_dim).transpose(1, 2)
            k = torch.cat([k, k_cond], dim=2)
            v = torch.cat([v, v_cond], dim=2)
        
        if self.use_flash:
            # Use Flash Attention if available
            attn_out = F.scaled_dot_product_attention(
                q, k, v, 
                attn_mask=None,  # No causal mask - full bidirectional
                dropout_p=self.attn_dropout.p if self.training else 0.0
            )
        else:
            # Standard attention
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if attention_mask is not None:
                scores = scores.masked_fill(attention_mask == 0, float('-inf'))
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.attn_dropout(attn_weights)
            attn_out = torch.matmul(attn_weights, v)
        
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        attn_out = self.out_proj(attn_out)
        
        # Residual connection
        x = x + attn_out
        
        # FFN with adaptive layer norm
        normed = self.adaln2(x, time_emb)
        x = x + self.ffn(normed)
        
        return x


class MathDiffusionTransformer(nn.Module):
    """
    Main diffusion transformer for math reasoning
    
    Architecture:
    - Token embeddings (learned, not frozen BERT)
    - Sinusoidal position embeddings
    - Time embedding MLP
    - Stack of transformer blocks with adaLN
    - Output projection to vocabulary
    """
    def __init__(
        self,
        vocab_size: int = 50257,
        hidden_dim: int = 512,
        num_layers: int = 8,
        num_heads: int = 8,
        intermediate_dim: int = 2048,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        use_self_conditioning: bool = True
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        self.use_self_conditioning = use_self_conditioning
        
        # Token embeddings (learned from scratch)
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_dim)
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        # Self-conditioning: project previous prediction
        if use_self_conditioning:
            self.self_cond_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Condition encoder (for question)
        self.condition_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                intermediate_dim=intermediate_dim,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        # Final layer norm and output projection
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, vocab_size, bias=False)
        
        # Tie input and output embeddings
        self.output_proj.weight = self.token_embedding.weight
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights with small values for stable training"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                
    def forward(
        self,
        input_ids: torch.Tensor,
        timesteps: torch.Tensor,
        condition_ids: Optional[torch.Tensor] = None,
        condition_mask: Optional[torch.Tensor] = None,
        self_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for denoising
        
        Args:
            input_ids: Noisy/masked token ids (batch, seq_len)
            timesteps: Diffusion timesteps (batch,)
            condition_ids: Question token ids (batch, cond_len)
            condition_mask: Attention mask for condition (batch, cond_len)
            self_cond: Previous prediction for self-conditioning (batch, seq_len, hidden)
            attention_mask: Attention mask for input (batch, seq_len)
            
        Returns:
            logits: Predicted token logits (batch, seq_len, vocab_size)
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Token + position embeddings
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        
        # Time embedding
        time_emb = self.time_mlp(timesteps)
        
        # Self-conditioning
        if self.use_self_conditioning and self_cond is not None:
            x = x + self.self_cond_proj(self_cond)
        
        # Encode condition (question)
        condition = None
        if condition_ids is not None:
            cond_positions = torch.arange(condition_ids.shape[1], device=device).unsqueeze(0)
            cond_emb = self.token_embedding(condition_ids) + self.position_embedding(cond_positions)
            condition = self.condition_encoder(cond_emb)
        
        # Transformer layers
        for layer in self.layers:
            x = layer(x, time_emb, attention_mask, condition)
        
        # Output projection
        x = self.final_norm(x)
        logits = self.output_proj(x)
        
        return logits


class DiscreteDiffusion(nn.Module):
    """
    Discrete Diffusion for Math Reasoning
    
    Uses masked diffusion (MDLM-style):
    - Forward process: gradually mask tokens
    - Reverse process: predict and unmask tokens
    
    This is more stable than continuous diffusion for text
    """
    def __init__(
        self,
        model: MathDiffusionTransformer,
        vocab_size: int = 50257,
        timesteps: int = 1000,
        mask_token_id: int = 50256,  # Use GPT-2 <|endoftext|> or special mask
        noise_schedule: str = "cosine"
    ):
        super().__init__()
        self.model = model
        self.vocab_size = vocab_size
        self.timesteps = timesteps
        self.mask_token_id = mask_token_id
        
        # Compute noise schedule (probability of masking at each timestep)
        self.register_buffer('mask_schedule', self._compute_schedule(noise_schedule))
        
    def _compute_schedule(self, schedule_type: str) -> torch.Tensor:
        """Compute masking probability schedule"""
        t = torch.linspace(0, 1, self.timesteps + 1)
        
        if schedule_type == "linear":
            mask_prob = t
        elif schedule_type == "cosine":
            # Cosine schedule (slower at extremes)
            mask_prob = 1 - torch.cos(t * math.pi / 2)
        elif schedule_type == "sqrt":
            mask_prob = torch.sqrt(t)
        else:
            raise ValueError(f"Unknown schedule: {schedule_type}")
        
        return mask_prob
    
    def q_sample(
        self, 
        x_0: torch.Tensor, 
        t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward diffusion: mask tokens according to schedule
        
        Args:
            x_0: Clean token ids (batch, seq_len)
            t: Timesteps (batch,)
            
        Returns:
            x_t: Masked token ids
            mask: Boolean mask of which tokens are masked
        """
        batch_size, seq_len = x_0.shape
        device = x_0.device
        
        # Get mask probability for each sample
        mask_prob = self.mask_schedule[t].unsqueeze(1)  # (batch, 1)
        
        # Sample which tokens to mask
        rand = torch.rand(batch_size, seq_len, device=device)
        mask = rand < mask_prob  # True = masked
        
        # Apply masking
        x_t = x_0.clone()
        x_t[mask] = self.mask_token_id
        
        return x_t, mask
    
    def compute_loss(
        self,
        x_0: torch.Tensor,
        condition_ids: Optional[torch.Tensor] = None,
        condition_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        self_cond_prob: float = 0.5
    ) -> Dict[str, torch.Tensor]:
        """
        Compute training loss
        
        Uses cross-entropy on masked positions (MDLM-style)
        """
        batch_size, seq_len = x_0.shape
        device = x_0.device
        
        # Sample random timesteps
        t = torch.randint(1, self.timesteps + 1, (batch_size,), device=device)
        
        # Forward diffusion (mask tokens)
        x_t, mask = self.q_sample(x_0, t)
        
        # Self-conditioning: sometimes predict without previous output
        self_cond = None
        if self.model.use_self_conditioning and torch.rand(1).item() < self_cond_prob:
            with torch.no_grad():
                # Get previous prediction for self-conditioning
                prev_logits = self.model(
                    x_t, t, condition_ids, condition_mask, None, attention_mask
                )
                # Convert to embeddings
                prev_probs = F.softmax(prev_logits, dim=-1)
                self_cond = torch.matmul(prev_probs, self.model.token_embedding.weight)
        
        # Predict clean tokens
        logits = self.model(x_t, t, condition_ids, condition_mask, self_cond, attention_mask)
        
        # Compute loss only on masked positions
        # Flatten for cross-entropy
        logits_flat = logits.view(-1, self.vocab_size)
        targets_flat = x_0.view(-1)
        mask_flat = mask.view(-1)
        
        # Cross-entropy on masked positions
        loss_all = F.cross_entropy(logits_flat, targets_flat, reduction='none')
        loss_masked = (loss_all * mask_flat.float()).sum() / (mask_flat.sum() + 1e-8)
        
        # Also compute accuracy on masked positions
        with torch.no_grad():
            preds = logits.argmax(dim=-1).view(-1)
            acc = ((preds == targets_flat) * mask_flat.float()).sum() / (mask_flat.sum() + 1e-8)
        
        return {
            'loss': loss_masked,
            'accuracy': acc,
            'num_masked': mask_flat.sum()
        }
    
    @torch.no_grad()
    def sample(
        self,
        condition_ids: torch.Tensor,
        condition_mask: Optional[torch.Tensor] = None,
        seq_len: int = 384,
        num_steps: Optional[int] = None,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = 0.95,
        verbose: bool = False
    ) -> Tuple[torch.Tensor, list]:
        """
        Sample from the model using reverse diffusion
        
        Args:
            condition_ids: Question token ids (batch, cond_len)
            condition_mask: Attention mask for condition
            seq_len: Length of sequence to generate
            num_steps: Number of sampling steps (default: self.timesteps)
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Nucleus sampling threshold
            verbose: Print progress
            
        Returns:
            samples: Generated token ids (batch, seq_len)
            history: List of intermediate states
        """
        self.model.eval()
        device = condition_ids.device
        batch_size = condition_ids.shape[0]
        num_steps = num_steps or self.timesteps
        
        # Start from fully masked sequence
        x = torch.full((batch_size, seq_len), self.mask_token_id, device=device)
        
        # Track history
        history = []
        self_cond = None
        
        # Reverse diffusion
        step_size = self.timesteps // num_steps
        timesteps = list(range(self.timesteps, 0, -step_size))
        
        if verbose:
            from tqdm import tqdm
            timesteps = tqdm(timesteps, desc="Sampling")
        
        for t in timesteps:
            t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)
            
            # Predict clean tokens
            logits = self.model(x, t_tensor, condition_ids, condition_mask, self_cond)
            
            # Apply temperature
            logits = logits / temperature
            
            # Top-k filtering
            if top_k is not None:
                indices_to_remove = logits < torch.topk(logits, top_k, dim=-1)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')
            
            # Top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')
            
            # Sample from distribution
            probs = F.softmax(logits, dim=-1)
            samples = torch.multinomial(probs.view(-1, self.vocab_size), 1).view(batch_size, seq_len)
            
            # Only update masked positions
            is_masked = (x == self.mask_token_id)
            
            # Compute which positions to unmask at this step
            # More positions get unmasked as t decreases
            current_mask_prob = self.mask_schedule[t]
            next_mask_prob = self.mask_schedule[max(0, t - step_size)]
            
            # Unmask some positions
            unmask_prob = (current_mask_prob - next_mask_prob) / (current_mask_prob + 1e-8)
            unmask = torch.rand_like(x.float()) < unmask_prob
            update_mask = is_masked & unmask
            
            x = torch.where(update_mask, samples, x)
            
            # Update self-conditioning
            if self.model.use_self_conditioning:
                self_cond = torch.matmul(probs, self.model.token_embedding.weight)
            
            # Save to history periodically
            if t % 100 == 0:
                history.append(x.clone())
        
        # Final prediction for any remaining masked tokens
        t_tensor = torch.ones((batch_size,), device=device, dtype=torch.long)
        logits = self.model(x, t_tensor, condition_ids, condition_mask, self_cond)
        final_samples = logits.argmax(dim=-1)
        
        # Replace any remaining masks
        is_masked = (x == self.mask_token_id)
        x = torch.where(is_masked, final_samples, x)
        
        return x, history


def create_model(config) -> DiscreteDiffusion:
    """Create model from config"""
    transformer = MathDiffusionTransformer(
        vocab_size=config.model.vocab_size,
        hidden_dim=config.model.hidden_dim,
        num_layers=config.model.num_layers,
        num_heads=config.model.num_heads,
        intermediate_dim=config.model.intermediate_dim,
        max_seq_len=config.model.max_seq_len,
        dropout=config.model.dropout,
        use_self_conditioning=config.model.use_self_conditioning
    )
    
    diffusion = DiscreteDiffusion(
        model=transformer,
        vocab_size=config.model.vocab_size,
        timesteps=config.model.timesteps,
        noise_schedule=config.model.noise_schedule
    )
    
    return diffusion
