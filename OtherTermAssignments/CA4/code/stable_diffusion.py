"""
Stable Diffusion DreamBooth Fine-tuning Implementation

This module implements DreamBooth fine-tuning for Stable Diffusion models.
DreamBooth enables personalized image generation by fine-tuning a pre-trained
Stable Diffusion model on a few images of a specific subject.

Key components:
- DreamBoothDataset: Custom dataset for subject images with text prompts
- DreamBoothTrainer: Fine-tuning loop with LoRA and prior preservation
- DreamBoothInference: Text-to-image generation with fine-tuned model

Analysis:
- DreamBooth uses a unique identifier (e.g., "sks dog") to bind subject to concept
- Prior preservation loss prevents catastrophic forgetting of general concepts
- LoRA (Low-Rank Adaptation) enables efficient fine-tuning with minimal parameters
- Mixed precision training reduces memory usage and improves speed

Performance Notes:
- Fine-tuning time: ~10-30 minutes on GPU depending on dataset size
- Memory usage: ~4-6GB GPU memory with mixed precision
- Requires: 3-5 images of the subject for good results
- Best for: Subject-specific image generation, style transfer, concept customization

Limitations:
- Can suffer from overfitting with too few images
- May require prompt engineering for best results
- Subject binding depends on choosing a unique, unused identifier
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import os

from utils import device, save_fig, REPORT_FIG_DIR

try:
    from diffusers import (
        StableDiffusionPipeline,
        DDPMScheduler as DiffusersDDPMScheduler,
        UNet2DConditionModel,
        AutoencoderKL
    )
    from transformers import CLIPTextModel, CLIPTokenizer
    from peft import LoraConfig, get_peft_model
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False


class DreamBoothDataset(Dataset):
    """
    DreamBooth Dataset for fine-tuning Stable Diffusion.

    This dataset handles the loading and preprocessing of images for DreamBooth training.
    It supports both instance images (specific subject) and class images (for prior preservation).

    Args:
        instance_data_root: Path to folder containing instance images of the subject
        instance_prompt: Text prompt describing the instance (e.g., "a photo of sks dog")
        tokenizer: CLIP tokenizer for encoding text prompts
        class_data_root: Optional path to class images for prior preservation loss
        class_prompt: Generic prompt for class images (e.g., "a photo of a dog")
        size: Target image size (default: 512 for Stable Diffusion)
        center_crop: Whether to use center cropping (default: False, uses random crop)

    Analysis:
    - Instance images teach the model the specific subject appearance
    - Class images prevent catastrophic forgetting of general concepts
    - Prior preservation ratio (typically 1:1) balances subject learning vs. concept retention
    - Random cropping during training improves generalization
    - Normalization to [-1, 1] matches Stable Diffusion's expected input range
    """
    def __init__(
        self,
        instance_data_root,
        instance_prompt,
        tokenizer,
        class_data_root=None,
        class_prompt=None,
        size=512,
        center_crop=False,
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.instance_prompt = instance_prompt
        self.class_prompt = class_prompt

        # Load instance images
        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError(f"Instance data root doesn't exist: {instance_data_root}")

        # Get all image files
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']
        self.instance_images_path = []
        for ext in image_extensions:
            self.instance_images_path.extend(list(self.instance_data_root.glob(ext)))

        self.num_instance_images = len(self.instance_images_path)
        if self.num_instance_images == 0:
            raise ValueError(f"No images found in {instance_data_root}")

        print(f"Found {self.num_instance_images} instance images")

        # Load class images (for Prior Preservation)
        self.class_data_root = None
        self.class_images_path = []
        self.num_class_images = 0

        if class_data_root is not None:
            self.class_data_root = Path(class_data_root)
            self.class_data_root.mkdir(parents=True, exist_ok=True)

            for ext in image_extensions:
                self.class_images_path.extend(list(self.class_data_root.glob(ext)))

            self.num_class_images = len(self.class_images_path)
            print(f"Found {self.num_class_images} class images for prior preservation")

        # Dataset length
        self._length = max(self.num_class_images, self.num_instance_images)
        if self.num_class_images > 0:
            self._length = self.num_instance_images  # Match instance images

        # Image transformations
        self.image_transforms = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),  # Normalize to [-1, 1]
        ])

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        """
        Returns a dictionary containing:
        - instance_images: Preprocessed instance image tensor
        - instance_prompt_ids: Tokenized instance prompt
        - class_images (optional): Preprocessed class image tensor
        - class_prompt_ids (optional): Tokenized class prompt
        """
        example = {}

        # ==========================================
        # Step 1: Load and preprocess instance image
        # ==========================================
        instance_image_path = self.instance_images_path[index % self.num_instance_images]
        instance_image = Image.open(instance_image_path)

        # Convert to RGB if necessary
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")

        # Apply transformations
        example["instance_images"] = self.image_transforms(instance_image)

        # ==========================================
        # Step 2: Tokenize instance prompt
        # ==========================================
        example["instance_prompt_ids"] = self.tokenizer(
            self.instance_prompt,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt"
        ).input_ids.squeeze(0)

        # ==========================================
        # Step 3: Load class image (Prior Preservation)
        # ==========================================
        if self.class_data_root and self.num_class_images > 0:
            class_image_path = self.class_images_path[index % self.num_class_images]
            class_image = Image.open(class_image_path)

            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")

            example["class_images"] = self.image_transforms(class_image)

            # Tokenize class prompt
            example["class_prompt_ids"] = self.tokenizer(
                self.class_prompt,
                padding="max_length",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt"
            ).input_ids.squeeze(0)

        return example


def collate_fn(examples, with_prior_preservation=True):
    """
    Custom collate function for DreamBooth dataloader.
    Combines instance and class examples for prior preservation loss.
    """
    input_ids = [example["instance_prompt_ids"] for example in examples]
    pixel_values = [example["instance_images"] for example in examples]

    # Add class images for prior preservation
    if with_prior_preservation and "class_images" in examples[0]:
        input_ids += [example["class_prompt_ids"] for example in examples]
        pixel_values += [example["class_images"] for example in examples]

    pixel_values = torch.stack(pixel_values)
    input_ids = torch.stack(input_ids)

    return {
        "input_ids": input_ids,
        "pixel_values": pixel_values,
    }


def generate_class_images(
    class_prompt,
    class_data_root,
    num_class_images=100,
    model_id="runwayml/stable-diffusion-v1-5",
    batch_size=4,
    device="cuda"
):
    """
    Step 3 (Bonus): Generate class images for prior preservation.

    Uses the pre-trained Stable Diffusion model to generate generic images
    (e.g., regular dogs) to prevent the model from forgetting.

    Args:
        class_prompt: Generic prompt (e.g., "a photo of a dog")
        class_data_root: Path to save generated images
        num_class_images: Number of images to generate
        model_id: HuggingFace model ID
        batch_size: Batch size for generation
        device: Device to use
    """
    if not DIFFUSERS_AVAILABLE:
        print("⚠️ Diffusers not available. Skipping class image generation.")
        return

    class_data_root = Path(class_data_root)
    class_data_root.mkdir(parents=True, exist_ok=True)

    # Check how many images already exist
    existing_images = list(class_data_root.glob("*.png"))
    if len(existing_images) >= num_class_images:
        print(f"Already have {len(existing_images)} class images. Skipping generation.")
        return

    print(f"Generating {num_class_images - len(existing_images)} class images...")

    # Load the pipeline
    pipeline = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None,
    ).to(device)

    pipeline.set_progress_bar_config(disable=True)

    # Generate images
    num_to_generate = num_class_images - len(existing_images)
    start_idx = len(existing_images)

    for i in tqdm(range(0, num_to_generate, batch_size), desc="Generating class images"):
        batch_count = min(batch_size, num_to_generate - i)

        images = pipeline(
            [class_prompt] * batch_count,
            num_inference_steps=50,
            guidance_scale=7.5,
        ).images

        for j, image in enumerate(images):
            image.save(class_data_root / f"class_image_{start_idx + i + j:04d}.png")

    print(f"Done: Generated {num_to_generate} class images in {class_data_root}")

    # Clear memory
    del pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class DreamBoothTrainer:
    """
    DreamBooth Trainer with LoRA support.

    Implements the full training pipeline:
    1. Encode images to latent space using VAE
    2. Add noise (forward diffusion)
    3. Predict noise with U-Net conditioned on text
    4. Compute loss (with optional prior preservation)
    5. Update only LoRA weights
    """
    def __init__(
        self,
        model_id="runwayml/stable-diffusion-v1-5",
        lora_rank=4,
        learning_rate=1e-4,
        prior_preservation=True,
        prior_loss_weight=1.0,
        device="cuda",
    ):
        self.device = device
        self.prior_preservation = prior_preservation
        self.prior_loss_weight = prior_loss_weight
        self.loss_history = []

        if not DIFFUSERS_AVAILABLE:
            print("⚠️ Diffusers not available. Using mock implementation.")
            self._setup_mock()
            return

        print(f"Loading Stable Diffusion model: {model_id}")

        # Load tokenizer
        self.tokenizer = CLIPTokenizer.from_pretrained(
            model_id, subfolder="tokenizer"
        )

        # Load text encoder
        self.text_encoder = CLIPTextModel.from_pretrained(
            model_id, subfolder="text_encoder"
        ).to(device)
        self.text_encoder.requires_grad_(False)  # Freeze text encoder

        # Load VAE
        self.vae = AutoencoderKL.from_pretrained(
            model_id, subfolder="vae"
        ).to(device)
        self.vae.requires_grad_(False)  # Freeze VAE

        # Load U-Net
        self.unet = UNet2DConditionModel.from_pretrained(
            model_id, subfolder="unet"
        ).to(device)

        # Apply LoRA to U-Net
        print(f"Applying LoRA with rank {lora_rank}...")
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_rank,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )
        self.unet = get_peft_model(self.unet, lora_config)
        self.unet.print_trainable_parameters()

        # Load scheduler
        self.noise_scheduler = DiffusersDDPMScheduler.from_pretrained(
            model_id, subfolder="scheduler"
        )

        # Optimizer (only LoRA parameters)
        self.optimizer = torch.optim.AdamW(
            self.unet.parameters(),
            lr=learning_rate,
            weight_decay=1e-2,
        )

        # Mixed precision scaler
        self.scaler = torch.cuda.amp.GradScaler() if device == "cuda" else None

        # VAE scaling factor
        self.vae_scale_factor = 0.18215

        print("Done: DreamBooth trainer initialized!")

    def _setup_mock(self):
        """Setup mock components when diffusers is not available."""
        self.tokenizer = None
        self.text_encoder = None
        self.vae = None
        self.unet = None
        self.noise_scheduler = None
        self.optimizer = None
        self.scaler = None

    def encode_images(self, images):
        """Encode images to latent space using VAE."""
        with torch.no_grad():
            latents = self.vae.encode(images).latent_dist.sample()
            latents = latents * self.vae_scale_factor
        return latents

    def encode_prompt(self, input_ids):
        """Encode text prompts using CLIP text encoder."""
        with torch.no_grad():
            encoder_hidden_states = self.text_encoder(input_ids)[0]
        return encoder_hidden_states

    def train_step(self, batch):
        """
        Single training step for DreamBooth.

        Args:
            batch: Dictionary with 'pixel_values' and 'input_ids'

        Returns:
            loss: Training loss value
        """
        if not DIFFUSERS_AVAILABLE:
            return 0.0

        self.unet.train()

        pixel_values = batch["pixel_values"].to(self.device)
        input_ids = batch["input_ids"].to(self.device)

        batch_size = pixel_values.shape[0]

        # Step 1: Encode images to latent space
        latents = self.encode_images(pixel_values)

        # Step 2: Sample noise
        noise = torch.randn_like(latents)

        # Step 3: Sample random timesteps
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (batch_size,), device=self.device
        ).long()

        # Step 4: Add noise (Forward process)
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        # Step 5: Encode text prompts
        encoder_hidden_states = self.encode_prompt(input_ids)

        # Step 6: Predict noise with U-Net
        with torch.cuda.amp.autocast(enabled=self.scaler is not None):
            noise_pred = self.unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states
            ).sample

            # Step 7: Calculate loss
            if self.prior_preservation:
                # Split batch: first half is instance, second half is class
                noise_pred_instance, noise_pred_class = noise_pred.chunk(2)
                noise_instance, noise_class = noise.chunk(2)

                # Instance loss
                instance_loss = F.mse_loss(
                    noise_pred_instance, noise_instance, reduction="mean"
                )

                # Prior preservation loss
                prior_loss = F.mse_loss(
                    noise_pred_class, noise_class, reduction="mean"
                )

                loss = instance_loss + self.prior_loss_weight * prior_loss
            else:
                loss = F.mse_loss(noise_pred, noise, reduction="mean")

        # Step 8: Backpropagation
        self.optimizer.zero_grad()

        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()

        return loss.item()

    def train(self, dataloader, num_epochs=100):
        """
        Full training loop.
        """
        if not DIFFUSERS_AVAILABLE:
            print("⚠️ Cannot train without diffusers library.")
            return

        print(f"Starting DreamBooth training for {num_epochs} epochs...")

        for epoch in range(1, num_epochs + 1):
            epoch_loss = 0
            pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

            for batch in pbar:
                loss = self.train_step(batch)
                epoch_loss += loss
                pbar.set_postfix({'loss': f'{loss:.4f}'})

            avg_loss = epoch_loss / len(dataloader)
            self.loss_history.append(avg_loss)
            print(f"Epoch {epoch}/{num_epochs} - Loss: {avg_loss:.4f}")

        print("Done: DreamBooth training complete!")

    def save_lora_weights(self, save_path):
        """Save LoRA weights."""
        if not DIFFUSERS_AVAILABLE:
            return

        self.unet.save_pretrained(save_path)
        print(f"Done: LoRA weights saved to {save_path}")

    def plot_loss(self):
        """Plot training loss."""
        plt.figure(figsize=(10, 4))
        plt.plot(self.loss_history)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('DreamBooth Training Loss')
        plt.grid(True)
        plt.show()


class DreamBoothInference:
    """
    Inference pipeline for DreamBooth-trained models.
    """
    def __init__(
        self,
        model_id="runwayml/stable-diffusion-v1-5",
        lora_weights_path=None,
        device="cuda"
    ):
        self.device = device

        if not DIFFUSERS_AVAILABLE:
            print("⚠️ Diffusers not available for inference.")
            self.pipeline = None
            return

        print("Loading inference pipeline...")

        # Load pipeline
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            safety_checker=None,
        ).to(device)

        # Load LoRA weights if provided
        if lora_weights_path:
            print(f"Loading LoRA weights from {lora_weights_path}")
            self.pipeline.unet.load_adapter(lora_weights_path)

        print("Done: Inference pipeline ready!")

    @torch.no_grad()
    def generate(
        self,
        prompt,
        negative_prompt="low quality, blurry, distorted",
        num_images=4,
        num_inference_steps=50,
        guidance_scale=7.5,
        seed=None
    ):
        """
        Generate images with the fine-tuned model.

        Args:
            prompt: Text prompt
            negative_prompt: Negative prompt for guidance
            num_images: Number of images to generate
            num_inference_steps: Denoising steps
            guidance_scale: CFG scale (higher = more prompt adherence)
            seed: Random seed for reproducibility

        Returns:
            List of PIL images
        """
        if self.pipeline is None:
            print("⚠️ Pipeline not available.")
            return []

        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None

        images = self.pipeline(
            prompt=[prompt] * num_images,
            negative_prompt=[negative_prompt] * num_images,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        ).images

        return images

    def visualize_guidance_scales(self, prompt, scales=[3.0, 5.0, 7.5, 10.0, 15.0]):
        """
        Compare different guidance scales for the same prompt.
        """
        if self.pipeline is None:
            return

        fig, axes = plt.subplots(1, len(scales), figsize=(4*len(scales), 4))

        for idx, scale in enumerate(scales):
            images = self.generate(prompt, num_images=1, guidance_scale=scale, seed=42)
            axes[idx].imshow(images[0])
            axes[idx].set_title(f'CFG Scale: {scale}')
            axes[idx].axis('off')

        plt.suptitle(f'Prompt: "{prompt}"')
        plt.tight_layout()
        plt.show()


# Example DreamBooth usage
DREAMBOOTH_CONFIG = {
    'instance_data_root': './instance_data',  # Put 3-5 images of your subject here
    'class_data_root': './class_data',         # Will be populated with generated class images
    'instance_prompt': 'a photo of sks dog',   # Replace 'sks' with your unique identifier
    'class_prompt': 'a photo of a dog',
    'num_class_images': 100,
    'learning_rate': 1e-4,
    'num_epochs': 100,
    'lora_rank': 4,
}