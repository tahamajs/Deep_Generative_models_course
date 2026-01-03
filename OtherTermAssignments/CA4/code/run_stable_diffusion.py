#!/usr/bin/env python3
"""
Stable Diffusion & DreamBooth Runner Script

This script demonstrates the DreamBooth fine-tuning implementation.
"""

from stable_diffusion import *

if __name__ == "__main__":
    # Complete DreamBooth training example
    # Uncomment and run when you have your instance images ready

    """
    # Step 1: Prepare data folders
    import os
    os.makedirs(DREAMBOOTH_CONFIG['instance_data_root'], exist_ok=True)
    os.makedirs(DREAMBOOTH_CONFIG['class_data_root'], exist_ok=True)

    # Step 2: Generate class images (if not already done)
    if DIFFUSERS_AVAILABLE:
        generate_class_images(
            class_prompt=DREAMBOOTH_CONFIG['class_prompt'],
            class_data_root=DREAMBOOTH_CONFIG['class_data_root'],
            num_class_images=DREAMBOOTH_CONFIG['num_class_images']
        )

    # Step 3: Create dataset
    if DIFFUSERS_AVAILABLE:
        tokenizer = CLIPTokenizer.from_pretrained(
            "runwayml/stable-diffusion-v1-5", subfolder="tokenizer"
        )

        train_dataset = DreamBoothDataset(
            instance_data_root=DREAMBOOTH_CONFIG['instance_data_root'],
            instance_prompt=DREAMBOOTH_CONFIG['instance_prompt'],
            tokenizer=tokenizer,
            class_data_root=DREAMBOOTH_CONFIG['class_data_root'],
            class_prompt=DREAMBOOTH_CONFIG['class_prompt'],
            size=512,
        )

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=1,
            shuffle=True,
            collate_fn=lambda x: collate_fn(x, with_prior_preservation=True)
        )

    # Step 4: Initialize trainer and train
    if DIFFUSERS_AVAILABLE:
        dreambooth_trainer = DreamBoothTrainer(
            lora_rank=DREAMBOOTH_CONFIG['lora_rank'],
            learning_rate=DREAMBOOTH_CONFIG['learning_rate'],
            prior_preservation=True,
        )

        dreambooth_trainer.train(train_dataloader, num_epochs=DREAMBOOTH_CONFIG['num_epochs'])
        dreambooth_trainer.save_lora_weights('./dreambooth_lora')
        dreambooth_trainer.plot_loss()

    # Step 5: Inference
    if DIFFUSERS_AVAILABLE:
        inference = DreamBoothInference(lora_weights_path='./dreambooth_lora')

        test_prompts = [
            "a photo of sks dog on the moon",
            "a photo of sks dog in a bucket",
            "a photo of sks dog wearing a hat",
            "a painting of sks dog in the style of Van Gogh"
        ]

        for prompt in test_prompts:
            images = inference.generate(prompt, num_images=4)
            # Note: visualize_samples is from ddpm module
            from ddpm import visualize_samples
            from torchvision import transforms
            visualize_samples(torch.stack([
                transforms.ToTensor()(img) for img in images
            ]), title=prompt)
    """

    print("DreamBooth training example commented out - uncomment when ready to train")