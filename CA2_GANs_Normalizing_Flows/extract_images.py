#!/usr/bin/env python3
"""
Extract images from Jupyter notebook outputs
"""
import json
import base64
import os
from pathlib import Path

# Paths
notebook_path = "code/Q2_final_res.ipynb"
images_dir = "images"

# Create images directory
os.makedirs(images_dir, exist_ok=True)

# Read notebook
with open(notebook_path, "r") as f:
    notebook = json.load(f)

# Extract images from outputs
image_count = 0
for cell_idx, cell in enumerate(notebook["cells"]):
    if "outputs" in cell:
        for output_idx, output in enumerate(cell["outputs"]):
            if "data" in output and "image/png" in output["data"]:
                image_data = output["data"]["image/png"]

                # Decode base64
                image_bytes = base64.b64decode(image_data)

                # Generate filename
                image_count += 1
                filename = f"generated_sample_epoch_{image_count}.png"
                filepath = os.path.join(images_dir, filename)

                # Save image
                with open(filepath, "wb") as img_file:
                    img_file.write(image_bytes)

                print(f"Extracted image {image_count}: {filename}")

print(f"\nTotal images extracted: {image_count}")
print(f"Images saved to: {images_dir}/")

