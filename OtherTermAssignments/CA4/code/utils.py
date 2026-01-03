import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import math
from torchvision import transforms, datasets
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import os

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# Report figures directory
REPORT_FIG_DIR = Path("../report/En_report/figures")
REPORT_FIG_DIR.mkdir(parents=True, exist_ok=True)

def save_fig(filename: str, dpi: int = 200):
    """
    Save the current matplotlib figure into the report figures folder.

    Args:
        filename: Name of the file to save
        dpi: Resolution for the saved figure
    """
    out_path = REPORT_FIG_DIR / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    print(f"Saved figure: {out_path}")