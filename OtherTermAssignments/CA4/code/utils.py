import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm import tqdm

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_FIG_DIR = REPO_ROOT / "report" / "En_report" / "figures"
REPORT_FIG_DIR.mkdir(parents=True, exist_ok=True)


def set_report_fig_dir(path):
    """Set report figure output directory at runtime."""
    global REPORT_FIG_DIR
    REPORT_FIG_DIR = Path(path).resolve()
    REPORT_FIG_DIR.mkdir(parents=True, exist_ok=True)
    return REPORT_FIG_DIR

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
