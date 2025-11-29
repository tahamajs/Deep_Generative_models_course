#!/usr/bin/env python3
"""
Fix notebooks to comply with project rules:
1. Remove duplicate imports
2. Add random seeds for reproducibility
3. Organize imports properly
4. Add proper configuration structure
"""

import json
import sys
from pathlib import Path

def fix_ca1_notebook():
    """Fix CA1 notebook imports and configuration"""
    notebook_path = Path("next_year/CA1/code/code.ipynb")
    
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    # Find the imports cell (second cell, index 1)
    imports_cell = nb['cells'][1]
    
    # Create new proper imports with configuration
    new_source = """# Setup and Configuration
import random
import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import networkx as nx

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.decomposition import PCA
from sklearn.metrics import mutual_info_score
from tqdm import tqdm

# Set random seeds for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Configuration parameters
# (Add hyperparameters here as needed for training)
"""
    
    # Update the cell source (split into list of lines)
    imports_cell['source'] = new_source.split('\n')
    # Add newline characters except for last line
    imports_cell['source'] = [line + '\n' for line in imports_cell['source'][:-1]] + [imports_cell['source'][-1]]
    
    # Save the notebook
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    
    print(f"✓ Fixed {notebook_path}")

def fix_ca2_notebook():
    """Fix CA2 notebook imports and configuration"""
    notebook_path = Path("next_year/CA2/code/code.ipynb")
    
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    # Find the imports cell (second cell, index 1)
    imports_cell = nb['cells'][1]
    
    # Create new proper imports with configuration
    new_source = """# Setup and Configuration
import random
import os
import shutil
import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm

# Set random seeds for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Configuration parameters
# (Add hyperparameters here as needed for training)
"""
    
    # Update the cell source
    imports_cell['source'] = new_source.split('\n')
    # Add newline characters except for last line
    imports_cell['source'] = [line + '\n' for line in imports_cell['source'][:-1]] + [imports_cell['source'][-1]]
    
    # Save the notebook
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    
    print(f"✓ Fixed {notebook_path}")

if __name__ == "__main__":
    # Change to project root
    project_root = Path(__file__).parent
    import os
    os.chdir(project_root)
    
    print("Fixing notebooks to comply with project rules...")
    fix_ca1_notebook()
    fix_ca2_notebook()
    print("\nAll notebooks fixed!")
