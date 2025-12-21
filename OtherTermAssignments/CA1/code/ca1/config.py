import os
import random
import json
import subprocess
from datetime import datetime
from typing import Optional

import numpy as np
import torch


CONFIG = {
    "seed": 42,
    "data_path": "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz",
    "train_split": 0.9,
    "batch_size": 128,
    "epochs": 30,
    "learning_rate": 1e-3,
    "latent_dim": 256,
    "betas": [1.0, 2.0, 5.0],
    "data_subset": 30000,
    "num_workers": 2,
    "pin_memory": True,
    "run_complete_analysis": False,
    "smoke_test": True,
    "smoke_subset": 2048,
    "smoke_epochs": 1,
}


def set_seed(seed: int) -> torch.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return torch.Generator().manual_seed(seed)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_run_info(config: dict, path: str = "run_info.json", extra: Optional[dict] = None) -> None:
    run_info = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "commit": None,
        "config": config,
        "device": str(device),
    }
    try:
        run_info["commit"] = (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]) .decode().strip()
        )
    except Exception:
        run_info["commit"] = "unknown"
    if extra:
        run_info["extra"] = extra
    with open(path, "w", encoding="utf-8") as f:
        json.dump(run_info, f, indent=2)
