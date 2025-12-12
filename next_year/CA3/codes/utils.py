"""
Shared utilities for CA3 experiments: seeding, I/O, and visualization.
"""

from pathlib import Path
import random
import json
import subprocess
from datetime import datetime
from typing import Dict, Any, Optional

import numpy as np
import torch
from torchvision.utils import make_grid, save_image


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_grid(
    images: torch.Tensor, path: Path, nrow: int = 8, normalize: bool = True
) -> None:
    ensure_dir(path.parent)
    grid = make_grid(images, nrow=nrow, normalize=normalize, value_range=(0, 1))
    save_image(grid, path)


def _repo_root() -> Path:
    """Return repository root (four levels up from this file)."""
    return Path(__file__).resolve().parents[3]


def git_commit_hash() -> str:
    """Short git hash for experiment tracking."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=_repo_root(),
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def write_run_info(
    output_dir: Path,
    configs: Dict[str, Any],
    notes: Optional[Dict[str, Any]] = None,
    device: Optional[str] = None,
) -> Path:
    """
    Persist minimal metadata for reproducibility.

    Args:
        output_dir: Directory where the JSON will be written.
        configs: Mapping of config names to dicts (e.g., {"data": ..., "model": ...}).
        notes: Optional free-form metadata (script name, purpose, etc.).
        device: Device string override; defaults to detected device.
    """
    ensure_dir(output_dir)
    def _to_jsonable(value: Any):
        if isinstance(value, dict):
            return {k: _to_jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_to_jsonable(v) for v in value]
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, torch.device):
            return str(value)
        return value

    payload: Dict[str, Any] = {
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "git_commit": git_commit_hash(),
        "device": device or ("cuda" if torch.cuda.is_available() else "cpu"),
        "configs": _to_jsonable(configs),
    }
    if notes:
        payload["notes"] = notes

    run_file = output_dir / "run_info.json"
    run_file.write_text(json.dumps(payload, indent=2))
    return run_file
