#!/usr/bin/env python3
"""Export report images from embedded notebook outputs (no notebook execution)."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NOTEBOOK = REPO_ROOT / "codes" / "CA3_Score_Based_Models.ipynb"
DEFAULT_IMAGES_DIR = REPO_ROOT / "images"
DEFAULT_MANIFEST = REPO_ROOT / "report" / "asset_manifest.json"


def _build_mapping() -> list[dict[str, Any]]:
    mapping: list[dict[str, Any]] = []

    for img_idx, file_idx in enumerate([0, 1]):
        mapping.append(
            {
                "cell_index": 11,
                "image_index": img_idx,
                "filename": f"output_cell_9_img_{file_idx}.png",
            }
        )

    for img_idx, file_idx in enumerate([3, 6, 8, 10, 12, 16, 19, 21, 23, 25, 29, 32, 34, 36, 38]):
        mapping.append(
            {
                "cell_index": 52,
                "image_index": img_idx,
                "filename": f"output_cell_25_img_{file_idx}.png",
            }
        )

    for img_idx, file_idx in enumerate([3, 6, 9, 11, 13, 15, 17]):
        mapping.append(
            {
                "cell_index": 54,
                "image_index": img_idx,
                "filename": f"output_cell_27_img_{file_idx}.png",
            }
        )

    for img_idx, file_idx in enumerate([2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32]):
        mapping.append(
            {
                "cell_index": 56,
                "image_index": img_idx,
                "filename": f"output_cell_29_img_{file_idx}.png",
            }
        )

    if len(mapping) != 35:
        raise RuntimeError(f"Expected 35 mapping entries, got {len(mapping)}")
    return mapping


EXTRACTION_MAPPING = _build_mapping()


def _atomic_write_bytes(target: Path, payload: bytes) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    with tmp.open("wb") as fh:
        fh.write(payload)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, target)


def _atomic_write_json(target: Path, data: Any) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, target)


def _collect_png_outputs(code_cell: dict[str, Any]) -> list[bytes]:
    png_blobs: list[bytes] = []
    for out in code_cell.get("outputs", []):
        if not isinstance(out, dict):
            continue
        data = out.get("data", {})
        if not isinstance(data, dict):
            continue
        image = data.get("image/png")
        if not image:
            continue
        if isinstance(image, list):
            image = "".join(image)
        if not isinstance(image, str):
            continue
        png_blobs.append(base64.b64decode(image))
    return png_blobs


def export_images(notebook: Path, images_dir: Path, manifest: Path) -> None:
    notebook_data = json.loads(notebook.read_text(encoding="utf-8"))
    cells = notebook_data.get("cells", [])

    manifest_entries: list[dict[str, Any]] = []
    exported = 0

    for item in EXTRACTION_MAPPING:
        cell_index = item["cell_index"]
        image_index = item["image_index"]
        filename = item["filename"]

        if cell_index >= len(cells):
            raise IndexError(f"Cell index {cell_index} out of range for {notebook}")
        cell = cells[cell_index]
        if cell.get("cell_type") != "code":
            raise ValueError(f"Cell {cell_index} is not a code cell")

        png_outputs = _collect_png_outputs(cell)
        if image_index >= len(png_outputs):
            raise IndexError(
                f"Cell {cell_index} has {len(png_outputs)} png outputs; "
                f"cannot access image index {image_index}"
            )

        png_blob = png_outputs[image_index]
        target = images_dir / filename
        _atomic_write_bytes(target, png_blob)

        digest = hashlib.sha256(png_blob).hexdigest()
        try:
            notebook_ref = str(notebook.relative_to(REPO_ROOT))
        except ValueError:
            notebook_ref = str(notebook)

        manifest_entries.append(
            {
                "bytes": len(png_blob),
                "cell_index": cell_index,
                "filename": filename,
                "image_index": image_index,
                "notebook": notebook_ref,
                "sha256": digest,
            }
        )
        exported += 1

    _atomic_write_json(manifest, manifest_entries)
    print(f"Exported {exported} image(s) from notebook outputs.")
    print(f"Manifest written to: {manifest}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export report figures from embedded notebook outputs."
    )
    parser.add_argument("--notebook", type=Path, default=DEFAULT_NOTEBOOK)
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    export_images(
        notebook=args.notebook.resolve(),
        images_dir=args.images_dir.resolve(),
        manifest=args.manifest.resolve(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
