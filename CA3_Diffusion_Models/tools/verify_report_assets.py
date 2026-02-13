#!/usr/bin/env python3
"""Verify report image references exist in images directory (and manifest)."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEX = REPO_ROOT / "report" / "DGM_CA3_Complete_Report.tex"
DEFAULT_IMAGES_DIR = REPO_ROOT / "images"
DEFAULT_MANIFEST = REPO_ROOT / "report" / "asset_manifest.json"

INCLUDEGRAPHICS_RE = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}")


def _parse_referenced_images(tex_path: Path) -> list[str]:
    text = tex_path.read_text(encoding="utf-8")
    refs = INCLUDEGRAPHICS_RE.findall(text)
    if not refs:
        raise RuntimeError(f"No \\includegraphics references found in {tex_path}")
    return refs


def _resolve_under_images(images_dir: Path, reference: str) -> Path:
    ref_path = Path(reference.strip())
    candidates = []
    if ref_path.is_absolute():
        candidates.append(ref_path)
    else:
        candidates.append(images_dir / ref_path)
        if ref_path.parts:
            candidates.append(images_dir / ref_path.name)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[-1]


def _load_manifest_filenames(manifest_path: Path) -> set[str]:
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Manifest at {manifest_path} must be a JSON list")
    names: set[str] = set()
    for row in data:
        if not isinstance(row, dict):
            continue
        filename = row.get("filename")
        if isinstance(filename, str):
            names.add(filename)
    return names


def verify_assets(tex_path: Path, images_dir: Path, manifest_path: Path | None) -> int:
    refs = _parse_referenced_images(tex_path)
    missing_files: list[str] = []
    resolved_names: list[str] = []

    for ref in refs:
        resolved = _resolve_under_images(images_dir, ref)
        resolved_names.append(Path(ref).name)
        if not resolved.exists():
            missing_files.append(ref)

    if missing_files:
        print("Missing report assets:")
        for ref in missing_files:
            print(f"  - {ref}")
        return 1

    if manifest_path is not None:
        manifest_names = _load_manifest_filenames(manifest_path)
        missing_in_manifest = sorted(set(resolved_names) - manifest_names)
        if missing_in_manifest:
            print(f"Manifest mismatch ({manifest_path}):")
            for name in missing_in_manifest:
                print(f"  - {name}")
            return 2
        print(
            f"Manifest check passed: {len(set(resolved_names))} referenced "
            f"asset(s) present in manifest."
        )

    print(f"Asset check passed: {len(refs)} \\includegraphics reference(s) resolved.")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate report image references.")
    parser.add_argument("--tex", type=Path, default=DEFAULT_TEX)
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIR)
    parser.add_argument(
        "--manifest",
        nargs="?",
        const=DEFAULT_MANIFEST,
        default=None,
        type=Path,
        help="Optionally verify references are present in manifest (default path used if no value is provided).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return verify_assets(
        tex_path=args.tex.resolve(),
        images_dir=args.images_dir.resolve(),
        manifest_path=args.manifest.resolve() if args.manifest is not None else None,
    )


if __name__ == "__main__":
    raise SystemExit(main())
