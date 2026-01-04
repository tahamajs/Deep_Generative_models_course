#!/usr/bin/env python3
"""Extract embedded images from an executed Jupyter notebook (.ipynb).

This script DOES NOT re-run the notebook. It only extracts images already
stored in the notebook JSON (cell outputs + markdown attachments).

Supported mime types:
- image/png  -> .png
- image/jpeg -> .jpg
- image/svg+xml -> .svg

Usage:
  python extract_notebook_images.py path/to/notebook.ipynb --out extracted_images

Notes:
- For large notebooks, images can be numerous; filenames include cell/output
  indices for traceability.
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
from typing import Any, Iterable, Optional, Tuple


IMAGE_MIME_TO_EXT = {
    "image/png": "png",
    "image/jpeg": "jpg",
    "image/svg+xml": "svg",
}


def _extract_first_markdown_heading(cell_source: Any) -> Optional[str]:
    """Return first markdown heading text if present (without leading #)."""
    text = _as_text(cell_source)
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            # Strip any number of leading # and whitespace
            heading = line.lstrip("#").strip()
            return heading or None
    return None


def _as_text(value: Any) -> str:
    """Normalize notebook data fields to a single string."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        # In notebooks, text data is sometimes a list of lines.
        return "".join(str(part) for part in value)
    return str(value)


def _iter_cells(nb: dict) -> Iterable[Tuple[int, dict]]:
    cells = nb.get("cells", [])
    for i, cell in enumerate(cells, start=1):
        if isinstance(cell, dict):
            yield i, cell


def _write_bytes(path: Path, content: bytes, overwrite: bool) -> bool:
    if path.exists() and not overwrite:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return True


def _write_text(path: Path, content: str, overwrite: bool) -> bool:
    if path.exists() and not overwrite:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return True


def _safe_slug(text: str, max_len: int = 64) -> str:
    # Keep filenames readable; avoid OS-unfriendly characters.
    cleaned = "".join(ch if (ch.isalnum() or ch in "-_" or ch.isspace()) else "_" for ch in text)
    cleaned = "_".join(cleaned.split())
    if not cleaned:
        return "untitled"
    return cleaned[:max_len].rstrip("_")


def extract_images_from_notebook(
    notebook_path: Path,
    out_dir: Path,
    prefix: str = "nb",
    overwrite: bool = False,
    use_headings: bool = True,
    max_slug_len: int = 64,
) -> int:
    nb = json.loads(notebook_path.read_text(encoding="utf-8"))

    extracted = 0

    # Track the most recent markdown heading to build readable filenames.
    last_heading: Optional[str] = None

    # 1) Extract from cell outputs.
    for cell_index, cell in _iter_cells(nb):
        if use_headings and cell.get("cell_type") == "markdown":
            heading = _extract_first_markdown_heading(cell.get("source"))
            if heading:
                last_heading = heading

        outputs = cell.get("outputs")
        if not isinstance(outputs, list):
            continue

        for output_index, output in enumerate(outputs, start=1):
            if not isinstance(output, dict):
                continue

            data = output.get("data")
            if not isinstance(data, dict):
                continue

            for mime, ext in IMAGE_MIME_TO_EXT.items():
                if mime not in data:
                    continue

                raw = data[mime]
                if use_headings and last_heading:
                    label = _safe_slug(last_heading, max_len=max_slug_len)
                    stem = f"{prefix}_{label}_cell{cell_index:03d}_out{output_index:02d}"
                else:
                    stem = f"{prefix}_cell{cell_index:03d}_out{output_index:02d}"
                out_path = out_dir / f"{stem}.{ext}"

                if mime in ("image/png", "image/jpeg"):
                    b64_text = _as_text(raw).strip()
                    if not b64_text:
                        continue
                    try:
                        payload = base64.b64decode(b64_text, validate=False)
                    except Exception:
                        # Some notebooks store base64 with newlines; retry without strictness.
                        payload = base64.b64decode(b64_text)

                    if _write_bytes(out_path, payload, overwrite=overwrite):
                        extracted += 1
                else:
                    svg_text = _as_text(raw)
                    if not svg_text.strip():
                        continue
                    if _write_text(out_path, svg_text, overwrite=overwrite):
                        extracted += 1

    # 2) Extract markdown attachments (common for pasted images).
    for cell_index, cell in _iter_cells(nb):
        if cell.get("cell_type") != "markdown":
            continue

        if use_headings:
            heading = _extract_first_markdown_heading(cell.get("source"))
            if heading:
                last_heading = heading

        attachments = cell.get("attachments")
        if not isinstance(attachments, dict):
            continue

        for attachment_name, attachment in attachments.items():
            if not isinstance(attachment, dict):
                continue

            for mime, ext in IMAGE_MIME_TO_EXT.items():
                if mime not in attachment:
                    continue

                raw = attachment[mime]
                att_slug = _safe_slug(str(attachment_name), max_len=max_slug_len)
                if use_headings and last_heading:
                    label = _safe_slug(last_heading, max_len=max_slug_len)
                    stem = f"{prefix}_{label}_cell{cell_index:03d}_att_{att_slug}"
                else:
                    stem = f"{prefix}_cell{cell_index:03d}_att_{att_slug}"
                out_path = out_dir / f"{stem}.{ext}"

                if mime in ("image/png", "image/jpeg"):
                    b64_text = _as_text(raw).strip()
                    if not b64_text:
                        continue
                    payload = base64.b64decode(b64_text)
                    if _write_bytes(out_path, payload, overwrite=overwrite):
                        extracted += 1
                else:
                    svg_text = _as_text(raw)
                    if not svg_text.strip():
                        continue
                    if _write_text(out_path, svg_text, overwrite=overwrite):
                        extracted += 1

    return extracted


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract embedded images from an executed .ipynb")
    parser.add_argument("notebook", type=Path, help="Path to the .ipynb notebook")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("extracted_notebook_images"),
        help="Output directory (default: ./extracted_notebook_images)",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="nb",
        help="Filename prefix (default: nb)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files (default: false)",
    )
    parser.add_argument(
        "--clean-out",
        action="store_true",
        help="Delete existing extracted images in --out before extracting (default: false)",
    )
    parser.add_argument(
        "--no-headings",
        action="store_true",
        help="Do not use markdown headings in filenames (default: false)",
    )
    parser.add_argument(
        "--max-slug-len",
        type=int,
        default=64,
        help="Max length for heading/attachment slugs in filenames (default: 64)",
    )

    args = parser.parse_args()

    notebook_path: Path = args.notebook
    out_dir: Path = args.out

    if not notebook_path.exists():
        raise SystemExit(f"Notebook not found: {notebook_path}")
    if notebook_path.suffix.lower() != ".ipynb":
        raise SystemExit("Input file must be a .ipynb")

    out_dir.mkdir(parents=True, exist_ok=True)

    if args.clean_out:
        removed = 0
        for ext in set(IMAGE_MIME_TO_EXT.values()):
            for p in out_dir.glob(f"*.{ext}"):
                try:
                    p.unlink()
                    removed += 1
                except OSError:
                    pass
        if removed:
            print(f"Removed {removed} existing image(s) from {out_dir.resolve()}")

    count = extract_images_from_notebook(
        notebook_path=notebook_path,
        out_dir=out_dir,
        prefix=args.prefix,
        overwrite=args.overwrite,
        use_headings=not args.no_headings,
        max_slug_len=args.max_slug_len,
    )

    print(f"Extracted {count} image(s) -> {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
