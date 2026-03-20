#!/usr/bin/env python
"""Generate heatmap visualizations for all RDM .npz files.

Looks for .npz files containing an `rdm` array and saves PNG heatmaps next to or
inside the requested output directory.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def find_rdm_files(input_dirs: list[Path]) -> list[Path]:
    files: list[Path] = []
    for d in input_dirs:
        if not d.exists():
            continue
        files.extend(sorted(d.glob("*.npz")))
    return files


def plot_rdm(npz_path: Path, output_dir: Path, cmap: str = "viridis") -> Path | None:
    try:
        data = np.load(npz_path)
    except Exception:
        return None

    if "rdm" not in data:
        return None

    rdm = data["rdm"]
    if rdm.ndim != 2:
        return None

    out_name = npz_path.stem + ".png"
    out_path = output_dir / out_name

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(rdm, cmap=cmap, interpolation="nearest", aspect="auto")
    ax.set_title(npz_path.stem)
    ax.set_xlabel("Stimulus index")
    ax.set_ylabel("Stimulus index")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot all RDM npz files to PNG heatmaps")
    parser.add_argument(
        "--input-dirs",
        nargs="+",
        default=["data/processed/fmri", "data/processed/language_models"],
        help="Directories to scan for .npz files",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed/rdm_visualizations",
        help="Directory for PNG outputs",
    )
    parser.add_argument("--cmap", default="viridis", help="Matplotlib colormap")

    args = parser.parse_args()

    input_dirs = [Path(p) for p in args.input_dirs]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    npz_files = find_rdm_files(input_dirs)
    if not npz_files:
        print("No .npz files found.")
        return

    count = 0
    for npz_file in npz_files:
        out = plot_rdm(npz_file, output_dir=output_dir, cmap=args.cmap)
        if out is not None:
            count += 1
            print(f"Saved: {out}")

    print(f"Done. Generated {count} heatmaps in {output_dir}")


if __name__ == "__main__":
    main()
