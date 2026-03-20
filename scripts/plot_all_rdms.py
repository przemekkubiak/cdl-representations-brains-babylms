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


def load_rdm(npz_path: Path) -> np.ndarray | None:
    try:
        data = np.load(npz_path)
    except Exception:
        return None

    if "rdm" not in data:
        return None

    rdm = data["rdm"]
    if rdm.ndim != 2:
        return None

    return rdm


def global_scale(npz_files: list[Path]) -> tuple[float, float]:
    vals = []
    for npz_path in npz_files:
        rdm = load_rdm(npz_path)
        if rdm is None:
            continue
        vals.append(rdm[np.isfinite(rdm)])

    if not vals:
        return 0.0, 1.0

    all_vals = np.concatenate(vals)
    return float(np.min(all_vals)), float(np.max(all_vals))


def plot_rdm(
    npz_path: Path,
    output_dir: Path,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
) -> Path | None:
    rdm = load_rdm(npz_path)
    if rdm is None:
        return None

    out_name = npz_path.stem + ".png"
    out_path = output_dir / out_name

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(
        rdm,
        cmap=cmap,
        interpolation="nearest",
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
    )
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
    parser.add_argument("--vmin", type=float, default=None, help="Optional fixed lower bound for color scale")
    parser.add_argument("--vmax", type=float, default=None, help="Optional fixed upper bound for color scale")

    args = parser.parse_args()

    input_dirs = [Path(p) for p in args.input_dirs]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    npz_files = find_rdm_files(input_dirs)
    if not npz_files:
        print("No .npz files found.")
        return

    scale_vmin, scale_vmax = global_scale(npz_files)
    if args.vmin is not None:
        scale_vmin = args.vmin
    if args.vmax is not None:
        scale_vmax = args.vmax
    print(f"Using uniform color scale: vmin={scale_vmin:.6f}, vmax={scale_vmax:.6f}")

    count = 0
    for npz_file in npz_files:
        out = plot_rdm(
            npz_file,
            output_dir=output_dir,
            cmap=args.cmap,
            vmin=scale_vmin,
            vmax=scale_vmax,
        )
        if out is not None:
            count += 1
            print(f"Saved: {out}")

    print(f"Done. Generated {count} heatmaps in {output_dir}")


if __name__ == "__main__":
    main()
