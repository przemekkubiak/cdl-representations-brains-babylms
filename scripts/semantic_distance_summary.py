#!/usr/bin/env python
"""Summarize semantic-distance structure from saved session RDMs."""

from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rsa.semantic_distance_analysis import summarize_directory


def main():
    parser = argparse.ArgumentParser(description="Summarize semantic-distance statistics from session RDMs")
    parser.add_argument("--input-dir", type=str, default="data/processed/fmri")
    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--roi-label", type=str)
    parser.add_argument("--task", type=str, default="Sem", choices=["Sem", "Phon", "Gram", "Plaus"])
    parser.add_argument("--sessions", nargs="+")
    args = parser.parse_args()

    pairwise_df, contrast_df = summarize_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        roi_label=args.roi_label,
        sessions=args.sessions,
    )

    print(f"Saved pairwise summary rows: {len(pairwise_df)}")
    print(f"Saved contrast summary rows: {len(contrast_df)}")


if __name__ == "__main__":
    main()