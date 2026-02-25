"""
Utility script to resume training from a checkpoint.
"""

import argparse
from pathlib import Path
from src.train import main


def find_latest_checkpoint(checkpoint_dir: str) -> str:
    """
    Find the latest checkpoint in a directory.
    
    Parameters
    ----------
    checkpoint_dir : str
        Directory containing checkpoints
        
    Returns
    -------
    str
        Path to latest checkpoint
    """
    checkpoint_path = Path(checkpoint_dir)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
    
    # Look for step checkpoints
    checkpoints = sorted(
        checkpoint_path.glob("checkpoint_step_*.pt"),
        key=lambda x: int(x.stem.split("_")[-1])
    )
    
    if not checkpoints:
        # Look for epoch checkpoints
        checkpoints = sorted(
            checkpoint_path.glob("checkpoint_epoch_*.pt"),
            key=lambda x: int(x.stem.split("_")[-1])
        )
    
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    
    return str(checkpoints[-1])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resume BabyLM training from checkpoint")
    parser.add_argument(
        "--dataset_size",
        type=str,
        required=True,
        choices=["60M", "100M", "150M"],
        help="Dataset size"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Directory containing checkpoints (defaults to checkpoints/babylm_{size})"
    )
    parser.add_argument(
        "--use_best",
        action="store_true",
        help="Resume from best checkpoint instead of latest"
    )
    
    args = parser.parse_args()
    
    # Determine checkpoint directory
    if args.checkpoint_dir is None:
        checkpoint_dir = f"checkpoints/babylm_{args.dataset_size}"
    else:
        checkpoint_dir = args.checkpoint_dir
    
    # Find checkpoint
    if args.use_best:
        checkpoint_path = Path(checkpoint_dir) / "checkpoint_best.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Best checkpoint not found: {checkpoint_path}")
        checkpoint_path = str(checkpoint_path)
    else:
        checkpoint_path = find_latest_checkpoint(checkpoint_dir)
    
    print(f"Resuming from checkpoint: {checkpoint_path}")
    
    # Update sys.argv for main()
    import sys
    sys.argv = [
        sys.argv[0],
        "--dataset_size", args.dataset_size,
        "--checkpoint", checkpoint_path
    ]
    
    main()
