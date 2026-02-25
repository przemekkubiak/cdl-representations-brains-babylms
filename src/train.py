"""
Training script for BabyLM models with checkpointing.
"""

import os
import logging
import argparse
from pathlib import Path
from typing import Optional, Dict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    get_scheduler,
    set_seed
)
from tqdm import tqdm
import json

from src.config import ModelConfig, TrainingConfig, get_config_for_size
from src.data_loader import load_babylm_data, create_dataloaders

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)


class Trainer:
    """Trainer class with checkpointing support."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainingConfig,
        tokenizer
    ):
        """
        Initialize trainer.
        
        Parameters
        ----------
        model : nn.Module
            Model to train
        train_loader : DataLoader
            Training data loader
        val_loader : DataLoader
            Validation data loader
        config : TrainingConfig
            Training configuration
        tokenizer : transformers.PreTrainedTokenizer
            Tokenizer
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.tokenizer = tokenizer
        
        # Setup device
        self.device = self._get_device()
        self.model.to(self.device)
        
        # Setup optimizer
        self.optimizer = self._create_optimizer()
        
        # Setup scheduler
        total_steps = len(train_loader) * config.num_train_epochs
        self.scheduler = get_scheduler(
            name=config.scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=total_steps
        )
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        
        # Create output directory
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        Path(config.log_dir).mkdir(parents=True, exist_ok=True)
        
        # Load from checkpoint if specified
        if config.checkpoint_path:
            self.load_checkpoint(config.checkpoint_path)
    
    def _get_device(self) -> torch.device:
        """Get training device."""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        return torch.device(self.config.device)
    
    def _create_optimizer(self):
        """Create optimizer."""
        # Separate weight decay for different parameter groups
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters()
                          if not any(nd in n for nd in no_decay)],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters()
                          if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        
        return torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.config.learning_rate,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            eps=self.config.adam_epsilon
        )
    
    def save_checkpoint(self, suffix: str = ""):
        """Save training checkpoint."""
        checkpoint_dir = Path(self.config.output_dir)
        
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "epoch": self.current_epoch,
            "best_val_loss": self.best_val_loss,
            "config": self.config.__dict__
        }
        
        if suffix:
            checkpoint_path = checkpoint_dir / f"checkpoint_{suffix}.pt"
        else:
            checkpoint_path = checkpoint_dir / f"checkpoint_step_{self.global_step}.pt"
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        # Clean up old checkpoints
        self._cleanup_checkpoints(checkpoint_dir)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(checkpoint_dir)
    
    def _cleanup_checkpoints(self, checkpoint_dir: Path):
        """Keep only the last N checkpoints."""
        if self.config.save_total_limit is None:
            return
        
        checkpoints = sorted(
            checkpoint_dir.glob("checkpoint_step_*.pt"),
            key=lambda x: int(x.stem.split("_")[-1])
        )
        
        if len(checkpoints) > self.config.save_total_limit:
            for ckpt in checkpoints[:-self.config.save_total_limit]:
                ckpt.unlink()
                logger.info(f"Deleted old checkpoint: {ckpt}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint."""
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.current_epoch = checkpoint["epoch"]
        self.best_val_loss = checkpoint.get("best_val_loss", float('inf'))
        
        logger.info(f"Resumed from epoch {self.current_epoch}, step {self.global_step}")
    
    def train(self):
        """Main training loop."""
        logger.info("Starting training...")
        logger.info(f"Device: {self.device}")
        logger.info(f"Total epochs: {self.config.num_train_epochs}")
        logger.info(f"Batch size: {self.config.batch_size}")
        logger.info(f"Total steps: {len(self.train_loader) * self.config.num_train_epochs}")
        
        for epoch in range(self.current_epoch, self.config.num_train_epochs):
            self.current_epoch = epoch
            logger.info(f"\nEpoch {epoch + 1}/{self.config.num_train_epochs}")
            
            # Training
            train_loss = self._train_epoch()
            logger.info(f"Train loss: {train_loss:.4f}")
            
            # Validation
            if self.config.do_eval:
                val_loss = self._evaluate()
                logger.info(f"Validation loss: {val_loss:.4f}")
                
                # Save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint(suffix="best")
            
            # Save epoch checkpoint
            self.save_checkpoint(suffix=f"epoch_{epoch + 1}")
        
        logger.info("Training complete!")
    
    def _train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        
        progress_bar = tqdm(self.train_loader, desc="Training")
        
        for batch in progress_bar:
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(**batch)
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm
                )
            
            # Optimizer step
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            
            # Update metrics
            total_loss += loss.item()
            self.global_step += 1
            
            # Logging
            if self.global_step % self.config.logging_steps == 0:
                avg_loss = total_loss / self.config.logging_steps
                progress_bar.set_postfix({
                    "loss": f"{avg_loss:.4f}",
                    "lr": f"{self.scheduler.get_last_lr()[0]:.2e}"
                })
                total_loss = 0
            
            # Save checkpoint
            if self.global_step % self.config.save_steps == 0:
                self.save_checkpoint()
        
        return total_loss / len(self.train_loader)
    
    @torch.no_grad()
    def _evaluate(self) -> float:
        """Evaluate on validation set."""
        self.model.eval()
        total_loss = 0
        
        progress_bar = tqdm(self.val_loader, desc="Evaluating")
        
        for batch in progress_bar:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            outputs = self.model(**batch)
            total_loss += outputs.loss.item()
        
        return total_loss / len(self.val_loader)


def main():
    parser = argparse.ArgumentParser(description="Train BabyLM models")
    parser.add_argument(
        "--dataset_size",
        type=str,
        required=True,
        choices=["60M", "100M", "150M"],
        help="Dataset size to train on"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/babylm",
        help="Directory containing training data"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="gpt2",
        choices=["gpt2", "bert", "roberta"],
        help="Model architecture to use"
    )
    
    args = parser.parse_args()
    
    # Get configuration
    config = get_config_for_size(args.dataset_size)
    config.data_dir = args.data_dir
    config.checkpoint_path = args.checkpoint
    
    # Set seed
    set_seed(config.seed)
    
    # Load data
    logger.info("Loading data...")
    dataset_dict = load_babylm_data(
        data_dir=config.data_dir,
        dataset_size=args.dataset_size,
        max_length=config.max_seq_length
    )
    
    dataloaders = create_dataloaders(
        dataset_dict,
        batch_size=config.batch_size,
        eval_batch_size=config.eval_batch_size,
        num_workers=config.dataloader_num_workers
    )
    
    # Initialize model
    logger.info("Initializing model...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model_config = AutoConfig.from_pretrained(
        args.model_type,
        vocab_size=len(tokenizer),
        n_positions=config.max_seq_length,
        n_ctx=config.max_seq_length,
        n_embd=768,
        n_layer=12,
        n_head=12
    )
    
    model = AutoModelForCausalLM.from_config(model_config)
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_loader=dataloaders["train"],
        val_loader=dataloaders["val"],
        config=config,
        tokenizer=tokenizer
    )
    
    # Train
    trainer.train()


if __name__ == "__main__":
    main()
