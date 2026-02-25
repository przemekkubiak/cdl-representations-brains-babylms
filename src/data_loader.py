"""
Data loading and preprocessing utilities for BabyLM training.
"""

import os
from pathlib import Path
from typing import Optional, Dict, List
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset, DatasetDict
import logging

logger = logging.getLogger(__name__)


class BabyLMDataset(Dataset):
    """Dataset for BabyLM training."""
    
    def __init__(
        self,
        texts: List[str],
        tokenizer,
        max_length: int = 512
    ):
        """
        Initialize BabyLM dataset.
        
        Parameters
        ----------
        texts : List[str]
            List of text samples
        tokenizer : transformers.PreTrainedTokenizer
            Tokenizer
        max_length : int
            Maximum sequence length
        """
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # For language modeling, labels are the same as input_ids
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": encoding["input_ids"].squeeze()
        }


def load_babylm_data(
    data_dir: str,
    dataset_size: str,
    tokenizer_name: str = "gpt2",
    train_split: float = 0.9,
    max_length: int = 512,
    cache_dir: Optional[str] = None
) -> DatasetDict:
    """
    Load BabyLM training data.
    
    Parameters
    ----------
    data_dir : str
        Directory containing the data
    dataset_size : str
        Dataset size ('60M', '100M', or '150M')
    tokenizer_name : str
        Name of the tokenizer
    train_split : float
        Fraction of data for training
    max_length : int
        Maximum sequence length
    cache_dir : str, optional
        Cache directory
        
    Returns
    -------
    DatasetDict
        Dictionary with 'train' and 'validation' splits
    """
    data_path = Path(data_dir) / dataset_size
    
    if not data_path.exists():
        raise FileNotFoundError(
            f"Data directory not found: {data_path}\n"
            f"Please prepare your training data first. See data/README.md"
        )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # Add pad token if it doesn't exist (for GPT-2)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load text files
    logger.info(f"Loading data from {data_path}")
    
    # Try to load from text files
    text_files = list(data_path.glob("*.txt"))
    
    if not text_files:
        raise FileNotFoundError(
            f"No .txt files found in {data_path}\n"
            f"Please add your training data."
        )
    
    # Load dataset using HuggingFace datasets
    dataset = load_dataset(
        "text",
        data_files={"train": str(text_files[0])},
        cache_dir=cache_dir,
        split="train"
    )
    
    # Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length"
        )
    
    logger.info("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing"
    )
    
    # Add labels (same as input_ids for language modeling)
    def add_labels(examples):
        examples["labels"] = examples["input_ids"].copy()
        return examples
    
    tokenized_dataset = tokenized_dataset.map(
        add_labels,
        batched=True,
        desc="Adding labels"
    )
    
    # Split into train/val
    split_dataset = tokenized_dataset.train_test_split(
        test_size=1.0 - train_split,
        seed=42
    )
    
    dataset_dict = DatasetDict({
        "train": split_dataset["train"],
        "validation": split_dataset["test"]
    })
    
    logger.info(f"Train size: {len(dataset_dict['train'])}")
    logger.info(f"Validation size: {len(dataset_dict['validation'])}")
    
    return dataset_dict


def create_dataloaders(
    dataset_dict: DatasetDict,
    batch_size: int = 32,
    eval_batch_size: int = 64,
    num_workers: int = 4
) -> Dict[str, DataLoader]:
    """
    Create data loaders from dataset dictionary.
    
    Parameters
    ----------
    dataset_dict : DatasetDict
        Dataset dictionary with train/validation splits
    batch_size : int
        Training batch size
    eval_batch_size : int
        Evaluation batch size
    num_workers : int
        Number of dataloader workers
        
    Returns
    -------
    Dict[str, DataLoader]
        Dictionary with 'train' and 'val' dataloaders
    """
    train_loader = DataLoader(
        dataset_dict["train"],
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        dataset_dict["validation"],
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return {
        "train": train_loader,
        "val": val_loader
    }
