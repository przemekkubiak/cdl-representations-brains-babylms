"""
Configuration for BabyLM training experiments.
"""

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    
    # Architecture
    vocab_size: int = 50257  # GPT-2 tokenizer size
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_dropout_prob: float = 0.1
    attention_dropout_prob: float = 0.1
    max_position_embeddings: int = 512
    layer_norm_epsilon: float = 1e-5
    
    # Model type
    model_type: str = "gpt2"  # Options: 'gpt2', 'bert', 'roberta'


@dataclass
class TrainingConfig:
    """Training configuration."""
    
    # Data
    data_dir: str = "data/babylm"
    dataset_size: str = "100M"  # Options: '60M', '100M', '150M'
    max_seq_length: int = 512
    
    # Training hyperparameters
    num_train_epochs: int = 10
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    learning_rate: float = 5e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    
    # Optimization
    optimizer: str = "adamw"
    scheduler: str = "linear"  # Options: 'linear', 'cosine', 'constant'
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    
    # Checkpointing
    output_dir: str = "checkpoints"
    save_steps: int = 1000
    save_total_limit: int = 3  # Keep only last 3 checkpoints
    checkpoint_path: Optional[str] = None  # Path to resume from
    
    # Logging
    logging_steps: int = 100
    eval_steps: int = 500
    log_dir: str = "logs"
    
    # Hardware
    device: str = "auto"  # Options: 'auto', 'cuda', 'cpu', 'mps'
    mixed_precision: bool = True  # Use fp16/bf16 training
    dataloader_num_workers: int = 4
    
    # Reproducibility
    seed: int = 42
    
    # Evaluation
    eval_batch_size: int = 64
    do_eval: bool = True
    
    # Experiment tracking
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None
    use_wandb: bool = False


@dataclass
class DataConfig:
    """Data preprocessing configuration."""
    
    # Tokenization
    tokenizer_name: str = "gpt2"
    use_fast_tokenizer: bool = True
    
    # Data splits
    train_split: float = 0.9
    val_split: float = 0.1
    
    # Processing
    num_proc: int = 4
    overwrite_cache: bool = False


def get_config_for_size(dataset_size: str) -> TrainingConfig:
    """
    Get training configuration for a specific dataset size.
    
    Parameters
    ----------
    dataset_size : str
        Dataset size ('60M', '100M', or '150M')
        
    Returns
    -------
    TrainingConfig
        Training configuration
    """
    base_config = TrainingConfig(dataset_size=dataset_size)
    
    # Adjust learning rate and warmup based on dataset size
    if dataset_size == "60M":
        base_config.learning_rate = 5e-4
        base_config.warmup_steps = 500
        base_config.num_train_epochs = 15
    elif dataset_size == "100M":
        base_config.learning_rate = 5e-4
        base_config.warmup_steps = 1000
        base_config.num_train_epochs = 10
    elif dataset_size == "150M":
        base_config.learning_rate = 3e-4
        base_config.warmup_steps = 1500
        base_config.num_train_epochs = 8
    else:
        raise ValueError(f"Unknown dataset size: {dataset_size}")
    
    base_config.output_dir = f"checkpoints/babylm_{dataset_size}"
    base_config.log_dir = f"logs/babylm_{dataset_size}"
    
    return base_config
