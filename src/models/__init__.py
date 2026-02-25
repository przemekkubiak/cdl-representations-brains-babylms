"""
Utilities for loading BabyLM models and extracting representations.
"""

import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer
from typing import List, Dict, Optional


class BabyLMExtractor:
    """Extract representations from BabyLM models."""
    
    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        checkpoint_path: Optional[str] = None
    ):
        """
        Initialize BabyLM extractor.
        
        Parameters
        ----------
        model_name : str
            Name or path to the model (e.g., 'gpt2' or path to checkpoint dir)
        device : str, optional
            Device to use ('cuda', 'cpu', 'mps')
        checkpoint_path : str, optional
            Path to specific checkpoint file (.pt) to load
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load from checkpoint if provided
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            # Assume model config is saved in checkpoint dir
            self.model = AutoModel.from_pretrained(model_name).to(self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model = AutoModel.from_pretrained(model_name).to(self.device)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model.eval()
    
    def extract_representations(
        self,
        texts: List[str],
        layers: Optional[List[int]] = None,
        aggregation: str = "mean"
    ) -> Dict[int, np.ndarray]:
        """
        Extract representations from specified layers.
        
        Parameters
        ----------
        texts : List[str]
            Input texts
        layers : List[int], optional
            Layers to extract from (None = all layers)
        aggregation : str
            How to aggregate tokens ('mean', 'last', 'cls')
            
        Returns
        -------
        Dict[int, np.ndarray]
            Dictionary mapping layer numbers to representations
        """
        representations = {}
        
        with torch.no_grad():
            for text in texts:
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                ).to(self.device)
                
                outputs = self.model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states
                
                target_layers = layers or range(len(hidden_states))
                
                for layer_idx in target_layers:
                    layer_repr = hidden_states[layer_idx].cpu().numpy()
                    
                    if aggregation == "mean":
                        layer_repr = layer_repr.mean(axis=1)
                    elif aggregation == "last":
                        layer_repr = layer_repr[:, -1, :]
                    elif aggregation == "cls":
                        layer_repr = layer_repr[:, 0, :]
                    
                    if layer_idx not in representations:
                        representations[layer_idx] = []
                    representations[layer_idx].append(layer_repr)
        
        # Concatenate all representations
        for layer_idx in representations:
            representations[layer_idx] = np.vstack(representations[layer_idx])
        
        return representations
