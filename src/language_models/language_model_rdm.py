"""
Language model RDM computation.
Loads pretrained language models and extracts representational dissimilarity matrices
for stimulus words.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pickle

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr

import torch
from transformers import AutoTokenizer, AutoModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LanguageModelEmbeddingExtractor:
    """
    Extract embeddings from pretrained language models.
    Supports both HuggingFace models and BabyLM models.
    """
    
    def __init__(
        self,
        model_name: str,
        cache_dir: str = ".cache/huggingface",
        layer: int = -1,  # -1 for last layer
        pooling: str = "mean"  # mean, max, or cls
    ):
        """
        Initialize language model.
        
        Args:
            model_name: Model name or path (HuggingFace format)
            cache_dir: HuggingFace cache directory
            layer: Which layer to extract (0-indexed, -1 for last)
            pooling: How to pool subword tokens (mean, max, cls)
        """
        self.model_name = model_name
        self.layer = layer
        self.pooling = pooling
        
        logger.info(f"Loading model: {model_name}")
        os.environ["HF_HOME"] = cache_dir
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                trust_remote_code=True
            )
            self.model = AutoModel.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                output_hidden_states=True,
                trust_remote_code=True
            )
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise
        
        # Move to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Model loaded on device: {self.device}")
    
    def extract_word_embedding(self, word: str) -> np.ndarray:
        """
        Extract embedding for a single word.
        
        Args:
            word: Word to embed
            
        Returns:
            Embedding vector (numpy array)
        """
        with torch.no_grad():
            # Tokenize
            inputs = self.tokenizer(
                word,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(self.device)
            
            # Forward pass
            outputs = self.model(**inputs)
            
            # Extract hidden states
            hidden_states = outputs.hidden_states
            
            # Select layer (-1 for last)
            layer_output = hidden_states[self.layer]  # (batch, seq_len, hidden_dim)
            
            # Convert to float32 (handles bfloat16 models)
            layer_output = layer_output.to(torch.float32)
            
            # Pool across tokens
            if self.pooling == "mean":
                embedding = layer_output.mean(dim=1)[0].cpu().numpy()
            elif self.pooling == "max":
                embedding = layer_output.max(dim=1)[0][0].cpu().numpy()
            elif self.pooling == "cls":
                embedding = layer_output[0, 0].cpu().numpy()
            else:
                raise ValueError(f"Unknown pooling: {self.pooling}")
        
        return embedding
    
    def extract_batch_embeddings(self, words: List[str]) -> np.ndarray:
        """
        Extract embeddings for multiple words.
        
        Args:
            words: List of words to embed
            
        Returns:
            Embedding matrix (n_words, embedding_dim)
        """
        embeddings = []
        
        for i, word in enumerate(words):
            if (i + 1) % 50 == 0:
                logger.info(f"Extracted {i + 1}/{len(words)} embeddings")
            
            try:
                embedding = self.extract_word_embedding(word)
                
                # Check for NaN
                if np.any(np.isnan(embedding)):
                    logger.warning(f"Word '{word}' produced NaN embedding, replacing with zeros")
                    embedding = np.nan_to_num(embedding, nan=0.0)
                
                embeddings.append(embedding)
            except Exception as e:
                logger.error(f"Failed to embed word '{word}': {e}")
                # Use zero vector as fallback
                embeddings.append(np.zeros(embeddings[0].shape if embeddings else 768))
        
        result = np.array(embeddings)
        
        # Final check and cleanup
        if np.any(np.isnan(result)):
            logger.warning(f"Cleaning {np.sum(np.isnan(result))} NaN values from embeddings")
            result = np.nan_to_num(result, nan=0.0)
        
        return result


class LanguageModelRDMComputer:
    """
    Compute RDMs from language model embeddings.
    Handles per-stimulus averaging and RDM computation.
    """
    
    def __init__(self, output_dir: str = "data/processed/language_models"):
        """
        Initialize RDM computer.
        
        Args:
            output_dir: Directory to save RDMs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.embeddings_cache = {}
    
    def load_stimulus_characteristics(
        self,
        characteristics_dir: str = "data/brain/ds003604/stimuli/Stimulus_Characteristics",
        task: str = "Sem"
    ) -> Tuple[pd.DataFrame, List[str], Dict]:
        """
        Load stimulus characteristics and extract word list.
        
        Args:
            characteristics_dir: Directory with stimulus characteristic TSV files
            task: Task name (Sem, Gram, Phon, Plaus)
            
        Returns:
            Tuple of (characteristics_df, unique_words, word_to_idx)
        """
        char_file = Path(characteristics_dir) / f"task-{task}_Stimulus_Characteristics.tsv"
        characteristics = pd.read_csv(char_file, sep="\t")
        
        # Extract unique words
        all_words = []
        for idx, row in characteristics.iterrows():
            word_a = row.get("word_A")
            word_b = row.get("word_B")
            if pd.notna(word_a):
                all_words.append(str(word_a).lower())
            if pd.notna(word_b):
                all_words.append(str(word_b).lower())
        
        # Get unique words in order
        unique_words = []
        seen = set()
        for word in all_words:
            if word not in seen:
                unique_words.append(word)
                seen.add(word)
        
        word_to_idx = {word: idx for idx, word in enumerate(unique_words)}
        
        logger.info(f"Task {task}: {len(unique_words)} unique words")
        
        return characteristics, unique_words, word_to_idx
    
    def compute_stimulus_rdm(
        self,
        characteristics: pd.DataFrame,
        embeddings: np.ndarray,
        word_list: List[str],
        word_to_idx: Dict[str, int]
    ) -> np.ndarray:
        """
        Compute RDM from stimulus embeddings.
        
        The RDM is computed as:
        1. For each stimulus (word pair), compute the average embedding
        2. Compute dissimilarity between all stimulus pairs
        
        Args:
            characteristics: Stimulus characteristics DataFrame
            embeddings: Embedding matrix (n_words, embedding_dim)
            word_list: List of unique words in order
            word_to_idx: Word to embedding index mapping
            
        Returns:
            RDM matrix (n_stimuli, n_stimuli)
        """
        # Check for NaN in embeddings
        if np.any(np.isnan(embeddings)):
            logger.warning(f"Found {np.sum(np.isnan(embeddings))} NaN values in embeddings")
            # Replace NaN with zeros
            embeddings = np.nan_to_num(embeddings, nan=0.0)
        
        stimulus_embeddings = []
        
        for idx, row in characteristics.iterrows():
            word_a = str(row["word_A"]).lower()
            word_b = str(row["word_B"]).lower()
            
            # Get embeddings
            emb_a = embeddings[word_to_idx[word_a]]
            emb_b = embeddings[word_to_idx[word_b]]
            
            # Average word pair embedding
            pair_embedding = (emb_a + emb_b) / 2
            stimulus_embeddings.append(pair_embedding)
        
        stimulus_embeddings = np.array(stimulus_embeddings)
        logger.info(f"Computed {len(stimulus_embeddings)} stimulus embeddings")
        
        # Check for NaN in stimulus embeddings
        if np.any(np.isnan(stimulus_embeddings)):
            logger.warning(f"Found {np.sum(np.isnan(stimulus_embeddings))} NaN values in stimulus embeddings")
            stimulus_embeddings = np.nan_to_num(stimulus_embeddings, nan=0.0)
        
        # Compute dissimilarity as 1 - cosine_similarity
        # First normalize embeddings with safety check
        norms = np.linalg.norm(stimulus_embeddings, axis=1, keepdims=True)
        
        # Handle zero-norm vectors (replace with small epsilon)
        norms = np.where(norms < 1e-8, 1.0, norms)
        embeddings_norm = stimulus_embeddings / norms
        
        # Ensure no NaN from normalization
        embeddings_norm = np.nan_to_num(embeddings_norm, nan=0.0)
        
        # Compute cosine similarity and convert to dissimilarity
        cosine_sim = np.dot(embeddings_norm, embeddings_norm.T)
        
        # Clip to valid range [-1, 1] to handle numerical errors
        cosine_sim = np.clip(cosine_sim, -1.0, 1.0)
        
        dissimilarity = 1 - cosine_sim
        
        # Ensure diagonal is exactly 0
        np.fill_diagonal(dissimilarity, 0)
        
        # Final NaN check
        if np.any(np.isnan(dissimilarity)):
            logger.error(f"RDM contains {np.sum(np.isnan(dissimilarity))} NaN values after computation")
            dissimilarity = np.nan_to_num(dissimilarity, nan=0.0)
        
        return dissimilarity
    
    def compute_rdm_from_embeddings(
        self,
        embeddings: np.ndarray,
        metric: str = "euclidean"
    ) -> np.ndarray:
        """
        Compute RDM from embedding matrix using pairwise distances.
        
        Args:
            embeddings: Embedding matrix (n_samples, embedding_dim)
            metric: Distance metric (euclidean, cosine, correlation)
            
        Returns:
            RDM matrix (n_samples, n_samples)
        """
        # Normalize for cosine distance
        if metric == "cosine":
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # Compute pairwise distances
        distances = pdist(embeddings, metric=metric)
        rdm = squareform(distances)
        
        # Ensure symmetric and diagonal is 0
        rdm = (rdm + rdm.T) / 2
        np.fill_diagonal(rdm, 0)
        
        return rdm
    
    def save_rdm(self, rdm: np.ndarray, filename: str):
        """Save RDM to file."""
        output_file = self.output_dir / filename
        np.savez_compressed(output_file, rdm=rdm)
        logger.info(f"Saved RDM to {output_file}")
        return output_file
    
    def load_rdm(self, filename: str) -> np.ndarray:
        """Load RDM from file."""
        rdm_file = self.output_dir / filename
        data = np.load(rdm_file)
        return data["rdm"]


def main():
    """Example usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Compute language model RDMs")
    parser.add_argument("--model", default="gpt2", help="Model name or path")
    parser.add_argument("--task", default="Sem", help="Task name")
    parser.add_argument("--layer", type=int, default=-1, help="Which layer to extract")
    parser.add_argument("--pooling", default="mean", choices=["mean", "max", "cls"])
    parser.add_argument("--output-dir", default="data/processed/language_models")
    
    args = parser.parse_args()
    
    # Initialize extractor and RDM computer
    extractor = LanguageModelEmbeddingExtractor(
        model_name=args.model,
        layer=args.layer,
        pooling=args.pooling
    )
    
    rdm_computer = LanguageModelRDMComputer(output_dir=args.output_dir)
    
    # Load stimulus characteristics
    characteristics, words, word_to_idx = rdm_computer.load_stimulus_characteristics(
        task=args.task
    )
    
    # Extract embeddings
    logger.info(f"Extracting embeddings for {len(words)} words")
    embeddings = extractor.extract_batch_embeddings(words)
    logger.info(f"Embedding shape: {embeddings.shape}")
    
    # Compute RDM
    logger.info("Computing RDM from stimulus characteristics")
    rdm = rdm_computer.compute_stimulus_rdm(
        characteristics, embeddings, words, word_to_idx
    )
    
    logger.info(f"RDM shape: {rdm.shape}")
    logger.info(f"RDM mean dissimilarity: {rdm[np.triu_indices_from(rdm, k=1)].mean():.4f}")
    
    # Save RDM
    output_file = f"lm_rdm_{args.model.split('/')[-1]}_{args.task}.npz"
    rdm_computer.save_rdm(rdm, output_file)


if __name__ == "__main__":
    main()
