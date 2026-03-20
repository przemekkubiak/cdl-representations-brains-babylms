"""
Speech recognition pipeline for converting stimulus .wav files to text.
Uses OpenAI Whisper for robust speech-to-text transcription.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

import whisper
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SpeechRecognitionPipeline:
    """
    Transcribes stimulus .wav files and maps them to stimulus characteristics.
    
    The semantic task uses word pairs (word_A and word_B) presented as stereo audio.
    This pipeline extracts transcriptions and validates them against the TSV characteristics.
    """
    
    def __init__(
        self,
        stimulus_dir: str = "data/brain/ds003604/stimuli",
        characteristics_dir: str = "data/brain/ds003604/stimuli/Stimulus_Characteristics",
        output_dir: str = "data/processed/language_models",
        model_size: str = "base"  # tiny, base, small, medium, large
    ):
        """
        Initialize speech recognition pipeline.
        
        Args:
            stimulus_dir: Root directory containing task stimulus folders
            characteristics_dir: Directory with stimulus characteristic TSV files
            output_dir: Output directory for transcriptions
            model_size: Whisper model size to use
        """
        self.stimulus_dir = Path(stimulus_dir)
        self.characteristics_dir = Path(characteristics_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load Whisper model
        logger.info(f"Loading Whisper model: {model_size}")
        self.model = whisper.load_model(model_size)
        
        self.transcriptions = {}  # Cache for transcriptions
        
    def transcribe_wav(self, wav_path: str, language: str = "en") -> str:
        """
        Transcribe a single .wav file using Whisper.
        
        Args:
            wav_path: Path to .wav file
            language: Language code (default: English)
            
        Returns:
            Transcribed text
        """
        if wav_path in self.transcriptions:
            return self.transcriptions[wav_path]
        
        logger.info(f"Transcribing: {wav_path}")
        result = self.model.transcribe(str(wav_path), language=language)
        text = result["text"].strip()
        
        self.transcriptions[wav_path] = text
        return text
    
    def process_task(self, task_name: str = "Sem") -> Dict:
        """
        Process all stimulus files for a given task (Sem, Gram, Phon, Plaus).
        
        Args:
            task_name: Task name (Sem, Gram, Phon, Plaus)
            
        Returns:
            Dictionary with transcriptions and metadata
        """
        logger.info(f"Processing task: {task_name}")
        
        # Load stimulus characteristics
        char_file = self.characteristics_dir / f"task-{task_name}_Stimulus_Characteristics.tsv"
        characteristics = pd.read_csv(char_file, sep="\t")
        
        task_results = {
            "task": task_name,
            "stimuli": [],
            "characteristics": characteristics.to_dict(orient="records")
        }
        
        # Process each run
        for run_num in [1, 2]:
            run_dir = self.stimulus_dir / f"{task_name}" / f"{task_name}_run-{run_num:02d}"
            
            if not run_dir.exists():
                logger.warning(f"Run directory not found: {run_dir}")
                continue
            
            logger.info(f"Processing run {run_num}")
            
            # Find all .wav files
            wav_files = sorted(run_dir.glob("*.wav"))
            logger.info(f"Found {len(wav_files)} .wav files in {run_dir}")
            
            for wav_file in wav_files:
                # Get characteristics for this stimulus
                stim_filename = wav_file.name
                char_row = characteristics[characteristics["stim_file"] == stim_filename]
                
                if len(char_row) == 0:
                    logger.warning(f"No characteristics found for {stim_filename}")
                    continue
                
                char_row = char_row.iloc[0]
                
                # Transcribe
                transcription = self.transcribe_wav(str(wav_file))
                
                # Expected words
                expected_words = [
                    char_row.get("word_A"),
                    char_row.get("word_B")
                ]
                
                stimulus_info = {
                    "run": run_num,
                    "filename": stim_filename,
                    "transcription": transcription,
                    "expected_words": expected_words,
                    "trial_type": char_row.get("trial_type"),
                    "word_A": char_row.get("word_A"),
                    "word_B": char_row.get("word_B"),
                }
                
                task_results["stimuli"].append(stimulus_info)
        
        return task_results
    
    def process_all_tasks(self, tasks: List[str] = None) -> Dict:
        """
        Process all tasks and save transcriptions.
        
        Args:
            tasks: List of task names to process (default: ["Sem"])
            
        Returns:
            Dictionary with results for all tasks
        """
        if tasks is None:
            tasks = ["Sem"]
        
        all_results = {}
        
        for task in tasks:
            results = self.process_task(task)
            all_results[task] = results
            
            # Save transcriptions
            output_file = self.output_dir / f"transcriptions_{task}.json"
            with open(output_file, "w") as f:
                json.dump(results, f, indent=2)
            logger.info(f"Saved transcriptions to {output_file}")
        
        return all_results
    
    def extract_word_list(self, task: str = "Sem") -> Tuple[List[str], Dict]:
        """
        Extract unique words and their indices from a task.
        
        Args:
            task: Task name
            
        Returns:
            Tuple of (word_list, word_to_idx mapping)
        """
        char_file = self.characteristics_dir / f"task-{task}_Stimulus_Characteristics.tsv"
        characteristics = pd.read_csv(char_file, sep="\t")
        
        # Collect all words
        all_words = []
        for idx, row in characteristics.iterrows():
            word_a = row.get("word_A")
            word_b = row.get("word_B")
            if pd.notna(word_a):
                all_words.append(word_a)
            if pd.notna(word_b):
                all_words.append(word_b)
        
        # Get unique words in order they appear
        unique_words = []
        seen = set()
        for word in all_words:
            word_lower = str(word).lower()
            if word_lower not in seen:
                unique_words.append(word_lower)
                seen.add(word_lower)
        
        word_to_idx = {word: idx for idx, word in enumerate(unique_words)}
        
        logger.info(f"Task {task}: {len(unique_words)} unique words")
        
        return unique_words, word_to_idx


def main():
    """Test speech recognition pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Transcribe stimulus files using Whisper")
    parser.add_argument("--stimulus-dir", default="data/brain/ds003604/stimuli")
    parser.add_argument("--output-dir", default="data/processed/language_models")
    parser.add_argument("--model-size", default="base", choices=["tiny", "base", "small", "medium", "large"])
    parser.add_argument("--task", default="Sem", help="Task to process (Sem, Gram, Phon, Plaus)")
    
    args = parser.parse_args()
    
    pipeline = SpeechRecognitionPipeline(
        stimulus_dir=args.stimulus_dir,
        output_dir=args.output_dir,
        model_size=args.model_size
    )
    
    # Extract word list first
    words, word_to_idx = pipeline.extract_word_list(args.task)
    print(f"Unique words: {words[:10]}...")
    print(f"Total words: {len(words)}")
    
    # Process task
    results = pipeline.process_task(args.task)
    print(f"\nProcessed {len(results['stimuli'])} stimuli")
    
    # Print some examples
    if results["stimuli"]:
        print(f"\nExample transcriptions:")
        for stim in results["stimuli"][:3]:
            print(f"  {stim['filename']}: {stim['transcription']}")
            print(f"    Expected: {stim['expected_words']}")


if __name__ == "__main__":
    main()
