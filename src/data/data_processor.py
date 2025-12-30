"""
Data Processor Module for DAIC-WOZ Transcripts

This module handles loading, preprocessing, and preparing DAIC-WOZ interview transcripts
for depression detection modeling.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import re
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DAICWOZDataProcessor:
    """
    Processes DAIC-WOZ interview transcripts for depression detection.
    
    The DAIC-WOZ dataset contains clinical interviews designed to support
    the diagnosis of psychological distress conditions such as anxiety,
    depression, and post-traumatic stress disorder.
    """
    
    def __init__(self, data_dir: str):
        """
        Initialize the data processor.
        
        Args:
            data_dir: Path to the directory containing DAIC-WOZ data
        """
        self.data_dir = Path(data_dir)
        self.transcripts = []
        self.labels = []
        self.metadata = []
        
    def load_transcripts(self, transcript_file: str) -> pd.DataFrame:
        """
        Load transcripts from a file.
        
        Args:
            transcript_file: Path to the transcript file
            
        Returns:
            DataFrame containing transcripts and associated data
        """
        logger.info(f"Loading transcripts from {transcript_file}")
        
        try:
            df = pd.read_csv(transcript_file)
            logger.info(f"Loaded {len(df)} transcripts")
            return df
        except Exception as e:
            logger.error(f"Error loading transcripts: {e}")
            raise
    
    def clean_text(self, text: str) -> str:
        """
        Clean and preprocess transcript text.
        
        Args:
            text: Raw transcript text
            
        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep punctuation that might be meaningful
        text = re.sub(r'[^\w\s\.\,\!\?\-\']', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def extract_participant_responses(self, transcript: str) -> str:
        """
        Extract only participant responses from the full transcript.
        
        Args:
            transcript: Full interview transcript
            
        Returns:
            Participant responses only
        """
        # Pattern to identify participant vs interviewer
        # Assumes format like "Participant: text" or "Ellie: text"
        lines = transcript.split('\n')
        participant_lines = []
        
        for line in lines:
            # Skip interviewer lines (commonly "Ellie" or "Interviewer")
            if not line.strip():
                continue
            if 'ellie' in line.lower()[:20] or 'interviewer' in line.lower()[:20]:
                continue
            if 'participant' in line.lower()[:20]:
                # Remove the "Participant:" prefix
                line = re.sub(r'^participant\s*:\s*', '', line, flags=re.IGNORECASE)
            participant_lines.append(line)
        
        return ' '.join(participant_lines)
    
    def create_dataset(
        self, 
        transcripts: List[str], 
        labels: List[int],
        ids: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Create a structured dataset from transcripts and labels.
        
        Args:
            transcripts: List of transcript texts
            labels: List of depression labels (0: no depression, 1: depression)
            ids: Optional list of participant IDs
            
        Returns:
            DataFrame with processed data
        """
        if ids is None:
            ids = [f"participant_{i}" for i in range(len(transcripts))]
        
        data = []
        for i, (transcript, label) in enumerate(zip(transcripts, labels)):
            cleaned_text = self.clean_text(transcript)
            participant_text = self.extract_participant_responses(transcript)
            
            data.append({
                'id': ids[i],
                'transcript': transcript,
                'cleaned_transcript': cleaned_text,
                'participant_text': self.clean_text(participant_text),
                'label': label,
                'text_length': len(cleaned_text),
                'word_count': len(cleaned_text.split())
            })
        
        df = pd.DataFrame(data)
        logger.info(f"Created dataset with {len(df)} samples")
        logger.info(f"Label distribution: {df['label'].value_counts().to_dict()}")
        
        return df
    
    def split_dataset(
        self, 
        df: pd.DataFrame, 
        train_size: float = 0.7,
        val_size: float = 0.15,
        test_size: float = 0.15,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split dataset into train, validation, and test sets.
        
        Args:
            df: DataFrame to split
            train_size: Proportion for training set
            val_size: Proportion for validation set
            test_size: Proportion for test set
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        from sklearn.model_selection import train_test_split
        
        assert abs(train_size + val_size + test_size - 1.0) < 1e-5, \
            "Split sizes must sum to 1.0"
        
        # First split: separate test set
        train_val_df, test_df = train_test_split(
            df, 
            test_size=test_size, 
            random_state=random_state,
            stratify=df['label']
        )
        
        # Second split: separate train and validation
        val_ratio = val_size / (train_size + val_size)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_ratio,
            random_state=random_state,
            stratify=train_val_df['label']
        )
        
        logger.info(f"Train set: {len(train_df)} samples")
        logger.info(f"Validation set: {len(val_df)} samples")
        logger.info(f"Test set: {len(test_df)} samples")
        
        return train_df, val_df, test_df
    
    def save_processed_data(
        self, 
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        output_dir: str
    ):
        """
        Save processed datasets to files.
        
        Args:
            train_df: Training DataFrame
            val_df: Validation DataFrame
            test_df: Test DataFrame
            output_dir: Directory to save files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        train_df.to_csv(output_path / "train.csv", index=False)
        val_df.to_csv(output_path / "val.csv", index=False)
        test_df.to_csv(output_path / "test.csv", index=False)
        
        logger.info(f"Saved processed data to {output_dir}")


def load_sample_data() -> Tuple[List[str], List[int]]:
    """
    Create sample data for demonstration purposes.
    
    Returns:
        Tuple of (transcripts, labels)
    """
    # Sample transcripts representing different depression indicators
    transcripts = [
        "Participant: I've been feeling really down lately. I can't seem to enjoy anything anymore. "
        "Even things I used to love don't interest me. I have trouble sleeping and I'm tired all the time.",
        
        "Participant: Life is going pretty well. I'm excited about my new job and I've been spending "
        "time with friends. I feel energetic and motivated most days.",
        
        "Participant: Everything feels overwhelming. I don't have the energy to do basic things. "
        "I feel hopeless about the future and I've been withdrawing from people.",
        
        "Participant: I'm doing okay. Some days are better than others, but overall I'm managing well. "
        "I exercise regularly and that helps my mood.",
        
        "Participant: I can't concentrate on anything. My mind feels foggy. I've lost my appetite "
        "and I don't want to see anyone. Nothing seems worth it anymore.",
        
        "Participant: I'm looking forward to the weekend. My family is visiting and we're planning "
        "a hiking trip. I've been feeling positive about things lately.",
        
        "Participant: I feel empty inside. I've been having dark thoughts and I don't see a way out. "
        "I can't remember the last time I felt happy.",
        
        "Participant: Work is challenging but in a good way. I'm learning new skills and my "
        "relationships are strong. I feel content with where I am in life."
    ]
    
    # Labels: 1 = depression indicators, 0 = no depression
    labels = [1, 0, 1, 0, 1, 0, 1, 0]
    
    return transcripts, labels
