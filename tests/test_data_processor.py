"""
Unit tests for the data processor module.
"""

import pytest
import numpy as np
import pandas as pd
from src.data.data_processor import DAICWOZDataProcessor, load_sample_data


class TestDAICWOZDataProcessor:
    """Test cases for DAICWOZDataProcessor class."""
    
    def test_initialization(self):
        """Test processor initialization."""
        processor = DAICWOZDataProcessor("data")
        assert processor.data_dir.name == "data"
    
    def test_clean_text(self):
        """Test text cleaning."""
        processor = DAICWOZDataProcessor("data")
        
        # Test basic cleaning
        text = "Hello   World!  "
        cleaned = processor.clean_text(text)
        assert cleaned == "hello world!"
        
        # Test with None
        assert processor.clean_text(None) == ""
        
        # Test with special characters
        text = "Test@#$%Text"
        cleaned = processor.clean_text(text)
        assert "@#$%" not in cleaned
    
    def test_create_dataset(self):
        """Test dataset creation."""
        processor = DAICWOZDataProcessor("data")
        
        transcripts = ["Test transcript 1", "Test transcript 2"]
        labels = [0, 1]
        
        df = processor.create_dataset(transcripts, labels)
        
        assert len(df) == 2
        assert 'label' in df.columns
        assert 'cleaned_transcript' in df.columns
        assert 'word_count' in df.columns
    
    def test_split_dataset(self):
        """Test dataset splitting."""
        processor = DAICWOZDataProcessor("data")
        
        # Create sample dataset
        data = {
            'id': [f'p{i}' for i in range(100)],
            'transcript': ['text'] * 100,
            'cleaned_transcript': ['text'] * 100,
            'label': [0, 1] * 50,
            'text_length': [100] * 100,
            'word_count': [20] * 100
        }
        df = pd.DataFrame(data)
        
        # Split
        train_df, val_df, test_df = processor.split_dataset(
            df,
            train_size=0.7,
            val_size=0.15,
            test_size=0.15
        )
        
        # Check sizes
        assert len(train_df) == 70
        assert len(val_df) == 15
        assert len(test_df) == 15
        
        # Check no overlap
        train_ids = set(train_df['id'])
        val_ids = set(val_df['id'])
        test_ids = set(test_df['id'])
        
        assert len(train_ids & val_ids) == 0
        assert len(train_ids & test_ids) == 0
        assert len(val_ids & test_ids) == 0


def test_load_sample_data():
    """Test loading sample data."""
    transcripts, labels = load_sample_data()
    
    assert len(transcripts) == len(labels)
    assert len(transcripts) > 0
    assert all(isinstance(t, str) for t in transcripts)
    assert all(l in [0, 1] for l in labels)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
