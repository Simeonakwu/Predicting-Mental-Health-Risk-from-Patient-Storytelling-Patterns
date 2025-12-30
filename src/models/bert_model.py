"""
BERT-based Depression Detection Model

This module implements a BERT-based classifier for detecting depression
from interview transcripts.
"""

import torch
import torch.nn as nn
from transformers import (
    BertModel, 
    BertTokenizer, 
    BertForSequenceClassification,
    AutoTokenizer,
    AutoModel,
    Trainer,
    TrainingArguments
)
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TranscriptDataset(Dataset):
    """
    PyTorch Dataset for interview transcripts.
    """
    
    def __init__(
        self, 
        texts: List[str], 
        labels: List[int],
        tokenizer,
        max_length: int = 512
    ):
        """
        Initialize the dataset.
        
        Args:
            texts: List of transcript texts
            labels: List of labels (0 or 1)
            tokenizer: BERT tokenizer
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class BERTDepressionClassifier(nn.Module):
    """
    BERT-based binary classifier for depression detection.
    """
    
    def __init__(
        self, 
        model_name: str = 'bert-base-uncased',
        num_classes: int = 2,
        dropout: float = 0.3
    ):
        """
        Initialize the classifier.
        
        Args:
            model_name: Name of the pre-trained BERT model
            num_classes: Number of output classes (2 for binary classification)
            dropout: Dropout rate
        """
        super(BERTDepressionClassifier, self).__init__()
        
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        
    def forward(
        self, 
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            
        Returns:
            Logits for each class
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return logits


class DepressionDetectionModel:
    """
    High-level wrapper for BERT-based depression detection.
    """
    
    def __init__(
        self,
        model_name: str = 'bert-base-uncased',
        max_length: int = 512,
        device: Optional[str] = None
    ):
        """
        Initialize the model.
        
        Args:
            model_name: Name of pre-trained BERT model
            max_length: Maximum sequence length
            device: Device to use (cuda/cpu)
        """
        self.model_name = model_name
        self.max_length = max_length
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f"Using device: {self.device}")
        
        # Initialize tokenizer and model
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BERTDepressionClassifier(model_name=model_name)
        self.model.to(self.device)
        
    def prepare_data(
        self,
        texts: List[str],
        labels: Optional[List[int]] = None,
        batch_size: int = 16
    ) -> DataLoader:
        """
        Prepare data for training or inference.
        
        Args:
            texts: List of transcript texts
            labels: Optional list of labels
            batch_size: Batch size
            
        Returns:
            DataLoader
        """
        if labels is None:
            labels = [0] * len(texts)  # Dummy labels for inference
        
        dataset = TranscriptDataset(
            texts=texts,
            labels=labels,
            tokenizer=self.tokenizer,
            max_length=self.max_length
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(labels is not None)
        )
        
        return dataloader
    
    def train_step(
        self,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module
    ) -> float:
        """
        Perform one training epoch.
        
        Args:
            dataloader: Training data loader
            optimizer: Optimizer
            criterion: Loss function
            
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0
        
        for batch in dataloader:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            optimizer.zero_grad()
            
            logits = self.model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        return avg_loss
    
    def evaluate(
        self,
        dataloader: DataLoader,
        criterion: nn.Module
    ) -> Tuple[float, float]:
        """
        Evaluate the model.
        
        Args:
            dataloader: Validation/test data loader
            criterion: Loss function
            
        Returns:
            Tuple of (loss, accuracy)
        """
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                logits = self.model(input_ids, attention_mask)
                loss = criterion(logits, labels)
                
                total_loss += loss.item()
                
                predictions = torch.argmax(logits, dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
        
        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def predict(
        self,
        texts: List[str],
        batch_size: int = 16
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on new texts.
        
        Args:
            texts: List of transcript texts
            batch_size: Batch size
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        self.model.eval()
        dataloader = self.prepare_data(texts, batch_size=batch_size)
        
        all_predictions = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                logits = self.model(input_ids, attention_mask)
                probabilities = torch.softmax(logits, dim=1)
                predictions = torch.argmax(logits, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        return np.array(all_predictions), np.array(all_probabilities)
    
    def save_model(self, save_path: str):
        """
        Save the model to disk.
        
        Args:
            save_path: Path to save the model
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save model weights
        torch.save(self.model.state_dict(), save_path / 'model.pth')
        
        # Save tokenizer
        self.tokenizer.save_pretrained(save_path / 'tokenizer')
        
        logger.info(f"Model saved to {save_path}")
    
    def load_model(self, load_path: str):
        """
        Load a saved model from disk.
        
        Args:
            load_path: Path to load the model from
        """
        load_path = Path(load_path)
        
        # Load model weights
        self.model.load_state_dict(
            torch.load(load_path / 'model.pth', map_location=self.device)
        )
        
        # Load tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(load_path / 'tokenizer')
        
        logger.info(f"Model loaded from {load_path}")
