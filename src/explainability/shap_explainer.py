"""
SHAP Explainability Module

This module provides interpretability for the BERT-based depression detection model
using SHAP (SHapley Additive exPlanations) values.
"""

import shap
import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Callable
import matplotlib.pyplot as plt
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelExplainer:
    """
    Provides SHAP-based explanations for the depression detection model.
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        device: Optional[str] = None
    ):
        """
        Initialize the explainer.
        
        Args:
            model: Trained BERT model
            tokenizer: BERT tokenizer
            device: Device to use (cuda/cpu)
        """
        self.model = model
        self.tokenizer = tokenizer
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.explainer = None
        self.model.eval()
        
    def _predict_wrapper(self, texts: List[str]) -> np.ndarray:
        """
        Wrapper function for model predictions (required by SHAP).
        
        Args:
            texts: List of input texts
            
        Returns:
            Prediction probabilities
        """
        predictions = []
        
        with torch.no_grad():
            for text in texts:
                encoding = self.tokenizer(
                    text,
                    add_special_tokens=True,
                    max_length=512,
                    padding='max_length',
                    truncation=True,
                    return_attention_mask=True,
                    return_tensors='pt'
                )
                
                input_ids = encoding['input_ids'].to(self.device)
                attention_mask = encoding['attention_mask'].to(self.device)
                
                logits = self.model(input_ids, attention_mask)
                probabilities = torch.softmax(logits, dim=1)
                predictions.append(probabilities.cpu().numpy())
        
        return np.vstack(predictions)
    
    def initialize_explainer(
        self,
        background_texts: List[str],
        algorithm: str = 'partition'
    ):
        """
        Initialize the SHAP explainer with background data.
        
        Args:
            background_texts: Sample texts to use as background
            algorithm: SHAP algorithm ('partition' or 'kernel')
        """
        logger.info(f"Initializing SHAP explainer with {len(background_texts)} background samples")
        
        if algorithm == 'partition':
            # Partition explainer for text models
            self.explainer = shap.Explainer(
                self._predict_wrapper,
                masker=shap.maskers.Text(tokenizer=r"\W+"),
                algorithm='partition'
            )
        else:
            # Kernel explainer (more general but slower)
            self.explainer = shap.KernelExplainer(
                self._predict_wrapper,
                background_texts
            )
        
        logger.info("SHAP explainer initialized")
    
    def explain_prediction(
        self,
        text: str,
        show_plot: bool = True
    ) -> shap.Explanation:
        """
        Generate SHAP explanation for a single prediction.
        
        Args:
            text: Input text to explain
            show_plot: Whether to display the explanation plot
            
        Returns:
            SHAP explanation object
        """
        if self.explainer is None:
            raise ValueError("Explainer not initialized. Call initialize_explainer() first.")
        
        logger.info("Generating SHAP explanation")
        
        # Get SHAP values
        shap_values = self.explainer([text])
        
        if show_plot:
            # Visualize the explanation
            shap.plots.text(shap_values[0, :, 1])  # Show explanation for positive class
        
        return shap_values
    
    def explain_batch(
        self,
        texts: List[str],
        max_samples: int = 100
    ) -> shap.Explanation:
        """
        Generate SHAP explanations for a batch of texts.
        
        Args:
            texts: List of texts to explain
            max_samples: Maximum number of samples to explain
            
        Returns:
            SHAP explanation object
        """
        if self.explainer is None:
            raise ValueError("Explainer not initialized. Call initialize_explainer() first.")
        
        texts_subset = texts[:max_samples]
        logger.info(f"Generating SHAP explanations for {len(texts_subset)} samples")
        
        shap_values = self.explainer(texts_subset)
        
        return shap_values
    
    def plot_summary(
        self,
        shap_values: shap.Explanation,
        save_path: Optional[str] = None
    ):
        """
        Create a summary plot of SHAP values.
        
        Args:
            shap_values: SHAP explanation object
            save_path: Optional path to save the plot
        """
        plt.figure(figsize=(10, 6))
        
        # Summary plot
        shap.summary_plot(
            shap_values.values[:, :, 1],  # Positive class
            shap_values.data,
            show=False
        )
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            logger.info(f"Summary plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def get_feature_importance(
        self,
        shap_values: shap.Explanation,
        top_k: int = 20
    ) -> pd.DataFrame:
        """
        Extract top important features from SHAP values.
        
        Args:
            shap_values: SHAP explanation object
            top_k: Number of top features to return
            
        Returns:
            DataFrame with feature importance scores
        """
        # Calculate mean absolute SHAP values
        mean_shap = np.abs(shap_values.values[:, :, 1]).mean(axis=0)
        
        # Get feature names (words/tokens)
        if hasattr(shap_values, 'data'):
            if isinstance(shap_values.data[0], str):
                # Text data
                words = []
                for text in shap_values.data:
                    words.extend(text.split())
                feature_names = list(set(words))[:len(mean_shap)]
            else:
                feature_names = [f"feature_{i}" for i in range(len(mean_shap))]
        else:
            feature_names = [f"feature_{i}" for i in range(len(mean_shap))]
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names[:len(mean_shap)],
            'importance': mean_shap
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        return importance_df.head(top_k)
    
    def explain_with_lime(
        self,
        text: str,
        num_features: int = 10
    ) -> Dict:
        """
        Alternative explanation using LIME (Local Interpretable Model-agnostic Explanations).
        
        Args:
            text: Input text to explain
            num_features: Number of features to include in explanation
            
        Returns:
            Dictionary with explanation results
        """
        try:
            from lime.lime_text import LimeTextExplainer
            
            # Initialize LIME explainer
            lime_explainer = LimeTextExplainer(class_names=['No Depression', 'Depression'])
            
            # Generate explanation
            explanation = lime_explainer.explain_instance(
                text,
                self._predict_wrapper,
                num_features=num_features
            )
            
            # Extract top features
            features = explanation.as_list()
            
            return {
                'features': features,
                'prediction': explanation.predict_proba[1],
                'local_explanation': explanation
            }
            
        except ImportError:
            logger.warning("LIME not installed. Install with: pip install lime")
            return {}


class InterpretabilityReport:
    """
    Generate comprehensive interpretability reports.
    """
    
    @staticmethod
    def create_report(
        texts: List[str],
        predictions: np.ndarray,
        shap_values: shap.Explanation,
        output_path: str
    ):
        """
        Create a comprehensive interpretability report.
        
        Args:
            texts: Input texts
            predictions: Model predictions
            shap_values: SHAP explanations
            output_path: Path to save the report
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create report data
        report_data = []
        
        for i, (text, pred) in enumerate(zip(texts, predictions)):
            report_data.append({
                'text': text[:100] + '...' if len(text) > 100 else text,
                'prediction': 'Depression' if pred == 1 else 'No Depression',
                'confidence': float(pred)
            })
        
        # Save as CSV
        df = pd.DataFrame(report_data)
        df.to_csv(output_path / 'interpretability_report.csv', index=False)
        
        logger.info(f"Interpretability report saved to {output_path}")
    
    @staticmethod
    def plot_word_importance(
        text: str,
        word_scores: Dict[str, float],
        save_path: Optional[str] = None
    ):
        """
        Plot word-level importance scores.
        
        Args:
            text: Original text
            word_scores: Dictionary mapping words to importance scores
            save_path: Optional path to save the plot
        """
        words = list(word_scores.keys())
        scores = list(word_scores.values())
        
        plt.figure(figsize=(12, 6))
        colors = ['red' if s > 0 else 'blue' for s in scores]
        plt.barh(words, scores, color=colors, alpha=0.7)
        plt.xlabel('SHAP Value')
        plt.ylabel('Words')
        plt.title('Word-level Feature Importance')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        else:
            plt.show()
        
        plt.close()
