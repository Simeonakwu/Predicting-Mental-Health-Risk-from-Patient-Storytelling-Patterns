"""
Evaluation Metrics Module

Provides comprehensive evaluation metrics for the depression detection model.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve
)
from typing import Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns


class ModelEvaluator:
    """
    Comprehensive model evaluation with various metrics.
    """
    
    @staticmethod
    def calculate_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray = None
    ) -> Dict[str, float]:
        """
        Calculate comprehensive evaluation metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities (optional)
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'specificity': ModelEvaluator._calculate_specificity(y_true, y_pred)
        }
        
        # Add AUC if probabilities are provided
        if y_proba is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_proba[:, 1])
            except (IndexError, ValueError):
                metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
        
        return metrics
    
    @staticmethod
    def _calculate_specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate specificity (true negative rate).
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Specificity score
        """
        cm = confusion_matrix(y_true, y_pred)
        
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        else:
            specificity = 0.0
        
        return specificity
    
    @staticmethod
    def print_classification_report(y_true: np.ndarray, y_pred: np.ndarray):
        """
        Print detailed classification report.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
        """
        print("\nClassification Report:")
        print("=" * 60)
        print(classification_report(
            y_true, 
            y_pred,
            target_names=['No Depression', 'Depression']
        ))
    
    @staticmethod
    def plot_confusion_matrix(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        save_path: str = None
    ):
        """
        Plot confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            save_path: Optional path to save the plot
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['No Depression', 'Depression'],
            yticklabels=['No Depression', 'Depression']
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        else:
            plt.show()
        
        plt.close()
    
    @staticmethod
    def plot_roc_curve(
        y_true: np.ndarray,
        y_proba: np.ndarray,
        save_path: str = None
    ):
        """
        Plot ROC curve.
        
        Args:
            y_true: True labels
            y_proba: Prediction probabilities
            save_path: Optional path to save the plot
        """
        # Handle different probability formats
        if len(y_proba.shape) > 1 and y_proba.shape[1] > 1:
            y_proba = y_proba[:, 1]
        
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        else:
            plt.show()
        
        plt.close()
    
    @staticmethod
    def plot_precision_recall_curve(
        y_true: np.ndarray,
        y_proba: np.ndarray,
        save_path: str = None
    ):
        """
        Plot precision-recall curve.
        
        Args:
            y_true: True labels
            y_proba: Prediction probabilities
            save_path: Optional path to save the plot
        """
        # Handle different probability formats
        if len(y_proba.shape) > 1 and y_proba.shape[1] > 1:
            y_proba = y_proba[:, 1]
        
        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.grid(alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        else:
            plt.show()
        
        plt.close()
    
    @staticmethod
    def create_evaluation_report(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray = None,
        output_dir: str = None
    ):
        """
        Create comprehensive evaluation report with all visualizations.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities (optional)
            output_dir: Directory to save outputs (optional)
        """
        # Calculate metrics
        metrics = ModelEvaluator.calculate_metrics(y_true, y_pred, y_proba)
        
        # Print metrics
        print("\nEvaluation Metrics:")
        print("=" * 60)
        for metric, value in metrics.items():
            print(f"{metric:20s}: {value:.4f}")
        
        # Print classification report
        ModelEvaluator.print_classification_report(y_true, y_pred)
        
        # Create plots
        if output_dir:
            from pathlib import Path
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            ModelEvaluator.plot_confusion_matrix(
                y_true, y_pred,
                save_path=str(output_path / 'confusion_matrix.png')
            )
            
            if y_proba is not None:
                ModelEvaluator.plot_roc_curve(
                    y_true, y_proba,
                    save_path=str(output_path / 'roc_curve.png')
                )
                
                ModelEvaluator.plot_precision_recall_curve(
                    y_true, y_proba,
                    save_path=str(output_path / 'precision_recall_curve.png')
                )
        else:
            ModelEvaluator.plot_confusion_matrix(y_true, y_pred)
            
            if y_proba is not None:
                ModelEvaluator.plot_roc_curve(y_true, y_proba)
                ModelEvaluator.plot_precision_recall_curve(y_true, y_proba)
