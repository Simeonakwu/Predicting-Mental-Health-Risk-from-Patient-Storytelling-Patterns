"""
Fairness Auditing Module

This module provides tools for auditing the fairness of the depression detection model
across different demographic groups.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score,
    f1_score,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FairnessMetrics:
    """
    Calculate various fairness metrics for model evaluation.
    """
    
    @staticmethod
    def demographic_parity_difference(
        y_pred: np.ndarray,
        sensitive_attr: np.ndarray
    ) -> float:
        """
        Calculate demographic parity difference.
        
        Measures the difference in positive prediction rates between groups.
        A value close to 0 indicates fairness.
        
        Args:
            y_pred: Model predictions
            sensitive_attr: Sensitive attribute (e.g., gender, age group)
            
        Returns:
            Demographic parity difference
        """
        groups = np.unique(sensitive_attr)
        
        if len(groups) != 2:
            logger.warning("Demographic parity is designed for binary sensitive attributes")
        
        rates = []
        for group in groups:
            mask = sensitive_attr == group
            if mask.sum() > 0:
                rate = y_pred[mask].mean()
                rates.append(rate)
        
        return abs(rates[0] - rates[1]) if len(rates) == 2 else 0.0
    
    @staticmethod
    def equal_opportunity_difference(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive_attr: np.ndarray
    ) -> float:
        """
        Calculate equal opportunity difference.
        
        Measures the difference in true positive rates between groups.
        
        Args:
            y_true: True labels
            y_pred: Model predictions
            sensitive_attr: Sensitive attribute
            
        Returns:
            Equal opportunity difference
        """
        groups = np.unique(sensitive_attr)
        tpr_list = []
        
        for group in groups:
            mask = sensitive_attr == group
            if mask.sum() > 0:
                y_true_group = y_true[mask]
                y_pred_group = y_pred[mask]
                
                # True Positive Rate (Recall for positive class)
                if (y_true_group == 1).sum() > 0:
                    tpr = recall_score(y_true_group, y_pred_group, zero_division=0)
                    tpr_list.append(tpr)
        
        return abs(tpr_list[0] - tpr_list[1]) if len(tpr_list) == 2 else 0.0
    
    @staticmethod
    def equalized_odds_difference(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive_attr: np.ndarray
    ) -> Tuple[float, float]:
        """
        Calculate equalized odds differences.
        
        Returns both TPR and FPR differences between groups.
        
        Args:
            y_true: True labels
            y_pred: Model predictions
            sensitive_attr: Sensitive attribute
            
        Returns:
            Tuple of (TPR difference, FPR difference)
        """
        groups = np.unique(sensitive_attr)
        tpr_list = []
        fpr_list = []
        
        for group in groups:
            mask = sensitive_attr == group
            if mask.sum() > 0:
                y_true_group = y_true[mask]
                y_pred_group = y_pred[mask]
                
                cm = confusion_matrix(y_true_group, y_pred_group)
                
                # Handle edge cases
                if cm.shape[0] == 2 and cm.shape[1] == 2:
                    tn, fp, fn, tp = cm.ravel()
                    
                    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
                    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                    
                    tpr_list.append(tpr)
                    fpr_list.append(fpr)
        
        tpr_diff = abs(tpr_list[0] - tpr_list[1]) if len(tpr_list) == 2 else 0.0
        fpr_diff = abs(fpr_list[0] - fpr_list[1]) if len(fpr_list) == 2 else 0.0
        
        return tpr_diff, fpr_diff


class FairnessAuditor:
    """
    Comprehensive fairness auditing for the depression detection model.
    """
    
    def __init__(self):
        """Initialize the fairness auditor."""
        self.metrics = FairnessMetrics()
        self.audit_results = {}
    
    def audit_model(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive_attributes: Dict[str, np.ndarray],
        threshold: float = 0.1
    ) -> Dict:
        """
        Perform comprehensive fairness audit.
        
        Args:
            y_true: True labels
            y_pred: Model predictions
            sensitive_attributes: Dictionary of sensitive attributes
            threshold: Threshold for fairness (values below are considered fair)
            
        Returns:
            Dictionary containing audit results
        """
        logger.info("Starting fairness audit")
        
        results = {
            'overall_metrics': self._calculate_overall_metrics(y_true, y_pred),
            'group_metrics': {},
            'fairness_metrics': {},
            'fairness_assessment': {}
        }
        
        # Audit each sensitive attribute
        for attr_name, attr_values in sensitive_attributes.items():
            logger.info(f"Auditing {attr_name}")
            
            # Group-specific metrics
            results['group_metrics'][attr_name] = self._calculate_group_metrics(
                y_true, y_pred, attr_values
            )
            
            # Fairness metrics
            dp_diff = self.metrics.demographic_parity_difference(y_pred, attr_values)
            eo_diff = self.metrics.equal_opportunity_difference(y_true, y_pred, attr_values)
            tpr_diff, fpr_diff = self.metrics.equalized_odds_difference(
                y_true, y_pred, attr_values
            )
            
            results['fairness_metrics'][attr_name] = {
                'demographic_parity_difference': dp_diff,
                'equal_opportunity_difference': eo_diff,
                'tpr_difference': tpr_diff,
                'fpr_difference': fpr_diff
            }
            
            # Assess fairness
            results['fairness_assessment'][attr_name] = {
                'demographic_parity': 'PASS' if dp_diff < threshold else 'FAIL',
                'equal_opportunity': 'PASS' if eo_diff < threshold else 'FAIL',
                'equalized_odds': 'PASS' if max(tpr_diff, fpr_diff) < threshold else 'FAIL'
            }
        
        self.audit_results = results
        logger.info("Fairness audit complete")
        
        return results
    
    def _calculate_overall_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict:
        """
        Calculate overall model performance metrics.
        
        Args:
            y_true: True labels
            y_pred: Model predictions
            
        Returns:
            Dictionary of metrics
        """
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0)
        }
    
    def _calculate_group_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive_attr: np.ndarray
    ) -> Dict:
        """
        Calculate metrics for each group.
        
        Args:
            y_true: True labels
            y_pred: Model predictions
            sensitive_attr: Sensitive attribute values
            
        Returns:
            Dictionary of group-specific metrics
        """
        groups = np.unique(sensitive_attr)
        group_metrics = {}
        
        for group in groups:
            mask = sensitive_attr == group
            if mask.sum() > 0:
                y_true_group = y_true[mask]
                y_pred_group = y_pred[mask]
                
                group_metrics[str(group)] = {
                    'sample_size': int(mask.sum()),
                    'accuracy': accuracy_score(y_true_group, y_pred_group),
                    'precision': precision_score(y_true_group, y_pred_group, zero_division=0),
                    'recall': recall_score(y_true_group, y_pred_group, zero_division=0),
                    'f1_score': f1_score(y_true_group, y_pred_group, zero_division=0),
                    'positive_rate': y_pred_group.mean()
                }
        
        return group_metrics
    
    def generate_report(self, output_path: str):
        """
        Generate a comprehensive fairness audit report.
        
        Args:
            output_path: Path to save the report
        """
        if not self.audit_results:
            raise ValueError("No audit results available. Run audit_model() first.")
        
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save detailed results as JSON
        import json
        with open(output_path / 'fairness_audit.json', 'w') as f:
            json.dump(self.audit_results, f, indent=2)
        
        # Create summary report
        self._create_summary_report(output_path)
        
        logger.info(f"Fairness report saved to {output_path}")
    
    def _create_summary_report(self, output_path: Path):
        """
        Create a human-readable summary report.
        
        Args:
            output_path: Directory to save the report
        """
        report_lines = ["# Fairness Audit Report\n\n"]
        
        # Overall metrics
        report_lines.append("## Overall Model Performance\n")
        for metric, value in self.audit_results['overall_metrics'].items():
            report_lines.append(f"- {metric}: {value:.4f}\n")
        
        report_lines.append("\n## Fairness Assessment by Attribute\n\n")
        
        # Fairness metrics for each attribute
        for attr_name in self.audit_results['fairness_metrics'].keys():
            report_lines.append(f"### {attr_name}\n\n")
            
            # Fairness metrics
            report_lines.append("**Fairness Metrics:**\n")
            for metric, value in self.audit_results['fairness_metrics'][attr_name].items():
                report_lines.append(f"- {metric}: {value:.4f}\n")
            
            # Fairness assessment
            report_lines.append("\n**Assessment:**\n")
            for criterion, status in self.audit_results['fairness_assessment'][attr_name].items():
                report_lines.append(f"- {criterion}: {status}\n")
            
            # Group metrics
            report_lines.append("\n**Group-Specific Performance:**\n")
            for group, metrics in self.audit_results['group_metrics'][attr_name].items():
                report_lines.append(f"\nGroup: {group}\n")
                for metric, value in metrics.items():
                    if isinstance(value, float):
                        report_lines.append(f"  - {metric}: {value:.4f}\n")
                    else:
                        report_lines.append(f"  - {metric}: {value}\n")
            
            report_lines.append("\n")
        
        # Save report
        with open(output_path / 'fairness_summary.md', 'w') as f:
            f.writelines(report_lines)
    
    def plot_group_comparison(
        self,
        sensitive_attr_name: str,
        save_path: Optional[str] = None
    ):
        """
        Create visualization comparing metrics across groups.
        
        Args:
            sensitive_attr_name: Name of the sensitive attribute
            save_path: Optional path to save the plot
        """
        if not self.audit_results:
            raise ValueError("No audit results available. Run audit_model() first.")
        
        group_metrics = self.audit_results['group_metrics'][sensitive_attr_name]
        
        # Prepare data for plotting
        groups = list(group_metrics.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        data = []
        for metric in metrics:
            for group in groups:
                data.append({
                    'Group': group,
                    'Metric': metric,
                    'Value': group_metrics[group][metric]
                })
        
        df = pd.DataFrame(data)
        
        # Create plot
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df, x='Metric', y='Value', hue='Group')
        plt.title(f'Performance Comparison Across {sensitive_attr_name}')
        plt.ylabel('Score')
        plt.ylim(0, 1)
        plt.legend(title=sensitive_attr_name)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        else:
            plt.show()
        
        plt.close()


def create_synthetic_sensitive_attributes(
    n_samples: int,
    random_state: int = 42
) -> Dict[str, np.ndarray]:
    """
    Create synthetic sensitive attributes for demonstration.
    
    Args:
        n_samples: Number of samples
        random_state: Random seed
        
    Returns:
        Dictionary of sensitive attributes
    """
    np.random.seed(random_state)
    
    return {
        'gender': np.random.choice(['Male', 'Female'], size=n_samples),
        'age_group': np.random.choice(['18-30', '31-50', '51+'], size=n_samples),
        'ethnicity': np.random.choice(['Group_A', 'Group_B'], size=n_samples)
    }
