"""
Visualization Utilities

Provides common visualization functions for the depression detection system.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
import plotly.graph_objects as go
import plotly.express as px


# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)


def plot_training_history(
    train_losses: List[float],
    val_losses: List[float],
    train_accs: Optional[List[float]] = None,
    val_accs: Optional[List[float]] = None,
    save_path: Optional[str] = None
):
    """
    Plot training history.
    
    Args:
        train_losses: Training losses per epoch
        val_losses: Validation losses per epoch
        train_accs: Training accuracies per epoch (optional)
        val_accs: Validation accuracies per epoch (optional)
        save_path: Optional path to save the plot
    """
    epochs = range(1, len(train_losses) + 1)
    
    if train_accs is not None and val_accs is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot losses
        ax1.plot(epochs, train_losses, 'b-o', label='Training Loss')
        ax1.plot(epochs, val_losses, 'r-o', label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot accuracies
        ax2.plot(epochs, train_accs, 'b-o', label='Training Accuracy')
        ax2.plot(epochs, val_accs, 'r-o', label='Validation Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    else:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(epochs, train_losses, 'b-o', label='Training Loss')
        ax.plot(epochs, val_losses, 'r-o', label='Validation Loss')
        ax.set_title('Training and Validation Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    else:
        plt.show()
    
    plt.close()


def plot_label_distribution(
    labels: np.ndarray,
    title: str = "Label Distribution",
    save_path: Optional[str] = None
):
    """
    Plot distribution of labels.
    
    Args:
        labels: Array of labels
        title: Plot title
        save_path: Optional path to save the plot
    """
    unique, counts = np.unique(labels, return_counts=True)
    
    plt.figure(figsize=(8, 6))
    plt.bar(['No Depression', 'Depression'], counts, color=['skyblue', 'salmon'])
    plt.title(title)
    plt.ylabel('Count')
    plt.xlabel('Label')
    
    # Add count labels on bars
    for i, count in enumerate(counts):
        plt.text(i, count, str(count), ha='center', va='bottom')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    else:
        plt.show()
    
    plt.close()


def plot_word_cloud(
    texts: List[str],
    title: str = "Word Cloud",
    save_path: Optional[str] = None
):
    """
    Generate and plot word cloud from texts.
    
    Args:
        texts: List of text strings
        title: Plot title
        save_path: Optional path to save the plot
    """
    try:
        from wordcloud import WordCloud
        
        # Combine all texts
        combined_text = ' '.join(texts)
        
        # Generate word cloud
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            colormap='viridis'
        ).generate(combined_text)
        
        plt.figure(figsize=(12, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.title(title)
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        else:
            plt.show()
        
        plt.close()
        
    except ImportError:
        print("WordCloud not installed. Install with: pip install wordcloud")


def plot_metrics_comparison(
    metrics_dict: Dict[str, Dict[str, float]],
    title: str = "Metrics Comparison",
    save_path: Optional[str] = None
):
    """
    Plot comparison of metrics across different models or configurations.
    
    Args:
        metrics_dict: Dictionary mapping model names to their metrics
        title: Plot title
        save_path: Optional path to save the plot
    """
    df = pd.DataFrame(metrics_dict).T
    
    df.plot(kind='bar', figsize=(12, 6))
    plt.title(title)
    plt.xlabel('Model')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    plt.legend(title='Metrics')
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    else:
        plt.show()
    
    plt.close()


def create_interactive_scatter(
    x: np.ndarray,
    y: np.ndarray,
    labels: np.ndarray,
    hover_text: Optional[List[str]] = None,
    title: str = "Interactive Scatter Plot"
) -> go.Figure:
    """
    Create an interactive scatter plot using Plotly.
    
    Args:
        x: X coordinates
        y: Y coordinates
        labels: Point labels for coloring
        hover_text: Optional hover text for each point
        title: Plot title
        
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    for label in np.unique(labels):
        mask = labels == label
        label_name = 'Depression' if label == 1 else 'No Depression'
        
        fig.add_trace(go.Scatter(
            x=x[mask],
            y=y[mask],
            mode='markers',
            name=label_name,
            text=hover_text[mask] if hover_text is not None else None,
            hoverinfo='text' if hover_text is not None else 'x+y',
            marker=dict(size=8, opacity=0.7)
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Component 1',
        yaxis_title='Component 2',
        hovermode='closest'
    )
    
    return fig


def plot_attention_weights(
    tokens: List[str],
    attention_weights: np.ndarray,
    title: str = "Attention Weights",
    save_path: Optional[str] = None
):
    """
    Visualize attention weights as a heatmap.
    
    Args:
        tokens: List of tokens
        attention_weights: Attention weight matrix
        title: Plot title
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        attention_weights,
        xticklabels=tokens,
        yticklabels=tokens,
        cmap='viridis',
        center=0,
        square=True,
        linewidths=0.5
    )
    plt.title(title)
    plt.xlabel('Tokens')
    plt.ylabel('Tokens')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    else:
        plt.show()
    
    plt.close()
