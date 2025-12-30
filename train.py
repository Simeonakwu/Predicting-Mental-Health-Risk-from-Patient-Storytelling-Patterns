"""
Main training script for Depression Detection Model

This script provides a complete pipeline for training, evaluating,
and saving a depression detection model.
"""

import argparse
import yaml
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data.data_processor import load_sample_data, DAICWOZDataProcessor
from models.bert_model import DepressionDetectionModel
from explainability.shap_explainer import ModelExplainer
from fairness.fairness_auditor import FairnessAuditor, create_synthetic_sensitive_attributes
from utils.evaluation import ModelEvaluator
from utils.logger import setup_logging
from utils.visualization import plot_training_history


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def train_model(config: dict, logger):
    """
    Train the depression detection model.
    
    Args:
        config: Configuration dictionary
        logger: Logger instance
    """
    logger.info("Starting training pipeline...")
    
    # 1. Load and process data
    logger.info("Loading data...")
    transcripts, labels = load_sample_data()
    
    processor = DAICWOZDataProcessor("data")
    df = processor.create_dataset(transcripts, labels)
    
    # Split data
    train_df, val_df, test_df = processor.split_dataset(
        df,
        train_size=config['data']['train_split'],
        val_size=config['data']['val_split'],
        test_size=config['data']['test_split']
    )
    
    logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # 2. Initialize model
    logger.info("Initializing model...")
    model = DepressionDetectionModel(
        model_name=config['model']['name'],
        max_length=config['model']['max_length']
    )
    
    # 3. Prepare data loaders
    train_loader = model.prepare_data(
        texts=train_df['cleaned_transcript'].tolist(),
        labels=train_df['label'].tolist(),
        batch_size=config['training']['batch_size']
    )
    
    val_loader = model.prepare_data(
        texts=val_df['cleaned_transcript'].tolist(),
        labels=val_df['label'].tolist(),
        batch_size=config['training']['batch_size']
    )
    
    # 4. Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # 5. Training loop
    logger.info("Starting training...")
    num_epochs = config['training']['num_epochs']
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        # Train
        train_loss = model.train_step(train_loader, optimizer, criterion)
        train_losses.append(train_loss)
        
        # Validate
        val_loss, val_acc = model.evaluate(val_loader, criterion)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        logger.info(f"Epoch {epoch+1}/{num_epochs}:")
        logger.info(f"  Train Loss: {train_loss:.4f}")
        logger.info(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_dir = Path(config['output']['model_dir'])
            model_dir.mkdir(parents=True, exist_ok=True)
            model.save_model(str(model_dir / 'best_model'))
            logger.info(f"  Saved new best model (acc: {val_acc:.4f})")
    
    # Plot training history
    plot_dir = Path(config['output']['plots_dir'])
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_training_history(
        train_losses,
        val_losses,
        val_accs=val_accuracies,
        save_path=str(plot_dir / 'training_history.png')
    )
    
    # 6. Evaluate on test set
    logger.info("Evaluating on test set...")
    test_texts = test_df['cleaned_transcript'].tolist()
    test_labels = test_df['label'].values
    
    predictions, probabilities = model.predict(test_texts, batch_size=config['training']['batch_size'])
    
    # Calculate metrics
    evaluator = ModelEvaluator()
    metrics = evaluator.calculate_metrics(test_labels, predictions, probabilities)
    
    logger.info("Test Set Metrics:")
    for metric, value in metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    # Create evaluation report
    evaluator.create_evaluation_report(
        test_labels,
        predictions,
        probabilities,
        output_dir=config['output']['plots_dir']
    )
    
    # 7. Generate explanations
    logger.info("Generating SHAP explanations...")
    explainer = ModelExplainer(model.model, model.tokenizer, model.device)
    
    background_texts = train_df['cleaned_transcript'].sample(
        min(config['explainability']['num_background_samples'], len(train_df))
    ).tolist()
    
    explainer.initialize_explainer(background_texts, algorithm=config['explainability']['shap_algorithm'])
    
    # Explain a few samples
    sample_texts = test_texts[:min(5, len(test_texts))]
    for i, text in enumerate(sample_texts):
        logger.info(f"Explaining sample {i+1}...")
        shap_values = explainer.explain_prediction(text, show_plot=False)
    
    # 8. Fairness audit
    logger.info("Performing fairness audit...")
    sensitive_attrs = create_synthetic_sensitive_attributes(len(test_labels))
    
    auditor = FairnessAuditor()
    audit_results = auditor.audit_model(
        y_true=test_labels,
        y_pred=predictions,
        sensitive_attributes={'gender': sensitive_attrs['gender']},
        threshold=config['fairness']['fairness_threshold']
    )
    
    # Generate fairness report
    report_dir = Path(config['output']['reports_dir'])
    report_dir.mkdir(parents=True, exist_ok=True)
    auditor.generate_report(str(report_dir))
    
    auditor.plot_group_comparison(
        'gender',
        save_path=str(plot_dir / 'fairness_comparison.png')
    )
    
    logger.info("Training pipeline completed!")
    logger.info(f"Results saved to: {config['output']['results_dir']}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train Depression Detection Model")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Path to configuration file'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup logging
    logger = setup_logging(
        log_level=config['logging']['level'],
        log_dir=config['logging']['log_dir']
    )
    
    logger.info("=" * 80)
    logger.info("Depression Detection Training Pipeline")
    logger.info("=" * 80)
    
    # Train model
    try:
        train_model(config, logger)
    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
