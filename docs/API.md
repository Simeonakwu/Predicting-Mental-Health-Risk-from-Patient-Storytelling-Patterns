# API Reference

## Data Processing

### `DAICWOZDataProcessor`

Processes DAIC-WOZ interview transcripts for depression detection.

```python
from src.data.data_processor import DAICWOZDataProcessor
```

#### Methods

##### `__init__(data_dir: str)`
Initialize the data processor.

**Parameters:**
- `data_dir` (str): Path to the data directory

**Example:**
```python
processor = DAICWOZDataProcessor("data/raw")
```

##### `load_transcripts(transcript_file: str) -> pd.DataFrame`
Load transcripts from a CSV file.

**Parameters:**
- `transcript_file` (str): Path to the transcript file

**Returns:**
- DataFrame containing transcripts

##### `clean_text(text: str) -> str`
Clean and preprocess transcript text.

**Parameters:**
- `text` (str): Raw transcript text

**Returns:**
- Cleaned text string

##### `create_dataset(transcripts, labels, ids=None) -> pd.DataFrame`
Create a structured dataset from transcripts and labels.

**Parameters:**
- `transcripts` (List[str]): List of transcript texts
- `labels` (List[int]): List of labels (0 or 1)
- `ids` (Optional[List[str]]): Optional participant IDs

**Returns:**
- DataFrame with processed data

##### `split_dataset(df, train_size=0.7, val_size=0.15, test_size=0.15) -> Tuple`
Split dataset into train, validation, and test sets.

**Returns:**
- Tuple of (train_df, val_df, test_df)

---

## Model

### `DepressionDetectionModel`

BERT-based binary classifier for depression detection.

```python
from src.models.bert_model import DepressionDetectionModel
```

#### Methods

##### `__init__(model_name='bert-base-uncased', max_length=512, device=None)`
Initialize the model.

**Parameters:**
- `model_name` (str): Name of pre-trained BERT model
- `max_length` (int): Maximum sequence length
- `device` (str, optional): Device to use ('cuda' or 'cpu')

**Example:**
```python
model = DepressionDetectionModel(
    model_name='bert-base-uncased',
    max_length=512
)
```

##### `prepare_data(texts, labels=None, batch_size=16) -> DataLoader`
Prepare data for training or inference.

**Parameters:**
- `texts` (List[str]): List of transcript texts
- `labels` (Optional[List[int]]): List of labels
- `batch_size` (int): Batch size

**Returns:**
- PyTorch DataLoader

##### `train_step(dataloader, optimizer, criterion) -> float`
Perform one training epoch.

**Parameters:**
- `dataloader` (DataLoader): Training data
- `optimizer` (Optimizer): PyTorch optimizer
- `criterion` (Module): Loss function

**Returns:**
- Average training loss

##### `evaluate(dataloader, criterion) -> Tuple[float, float]`
Evaluate the model.

**Returns:**
- Tuple of (loss, accuracy)

##### `predict(texts, batch_size=16) -> Tuple[np.ndarray, np.ndarray]`
Make predictions on new texts.

**Parameters:**
- `texts` (List[str]): List of texts
- `batch_size` (int): Batch size

**Returns:**
- Tuple of (predictions, probabilities)

**Example:**
```python
predictions, probabilities = model.predict(test_texts)
```

##### `save_model(save_path: str)`
Save the model to disk.

##### `load_model(load_path: str)`
Load a saved model from disk.

---

## Explainability

### `ModelExplainer`

Provides SHAP-based explanations for model predictions.

```python
from src.explainability.shap_explainer import ModelExplainer
```

#### Methods

##### `__init__(model, tokenizer, device=None)`
Initialize the explainer.

**Parameters:**
- `model`: Trained BERT model
- `tokenizer`: BERT tokenizer
- `device` (str, optional): Device to use

##### `initialize_explainer(background_texts, algorithm='partition')`
Initialize the SHAP explainer with background data.

**Parameters:**
- `background_texts` (List[str]): Sample texts for background
- `algorithm` (str): SHAP algorithm ('partition' or 'kernel')

##### `explain_prediction(text, show_plot=True) -> shap.Explanation`
Generate SHAP explanation for a single prediction.

**Parameters:**
- `text` (str): Text to explain
- `show_plot` (bool): Whether to display visualization

**Returns:**
- SHAP explanation object

**Example:**
```python
explainer = ModelExplainer(model.model, model.tokenizer)
explainer.initialize_explainer(background_texts)
shap_values = explainer.explain_prediction(test_text)
```

##### `get_feature_importance(shap_values, top_k=20) -> pd.DataFrame`
Extract top important features from SHAP values.

**Returns:**
- DataFrame with feature importance scores

---

## Fairness Auditing

### `FairnessAuditor`

Comprehensive fairness auditing for the model.

```python
from src.fairness.fairness_auditor import FairnessAuditor
```

#### Methods

##### `__init__()`
Initialize the fairness auditor.

##### `audit_model(y_true, y_pred, sensitive_attributes, threshold=0.1) -> Dict`
Perform comprehensive fairness audit.

**Parameters:**
- `y_true` (np.ndarray): True labels
- `y_pred` (np.ndarray): Model predictions
- `sensitive_attributes` (Dict[str, np.ndarray]): Dictionary of sensitive attributes
- `threshold` (float): Fairness threshold

**Returns:**
- Dictionary containing audit results

**Example:**
```python
auditor = FairnessAuditor()
results = auditor.audit_model(
    y_true=test_labels,
    y_pred=predictions,
    sensitive_attributes={'gender': gender_data},
    threshold=0.1
)
```

##### `generate_report(output_path: str)`
Generate a comprehensive fairness audit report.

##### `plot_group_comparison(sensitive_attr_name, save_path=None)`
Create visualization comparing metrics across groups.

---

## Evaluation

### `ModelEvaluator`

Comprehensive model evaluation with various metrics.

```python
from src.utils.evaluation import ModelEvaluator
```

#### Static Methods

##### `calculate_metrics(y_true, y_pred, y_proba=None) -> Dict[str, float]`
Calculate comprehensive evaluation metrics.

**Parameters:**
- `y_true` (np.ndarray): True labels
- `y_pred` (np.ndarray): Predicted labels
- `y_proba` (np.ndarray, optional): Prediction probabilities

**Returns:**
- Dictionary of metrics

**Example:**
```python
metrics = ModelEvaluator.calculate_metrics(y_true, y_pred, y_proba)
print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"F1 Score: {metrics['f1_score']:.4f}")
```

##### `plot_confusion_matrix(y_true, y_pred, save_path=None)`
Plot confusion matrix.

##### `plot_roc_curve(y_true, y_proba, save_path=None)`
Plot ROC curve.

##### `create_evaluation_report(y_true, y_pred, y_proba=None, output_dir=None)`
Create comprehensive evaluation report with all visualizations.

---

## Utilities

### Logging

```python
from src.utils.logger import setup_logging, get_logger

# Setup logging
logger = setup_logging(log_level="INFO", log_dir="logs")

# Get logger
logger = get_logger("MyModule")
logger.info("Processing data...")
```

### Visualization

```python
from src.utils.visualization import (
    plot_training_history,
    plot_label_distribution,
    create_interactive_scatter
)

# Plot training history
plot_training_history(train_losses, val_losses, train_accs, val_accs)

# Plot label distribution
plot_label_distribution(labels, title="Label Distribution")
```

---

## Configuration

### Loading Configuration

```python
import yaml

with open('configs/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

model_config = config['model']
training_config = config['training']
```

### Configuration Structure

```yaml
model:
  name: "bert-base-uncased"
  max_length: 512
  dropout: 0.3

training:
  batch_size: 16
  learning_rate: 2.0e-5
  num_epochs: 10

fairness:
  fairness_threshold: 0.1
```

---

## Dashboard

### Running the Dashboard

```bash
streamlit run src/dashboard/app.py
```

### Dashboard Features

- **Home**: Overview and quick stats
- **Data Explorer**: Visualize and explore datasets
- **Model Predictions**: Make predictions on new data
- **Explainability**: View SHAP explanations
- **Fairness Audit**: Review fairness metrics

---

## Complete Example

```python
# Import modules
from src.data.data_processor import load_sample_data, DAICWOZDataProcessor
from src.models.bert_model import DepressionDetectionModel
from src.explainability.shap_explainer import ModelExplainer
from src.fairness.fairness_auditor import FairnessAuditor
from src.utils.evaluation import ModelEvaluator
import torch.nn as nn
import torch.optim as optim

# 1. Load and process data
transcripts, labels = load_sample_data()
processor = DAICWOZDataProcessor("data")
df = processor.create_dataset(transcripts, labels)
train_df, val_df, test_df = processor.split_dataset(df)

# 2. Train model
model = DepressionDetectionModel()
train_loader = model.prepare_data(
    train_df['cleaned_transcript'].tolist(),
    train_df['label'].tolist()
)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.model.parameters(), lr=2e-5)

for epoch in range(3):
    loss = model.train_step(train_loader, optimizer, criterion)
    print(f"Epoch {epoch+1}, Loss: {loss:.4f}")

# 3. Evaluate
test_texts = test_df['cleaned_transcript'].tolist()
test_labels = test_df['label'].values
predictions, probabilities = model.predict(test_texts)
metrics = ModelEvaluator.calculate_metrics(test_labels, predictions, probabilities)

# 4. Explain
explainer = ModelExplainer(model.model, model.tokenizer)
explainer.initialize_explainer(train_df['cleaned_transcript'].sample(10).tolist())
shap_values = explainer.explain_prediction(test_texts[0])

# 5. Audit fairness
from src.fairness.fairness_auditor import create_synthetic_sensitive_attributes
sensitive_attrs = create_synthetic_sensitive_attributes(len(test_labels))
auditor = FairnessAuditor()
results = auditor.audit_model(test_labels, predictions, sensitive_attrs)
```
