# Depression Detection NLP System

An explainable BERT-based machine learning system for detecting depression risk from DAIC-WOZ interview transcripts, featuring SHAP explainability, fairness auditing, and an interactive dashboard.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🌟 Features

- **BERT-based Modeling**: State-of-the-art transformer architecture for depression detection
- **SHAP Explainability**: Understand which words and phrases influence model predictions
- **Fairness Auditing**: Comprehensive bias detection and mitigation across demographic groups
- **Interactive Dashboard**: Streamlit-based UI for exploration and analysis
- **Modular Architecture**: Clean, well-documented, and extensible codebase
- **Example Notebooks**: Step-by-step tutorials for all features

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Usage](#usage)
  - [Data Processing](#data-processing)
  - [Model Training](#model-training)
  - [Explainability](#explainability)
  - [Fairness Auditing](#fairness-auditing)
  - [Interactive Dashboard](#interactive-dashboard)
- [Notebooks](#notebooks)
- [Configuration](#configuration)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (optional, but recommended for training)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/Simeonakwu/Predicting-Mental-Health-Risk-from-Patient-Storytelling-Patterns.git
cd Predicting-Mental-Health-Risk-from-Patient-Storytelling-Patterns
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. (Optional) Install the package in development mode:
```bash
pip install -e .
```

## ⚡ Quick Start

### Using Sample Data

```python
from src.data.data_processor import load_sample_data, DAICWOZDataProcessor
from src.models.bert_model import DepressionDetectionModel

# Load sample data
transcripts, labels = load_sample_data()

# Create dataset
processor = DAICWOZDataProcessor("data")
df = processor.create_dataset(transcripts, labels)

# Initialize and train model
model = DepressionDetectionModel()
# ... training code (see notebooks for details)
```

### Launch Interactive Dashboard

```bash
cd src/dashboard
streamlit run app.py
```

## 📁 Project Structure

```
.
├── src/                          # Source code
│   ├── data/                     # Data processing modules
│   │   ├── __init__.py
│   │   └── data_processor.py     # DAIC-WOZ data processing
│   ├── models/                   # Model implementations
│   │   ├── __init__.py
│   │   └── bert_model.py         # BERT-based classifier
│   ├── explainability/           # Explainability modules
│   │   ├── __init__.py
│   │   └── shap_explainer.py     # SHAP-based explanations
│   ├── fairness/                 # Fairness auditing
│   │   ├── __init__.py
│   │   └── fairness_auditor.py   # Bias detection and metrics
│   ├── dashboard/                # Interactive dashboard
│   │   ├── __init__.py
│   │   └── app.py                # Streamlit application
│   └── utils/                    # Utility modules
│       ├── __init__.py
│       ├── evaluation.py         # Evaluation metrics
│       ├── logger.py             # Logging configuration
│       └── visualization.py      # Plotting utilities
├── notebooks/                    # Jupyter notebooks
│   ├── 01_complete_tutorial.ipynb
│   └── 02_data_exploration.ipynb
├── data/                         # Data directory
│   ├── raw/                      # Raw DAIC-WOZ data
│   └── processed/                # Processed datasets
├── models/                       # Saved models
├── configs/                      # Configuration files
│   └── config.yaml
├── tests/                        # Unit tests
├── docs/                         # Documentation
├── requirements.txt              # Python dependencies
├── setup.py                      # Package setup
├── .gitignore
└── README.md
```

## 📖 Usage

### Data Processing

```python
from src.data.data_processor import DAICWOZDataProcessor

# Initialize processor
processor = DAICWOZDataProcessor("data/raw")

# Load and process transcripts
df = processor.load_transcripts("transcripts.csv")

# Split data
train_df, val_df, test_df = processor.split_dataset(df)

# Save processed data
processor.save_processed_data(train_df, val_df, test_df, "data/processed")
```

### Model Training

```python
from src.models.bert_model import DepressionDetectionModel
import torch.nn as nn
import torch.optim as optim

# Initialize model
model = DepressionDetectionModel(
    model_name='bert-base-uncased',
    max_length=512
)

# Prepare data
train_loader = model.prepare_data(train_texts, train_labels, batch_size=16)
val_loader = model.prepare_data(val_texts, val_labels, batch_size=16)

# Training
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.model.parameters(), lr=2e-5)

for epoch in range(num_epochs):
    train_loss = model.train_step(train_loader, optimizer, criterion)
    val_loss, val_acc = model.evaluate(val_loader, criterion)
    print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Acc={val_acc:.4f}")

# Save model
model.save_model("models/trained_model")
```

### Explainability

```python
from src.explainability.shap_explainer import ModelExplainer

# Initialize explainer
explainer = ModelExplainer(
    model=model.model,
    tokenizer=model.tokenizer
)

# Initialize with background data
explainer.initialize_explainer(background_texts)

# Explain a prediction
shap_values = explainer.explain_prediction(test_text, show_plot=True)

# Get feature importance
importance_df = explainer.get_feature_importance(shap_values, top_k=20)
```

### Fairness Auditing

```python
from src.fairness.fairness_auditor import FairnessAuditor

# Initialize auditor
auditor = FairnessAuditor()

# Perform audit
results = auditor.audit_model(
    y_true=test_labels,
    y_pred=predictions,
    sensitive_attributes={'gender': gender_data},
    threshold=0.1
)

# Generate report
auditor.generate_report("results/fairness_report")

# Visualize group comparison
auditor.plot_group_comparison('gender', save_path='results/group_comparison.png')
```

### Interactive Dashboard

The dashboard provides an intuitive interface for:
- Loading and exploring data
- Making predictions on new transcripts
- Visualizing SHAP explanations
- Reviewing fairness audit results

Launch with:
```bash
streamlit run src/dashboard/app.py
```

## 📓 Notebooks

Two comprehensive Jupyter notebooks are provided:

1. **Complete Tutorial** (`01_complete_tutorial.ipynb`): End-to-end workflow including:
   - Data loading and preprocessing
   - Model training
   - Evaluation
   - SHAP explainability
   - Fairness auditing

2. **Data Exploration** (`02_data_exploration.ipynb`): Detailed data analysis:
   - Dataset statistics
   - Label distributions
   - Text visualizations
   - Pattern identification

## ⚙️ Configuration

Model and training parameters can be configured in `configs/config.yaml`:

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

## 📚 Documentation

Additional documentation is available in the `docs/` directory:
- API Reference
- Architecture Overview
- Best Practices
- Troubleshooting Guide

## 🧪 Testing

Run tests with pytest:
```bash
pytest tests/
```

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔗 References

- **DAIC-WOZ Dataset**: [https://dcapswoz.ict.usc.edu/](https://dcapswoz.ict.usc.edu/)
- **BERT**: Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
- **SHAP**: Lundberg & Lee, "A Unified Approach to Interpreting Model Predictions"
- **Fairness Metrics**: Mehrabi et al., "A Survey on Bias and Fairness in Machine Learning"

## 📧 Contact

For questions or issues, please open a GitHub issue or contact the maintainers.

## 🙏 Acknowledgments

- DAIC-WOZ dataset creators at USC
- Hugging Face for transformer implementations
- SHAP library developers
- Open-source ML community
