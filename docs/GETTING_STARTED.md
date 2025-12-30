# Getting Started Guide

## Introduction

This guide will help you get started with the Depression Detection NLP System. By the end of this guide, you'll be able to:
- Set up your environment
- Load and process data
- Train a model
- Generate explanations
- Audit model fairness

## Prerequisites

- Python 3.8 or higher
- Basic understanding of machine learning and NLP
- Familiarity with PyTorch (helpful but not required)

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/Simeonakwu/Predicting-Mental-Health-Risk-from-Patient-Storytelling-Patterns.git
cd Predicting-Mental-Health-Risk-from-Patient-Storytelling-Patterns
```

### 2. Create Virtual Environment

**On Linux/Mac:**
```bash
python -m venv venv
source venv/bin/activate
```

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This will install all required packages including:
- PyTorch
- Transformers (Hugging Face)
- SHAP
- Streamlit
- And other dependencies

### 4. Verify Installation

```python
python -c "import torch; import transformers; import shap; print('All packages installed successfully!')"
```

## Using the System

### Option 1: Interactive Dashboard (Recommended for Beginners)

The easiest way to explore the system is through the interactive dashboard:

```bash
cd src/dashboard
streamlit run app.py
```

This will open a web interface where you can:
- Load sample data
- Make predictions
- View explanations
- Review fairness metrics

### Option 2: Jupyter Notebooks

For a guided learning experience, use the provided notebooks:

```bash
jupyter notebook notebooks/01_complete_tutorial.ipynb
```

### Option 3: Python Scripts

For programmatic access, use the Python API:

```python
from src.data.data_processor import load_sample_data
from src.models.bert_model import DepressionDetectionModel

# Load data
transcripts, labels = load_sample_data()

# Initialize model
model = DepressionDetectionModel()

# Make predictions
predictions, probabilities = model.predict(transcripts)
```

## Working with Your Own Data

### Data Format

The system expects transcript data in the following format:

**CSV Format:**
```csv
id,transcript,label
1,"Participant: I've been feeling down...",1
2,"Participant: Life is going well...",0
```

**Fields:**
- `id`: Unique identifier
- `transcript`: Interview transcript text
- `label`: 0 (no depression) or 1 (depression)

### Loading Custom Data

```python
from src.data.data_processor import DAICWOZDataProcessor

# Initialize processor
processor = DAICWOZDataProcessor("path/to/your/data")

# Load your data
df = processor.load_transcripts("your_data.csv")

# Process and split
train_df, val_df, test_df = processor.split_dataset(df)
```

## Training Your First Model

Here's a minimal example to train a model:

```python
from src.models.bert_model import DepressionDetectionModel
import torch.nn as nn
import torch.optim as optim

# 1. Load and prepare data
model = DepressionDetectionModel()
train_loader = model.prepare_data(train_texts, train_labels)

# 2. Set up training
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.model.parameters(), lr=2e-5)

# 3. Train
for epoch in range(3):  # Start with 3 epochs
    loss = model.train_step(train_loader, optimizer, criterion)
    print(f"Epoch {epoch+1}, Loss: {loss:.4f}")

# 4. Save
model.save_model("models/my_first_model")
```

## Common Issues and Solutions

### Issue: CUDA Out of Memory

**Solution:** Reduce batch size in your configuration:
```python
train_loader = model.prepare_data(texts, labels, batch_size=4)
```

### Issue: Module Not Found Error

**Solution:** Ensure you're running from the correct directory and have activated your virtual environment:
```bash
source venv/bin/activate  # or venv\Scripts\activate on Windows
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Issue: Slow Training

**Solution:** 
1. Ensure you're using GPU if available
2. Reduce max_length: `model = DepressionDetectionModel(max_length=256)`
3. Use a smaller BERT model: `model_name='distilbert-base-uncased'`

## Next Steps

1. **Explore the Notebooks**: Start with `01_complete_tutorial.ipynb`
2. **Try the Dashboard**: Get familiar with the interactive interface
3. **Read the API Documentation**: Learn about advanced features
4. **Experiment**: Try different configurations and datasets

## Getting Help

- Check the [FAQ](FAQ.md)
- Review the [API Reference](API.md)
- Open an issue on GitHub
- Contact the maintainers

## Best Practices

1. **Always split your data**: Use train/val/test splits
2. **Monitor fairness**: Run fairness audits on all models
3. **Explain predictions**: Use SHAP for critical decisions
4. **Version your models**: Keep track of model versions and configurations
5. **Document changes**: Maintain logs of experiments

## Additional Resources

- [BERT Paper](https://arxiv.org/abs/1810.04805)
- [SHAP Documentation](https://shap.readthedocs.io/)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [Transformers Documentation](https://huggingface.co/docs/transformers/)
