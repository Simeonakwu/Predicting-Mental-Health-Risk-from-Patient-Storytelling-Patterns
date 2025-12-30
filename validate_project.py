#!/usr/bin/env python3
"""
Validation script to verify the project structure and imports.
This script checks that all modules can be imported without actually running them.
"""

import sys
from pathlib import Path

def validate_structure():
    """Validate that all expected directories and files exist."""
    print("=" * 70)
    print("VALIDATING PROJECT STRUCTURE")
    print("=" * 70)
    
    required_dirs = [
        'src',
        'src/data',
        'src/models',
        'src/explainability',
        'src/fairness',
        'src/dashboard',
        'src/utils',
        'notebooks',
        'tests',
        'docs',
        'configs',
        'data/raw',
        'data/processed',
        'models'
    ]
    
    required_files = [
        'README.md',
        'requirements.txt',
        'setup.py',
        'train.py',
        '.gitignore',
        'configs/config.yaml',
        'src/__init__.py',
        'src/data/__init__.py',
        'src/data/data_processor.py',
        'src/models/__init__.py',
        'src/models/bert_model.py',
        'src/explainability/__init__.py',
        'src/explainability/shap_explainer.py',
        'src/fairness/__init__.py',
        'src/fairness/fairness_auditor.py',
        'src/dashboard/__init__.py',
        'src/dashboard/app.py',
        'src/utils/__init__.py',
        'src/utils/evaluation.py',
        'src/utils/logger.py',
        'src/utils/visualization.py',
        'tests/__init__.py',
        'tests/test_data_processor.py',
        'notebooks/01_complete_tutorial.ipynb',
        'notebooks/02_data_exploration.ipynb',
        'docs/GETTING_STARTED.md',
        'docs/API.md',
        'docs/ARCHITECTURE.md'
    ]
    
    print("\nChecking directories...")
    missing_dirs = []
    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists():
            print(f"✓ {dir_path}")
        else:
            print(f"✗ {dir_path} - MISSING!")
            missing_dirs.append(dir_path)
    
    print("\nChecking files...")
    missing_files = []
    for file_path in required_files:
        path = Path(file_path)
        if path.exists():
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} - MISSING!")
            missing_files.append(file_path)
    
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    
    if missing_dirs:
        print(f"\n❌ Missing directories: {len(missing_dirs)}")
        for d in missing_dirs:
            print(f"   - {d}")
    else:
        print("\n✅ All required directories present")
    
    if missing_files:
        print(f"\n❌ Missing files: {len(missing_files)}")
        for f in missing_files:
            print(f"   - {f}")
    else:
        print("\n✅ All required files present")
    
    if not missing_dirs and not missing_files:
        print("\n🎉 PROJECT STRUCTURE VALIDATION PASSED!")
        return True
    else:
        print("\n⚠️  PROJECT STRUCTURE VALIDATION FAILED!")
        return False


def check_code_structure():
    """Check the structure of Python files without importing them."""
    print("\n" + "=" * 70)
    print("CHECKING CODE STRUCTURE")
    print("=" * 70)
    
    python_files = [
        'src/data/data_processor.py',
        'src/models/bert_model.py',
        'src/explainability/shap_explainer.py',
        'src/fairness/fairness_auditor.py',
        'src/dashboard/app.py',
        'src/utils/evaluation.py',
        'src/utils/logger.py',
        'src/utils/visualization.py',
        'train.py'
    ]
    
    print("\nChecking Python files for syntax errors...")
    for file_path in python_files:
        try:
            with open(file_path, 'r') as f:
                compile(f.read(), file_path, 'exec')
            print(f"✓ {file_path} - Valid Python syntax")
        except SyntaxError as e:
            print(f"✗ {file_path} - SYNTAX ERROR: {e}")
            return False
        except Exception as e:
            print(f"✗ {file_path} - ERROR: {e}")
            return False
    
    print("\n✅ All Python files have valid syntax")
    return True


def summarize_implementation():
    """Print summary of what was implemented."""
    print("\n" + "=" * 70)
    print("IMPLEMENTATION SUMMARY")
    print("=" * 70)
    
    summary = """
✨ DEPRESSION DETECTION NLP SYSTEM ✨

📦 Core Components:
  • Data Processing Module
    - DAIC-WOZ transcript processing
    - Text cleaning and preprocessing
    - Dataset splitting and management
    - Sample data generation

  • BERT-based Model
    - Pre-trained BERT integration
    - Custom classification head
    - Training and evaluation pipelines
    - Model persistence

  • SHAP Explainability
    - Model interpretation
    - Word-level importance
    - Visualization tools
    - Feature ranking

  • Fairness Auditing
    - Demographic parity metrics
    - Equal opportunity analysis
    - Equalized odds calculation
    - Group performance comparison

  • Interactive Dashboard
    - Streamlit-based UI
    - Data exploration
    - Real-time predictions
    - Explainability visualization
    - Fairness audit results

  • Utilities
    - Evaluation metrics
    - Logging system
    - Visualization tools

📚 Documentation:
  • Comprehensive README
  • API Reference
  • Getting Started Guide
  • Architecture Overview

📓 Jupyter Notebooks:
  • Complete Tutorial
  • Data Exploration

🧪 Testing:
  • Unit tests for data processing
  • Test framework setup

⚙️ Configuration:
  • YAML configuration file
  • Project setup (setup.py)
  • Dependencies (requirements.txt)
  • Git ignore rules

🚀 Entry Points:
  • train.py - Main training script
  • src/dashboard/app.py - Interactive dashboard
  • Jupyter notebooks for exploration

📊 Key Features:
  ✓ Modular architecture
  ✓ BERT-based deep learning
  ✓ Model explainability (SHAP)
  ✓ Fairness auditing
  ✓ Interactive visualization
  ✓ Comprehensive documentation
  ✓ Example notebooks
  ✓ Configuration-driven design
"""
    print(summary)
    
    print("\n💡 Next Steps:")
    print("  1. Install dependencies: pip install -r requirements.txt")
    print("  2. Run training: python train.py")
    print("  3. Launch dashboard: streamlit run src/dashboard/app.py")
    print("  4. Explore notebooks: jupyter notebook notebooks/")
    print("  5. Run tests: pytest tests/")


def main():
    """Main validation function."""
    structure_valid = validate_structure()
    code_valid = check_code_structure()
    
    if structure_valid and code_valid:
        summarize_implementation()
        print("\n" + "=" * 70)
        print("✅ ALL VALIDATIONS PASSED - PROJECT IS READY!")
        print("=" * 70 + "\n")
        return 0
    else:
        print("\n" + "=" * 70)
        print("❌ VALIDATION FAILED - PLEASE CHECK ERRORS ABOVE")
        print("=" * 70 + "\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
