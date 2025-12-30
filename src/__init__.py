"""
Depression Detection NLP System

An explainable BERT-based machine learning system for detecting depression risk
from DAIC-WOZ interview transcripts, featuring SHAP explainability, fairness auditing,
and an interactive dashboard.
"""

__version__ = "1.0.0"
__author__ = "Mental Health NLP Team"

from . import data
from . import models
from . import explainability
from . import fairness
from . import utils

__all__ = [
    'data',
    'models',
    'explainability',
    'fairness',
    'utils'
]
