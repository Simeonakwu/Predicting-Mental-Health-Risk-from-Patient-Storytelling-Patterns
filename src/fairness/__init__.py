"""Fairness auditing module initialization."""

from .fairness_auditor import (
    FairnessMetrics,
    FairnessAuditor,
    create_synthetic_sensitive_attributes
)

__all__ = [
    'FairnessMetrics',
    'FairnessAuditor',
    'create_synthetic_sensitive_attributes'
]
