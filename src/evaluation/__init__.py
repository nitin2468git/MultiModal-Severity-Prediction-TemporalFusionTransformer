#!/usr/bin/env python3
"""
Evaluation module for COVID-19 TFT Severity Prediction.

This module contains evaluation metrics and clinical validation tools
for COVID-19 severity prediction models.
"""

from .metrics import COVID19Metrics

__all__ = [
    'COVID19Metrics'
]

__version__ = "1.0.0"
__author__ = "COVID-19 TFT Project Team"
__description__ = "Evaluation metrics for COVID-19 severity prediction" 