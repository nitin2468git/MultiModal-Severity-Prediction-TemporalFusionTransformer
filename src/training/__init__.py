#!/usr/bin/env python3
"""
Training module for COVID-19 TFT Severity Prediction.

This module contains the training pipeline for the Temporal Fusion Transformer
model with multi-task learning and comprehensive monitoring.
"""

from .trainer import COVID19Trainer

__all__ = [
    'COVID19Trainer'
]

__version__ = "1.0.0"
__author__ = "COVID-19 TFT Project Team"
__description__ = "Training pipeline for COVID-19 TFT severity prediction" 