#!/usr/bin/env python3
"""
Data processing module for COVID-19 TFT Severity Prediction.

This module contains all data processing components for the Temporal Fusion Transformer
model, including data loading, validation, feature engineering, timeline building,
and TFT-specific formatting.
"""

from .synthea_loader import SyntheaLoader
from .timeline_builder import TimelineBuilder
from .feature_engineer import FeatureEngineer
from .tft_formatter import TFTFormatter, TFTDataset
from .data_validator import DataValidator
from .data_pipeline import DataPipeline

__all__ = [
    'SyntheaLoader',
    'TimelineBuilder', 
    'FeatureEngineer',
    'TFTFormatter',
    'TFTDataset',
    'DataValidator',
    'DataPipeline'
]

__version__ = "1.0.0"
__author__ = "COVID-19 TFT Project Team"
__description__ = "Data processing pipeline for COVID-19 severity prediction using Temporal Fusion Transformer" 