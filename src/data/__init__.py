"""
Data Module

This module contains data loading, preprocessing, and feature extraction utilities.
"""

from .dataset import IELTSDataset, load_ielts_dataset, create_data_loaders
from .features import extract_linguistic_features, normalize_features, apply_normalization

__all__ = [
    'IELTSDataset',
    'load_ielts_dataset',
    'create_data_loaders',
    'extract_linguistic_features',
    'normalize_features',
    'apply_normalization'
]
