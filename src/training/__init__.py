"""
Training Module

This module contains training utilities, loss functions, and training loops.
"""

from .trainer import (
    SmoothL1WithLabelSmoothing,
    train_epoch,
    evaluate,
    train_model
)

__all__ = [
    'SmoothL1WithLabelSmoothing',
    'train_epoch',
    'evaluate',
    'train_model'
]
