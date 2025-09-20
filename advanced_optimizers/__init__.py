"""
Advanced Training Algorithms
============================

This package implements advanced training algorithms for deep neural networks:
- Reducing Flipping Errors (FER)
- Gradient Correction beyond Gradient Descent 
- Forward Signal Propagation Learning
- Selective Localized Learning
- Zero-Shot Hyperparameter Transfer

All optimizers follow PyTorch's optimizer API for easy integration.
"""

from .fer_optimizer import FEROptimizer
from .gradient_correction import GradientCorrectionOptimizer
from .forward_signal import ForwardSignalOptimizer
from .localized_learning import LocalizedLearningOptimizer
from .zero_shot_transfer import ZeroShotTransferOptimizer

__all__ = [
    'FEROptimizer',
    'GradientCorrectionOptimizer', 
    'ForwardSignalOptimizer',
    'LocalizedLearningOptimizer',
    'ZeroShotTransferOptimizer'
]

__version__ = '1.0.0'