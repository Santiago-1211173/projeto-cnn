"""
CIFAR-100 subpackage initialization.
Exposes core model architecture, AdamW optimizer, and OOD uncertainty arbiter.
"""

from src.cifar100.layers import (
    BatchNorm2DLayer,
    Conv2DLayer,
    DenseLayer,
    GlobalAvgPool2DLayer,
    MaxPool2DLayer,
    ResidualBlock,
)
from src.cifar100.model import RawModelCIFAR100
from src.cifar100.ood_arbiter import DualUncertaintyArbiter
from src.cifar100.optimizers import Adam

__all__ = [
    "RawModelCIFAR100",
    "DualUncertaintyArbiter",
    "Adam",
    "BatchNorm2DLayer",
    "Conv2DLayer",
    "DenseLayer",
    "GlobalAvgPool2DLayer",
    "MaxPool2DLayer",
    "ResidualBlock",
]
