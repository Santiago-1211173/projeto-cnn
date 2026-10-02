"""
CIFAR-10 Isolated Module.
Zero-regression architecture for CIFAR-10 active semiparametric learning.
"""
from src.cifar10.model import RawModelCIFAR10
from src.cifar10.optimizers import Adam
from src.cifar10.ood_arbiter import DualUncertaintyArbiter
from src.cifar10.layers import (
    Conv2DLayer,
    BatchNorm2DLayer,
    MaxPool2DLayer,
    DenseLayer,
    ResidualBlock,
    GlobalAvgPool2DLayer,
)

__all__ = [
    "RawModelCIFAR10",
    "Adam",
    "DualUncertaintyArbiter",
    "Conv2DLayer",
    "BatchNorm2DLayer",
    "MaxPool2DLayer",
    "DenseLayer",
    "ResidualBlock",
    "GlobalAvgPool2DLayer",
]
