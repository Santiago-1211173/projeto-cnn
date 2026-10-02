"""
Unit and Acceptance Tests for CIFAR-10 and Unified Data Loading Infrastructure.
"""

import os
import sys
import numpy as np
import tensorflow as tf

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import CIFAR10_DATA_DIR, MNIST_DATA_DIR
from src.data.cifar10_loader import load_cifar10_raw, create_cifar10_dataset
from src.data.loader import load_dataset_raw, create_dataset_for, load_mnist_raw, create_dataset


def test_cifar10_raw_shapes():
    print("Testing CIFAR-10 raw loader shapes and dtypes...")
    x_train, y_train = load_cifar10_raw(CIFAR10_DATA_DIR, kind='train')
    assert x_train.shape == (50000, 32, 32, 3), f"Expected (50000, 32, 32, 3), got {x_train.shape}"
    assert y_train.shape == (50000,), f"Expected (50000,), got {y_train.shape}"
    assert x_train.dtype == np.uint8, f"Expected uint8, got {x_train.dtype}"
    assert set(np.unique(y_train)) == set(range(10)), "Expected classes 0 to 9"

    x_test, y_test = load_cifar10_raw(CIFAR10_DATA_DIR, kind='t10k')
    assert x_test.shape == (10000, 32, 32, 3), f"Expected (10000, 32, 32, 3), got {x_test.shape}"
    assert y_test.shape == (10000,), f"Expected (10000,), got {y_test.shape}"
    print("  -> CIFAR-10 raw loader passed.")


def test_cifar10_dataset_pipeline():
    print("Testing CIFAR-10 tf.data pipeline...")
    dataset = create_cifar10_dataset(CIFAR10_DATA_DIR, batch_size=64)
    for x_batch, y_batch in dataset.take(1):
        assert x_batch.shape == (64, 32, 32, 3), f"Expected (64, 32, 32, 3), got {x_batch.shape}"
        assert y_batch.shape == (64,), f"Expected (64,), got {y_batch.shape}"
        assert float(tf.reduce_min(x_batch)) >= 0.0, "Min value below 0.0"
        assert float(tf.reduce_max(x_batch)) <= 1.0, "Max value above 1.0"
        assert x_batch.dtype == tf.float32, f"Expected float32, got {x_batch.dtype}"
        assert y_batch.dtype == tf.int32, f"Expected int32, got {y_batch.dtype}"
    print("  -> CIFAR-10 pipeline passed.")


def test_unified_interface():
    print("Testing Unified Loader Interface...")
    c_imgs, c_lbls = load_dataset_raw('cifar10', CIFAR10_DATA_DIR, kind='train')
    m_imgs, m_lbls = load_dataset_raw('mnist', MNIST_DATA_DIR, kind='train')
    assert c_imgs.shape == (50000, 32, 32, 3)
    assert m_imgs.shape == (60000, 28, 28, 1)

    c_ds = create_dataset_for('cifar10', CIFAR10_DATA_DIR, batch_size=32)
    m_ds = create_dataset_for('mnist', MNIST_DATA_DIR, batch_size=32)
    for cx, cy in c_ds.take(1):
        assert cx.shape == (32, 32, 32, 3)
    for mx, my in m_ds.take(1):
        assert mx.shape == (32, 28, 28, 1)
    print("  -> Unified interface passed.")


def test_legacy_mnist_compatibility():
    print("Testing Legacy MNIST Compatibility...")
    imgs, lbls = load_mnist_raw(MNIST_DATA_DIR, kind='train')
    assert imgs.shape == (60000, 28, 28, 1)
    ds = create_dataset(MNIST_DATA_DIR, batch_size=32)
    for bx, by in ds.take(1):
        assert bx.shape == (32, 28, 28, 1)
    print("  -> Legacy MNIST loader passed.")


if __name__ == "__main__":
    print("=" * 70)
    print("RUNNING DATA LOADER TESTS")
    print("=" * 70)
    test_cifar10_raw_shapes()
    test_cifar10_dataset_pipeline()
    test_unified_interface()
    test_legacy_mnist_compatibility()
    print("=" * 70)
    print("ALL DATA LOADER TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 70)
