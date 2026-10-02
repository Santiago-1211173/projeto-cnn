"""
CIFAR-10 data loader matching the MNIST loader API.
Reads the official Python pickle batch files.
"""

import os
import pickle
import numpy as np
import tensorflow as tf
from src.config import CNN_BATCH_SIZE


def _unpickle(file_path: str) -> dict:
    """Reads a single CIFAR-10 batch file."""
    with open(file_path, 'rb') as f:
        data_dict = pickle.load(f, encoding='bytes')
    return data_dict


def load_cifar10_raw(data_dir: str, kind: str = 'train') -> tuple:
    """
    Reads the CIFAR-10 dataset from the pickle batch files.

    Args:
        data_dir: Path to the directory containing 'cifar-10-batches-py/'.
        kind: 'train' for training data (50,000 images) or 't10k' for test data (10,000 images).

    Returns:
        Tuple of (images, labels):
            - images: np.ndarray of shape (N, 32, 32, 3), dtype np.uint8
            - labels: np.ndarray of shape (N,), dtype np.uint8
    """
    batch_dir = os.path.join(data_dir, "cifar-10-batches-py")

    if kind == 'train':
        # Concatenate all 5 training batches
        images_list = []
        labels_list = []
        for i in range(1, 6):
            batch_path = os.path.join(batch_dir, f"data_batch_{i}")
            batch = _unpickle(batch_path)
            images_list.append(batch[b'data'])
            labels_list.append(batch[b'labels'])

        images_flat = np.concatenate(images_list, axis=0)
        labels = np.concatenate(labels_list, axis=0).astype(np.uint8)
    elif kind == 't10k':
        batch_path = os.path.join(batch_dir, "test_batch")
        batch = _unpickle(batch_path)
        images_flat = batch[b'data']
        labels = np.array(batch[b'labels'], dtype=np.uint8)
    else:
        raise ValueError(f"Unknown kind '{kind}'. Use 'train' or 't10k'.")

    # CIFAR-10 stores images as flat (N, 3072) in R, G, B channel order.
    # Reshape to (N, 3, 32, 32) then transpose to (N, 32, 32, 3) for TensorFlow NHWC.
    images = images_flat.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype(np.uint8)

    return images, labels


def create_cifar10_dataset(data_dir: str, batch_size: int = CNN_BATCH_SIZE):
    """Creates an optimized dataset pipeline for CIFAR-10."""
    X_train, y_train = load_cifar10_raw(data_dir, kind='train')

    # Normalization to [0.0, 1.0]
    X_train = X_train.astype(np.float32) / 255.0
    y_train = y_train.astype(np.int32)

    dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    return dataset.shuffle(10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
