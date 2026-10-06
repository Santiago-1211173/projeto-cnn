import os
import struct
import numpy as np
import tensorflow as tf
from src.config import CNN_BATCH_SIZE

def load_mnist_raw(data_dir: str, kind: str = 'train') -> tuple:
    """Reads the original binary files of MNIST dataset."""
    labels_path = os.path.join(data_dir, f'{kind}-labels-idx1-ubyte')
    images_path = os.path.join(data_dir, f'{kind}-images-idx3-ubyte')

    # Read Labels
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)

    # Read Images
    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 28, 28, 1)

    return images, labels

def create_dataset(data_dir: str, batch_size: int = CNN_BATCH_SIZE):
    """Creates an optimized dataset pipeline based on raw bytes."""
    X_train, y_train = load_mnist_raw(data_dir, kind='train')
    
    # Normalization
    X_train = X_train.astype(np.float32) / 255.0
    y_train = y_train.astype(np.int32)
    
    dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    return dataset.shuffle(10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)


# --- Unified Dataset Interface ---

def load_dataset_raw(dataset_name: str, data_dir: str, kind: str = 'train') -> tuple:
    """
    Unified loader dispatching to MNIST, CIFAR-10, or CIFAR-100.

    Args:
        dataset_name: 'mnist', 'cifar10', or 'cifar100'.
        data_dir: Path to the raw data directory.
        kind: 'train' or 't10k'.

    Returns:
        Tuple of (images, labels) as numpy arrays.
    """
    if dataset_name == 'cifar100':
        from src.data.cifar100_loader import load_cifar100_raw
        return load_cifar100_raw(data_dir, kind=kind)
    elif dataset_name == 'cifar10':
        from src.data.cifar10_loader import load_cifar10_raw
        return load_cifar10_raw(data_dir, kind=kind)
    else:
        return load_mnist_raw(data_dir, kind=kind)


def create_dataset_for(dataset_name: str, data_dir: str, batch_size: int = CNN_BATCH_SIZE):
    """
    Unified dataset pipeline factory.

    Args:
        dataset_name: 'mnist', 'cifar10', or 'cifar100'.
        data_dir: Path to the raw data directory.
        batch_size: Training batch size.

    Returns:
        tf.data.Dataset pipeline.
    """
    if dataset_name == 'cifar100':
        from src.data.cifar100_loader import create_cifar100_dataset
        return create_cifar100_dataset(data_dir, batch_size=batch_size)
    elif dataset_name == 'cifar10':
        from src.data.cifar10_loader import create_cifar10_dataset
        return create_cifar10_dataset(data_dir, batch_size=batch_size)
    else:
        return create_dataset(data_dir, batch_size=batch_size)