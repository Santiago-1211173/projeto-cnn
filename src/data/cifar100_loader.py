"""
CIFAR-100 data loader.
Reads official Python pickle files (train, test, meta) with fine_labels (100 classes).
"""

import os
import pickle
from typing import Optional, Tuple, Union, List, Any
import numpy as np
from src.config import CIFAR100_DATA_DIR, CNN_BATCH_SIZE


def _find_cifar100_file(data_dir: str, filename: str) -> str:
    """
    Locates a CIFAR-100 batch file either directly in data_dir
    or in the nested 'cifar-100-python' subdirectory.
    """
    direct_path = os.path.join(data_dir, filename)
    if os.path.exists(direct_path):
        return direct_path

    nested_path = os.path.join(data_dir, "cifar-100-python", filename)
    if os.path.exists(nested_path):
        return nested_path

    raise FileNotFoundError(
        f"Could not find CIFAR-100 file '{filename}' in '{data_dir}' or '{nested_path}'. "
        "Please run 'python scripts/cifar100/download_cifar100.py' to download the dataset."
    )


def _unpickle(file_path: str) -> dict:
    """Reads a single CIFAR-100 batch file using byte encoding."""
    with open(file_path, "rb") as f:
        data_dict = pickle.load(f, encoding="bytes")
    return data_dict


def _parse_batch(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parses a CIFAR-100 batch file into normalized images and int32 fine labels.

    Returns:
        images: np.ndarray of shape (N, 32, 32, 3), float32 in [0, 1]
        labels: np.ndarray of shape (N,), int32 in [0, 99]
    """
    batch = _unpickle(file_path)
    images_flat = batch[b"data"]
    fine_labels = batch[b"fine_labels"]

    # Reshape from flat (N, 3072) -> (N, 3, 32, 32) -> (N, 32, 32, 3) NHWC
    images = (
        images_flat.reshape(-1, 3, 32, 32)
        .transpose(0, 2, 3, 1)
        .astype(np.float32)
        / 255.0
    )
    labels = np.array(fine_labels, dtype=np.int32)
    return images, labels


def load_cifar100_raw(
    data_dir: Optional[str] = None,
    kind: Optional[str] = None,
) -> Union[
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
]:
    """
    Loads raw CIFAR-100 data from pickle files.

    Args:
        data_dir: Directory containing CIFAR-100 files or subfolder 'cifar-100-python'.
                  Defaults to CIFAR100_DATA_DIR from src.config.
        kind: Optional partition selector ('train', 't10k', 'test').
              If None, returns all 4 partitions: (X_train, y_train, X_test, y_test).

    Returns:
        If kind is None:
            (X_train, y_train, X_test, y_test)
        If kind == 'train':
            (X_train, y_train)
        If kind in ('t10k', 'test'):
            (X_test, y_test)
    """
    if data_dir is None:
        data_dir = CIFAR100_DATA_DIR

    train_path = _find_cifar100_file(data_dir, "train")
    test_path = _find_cifar100_file(data_dir, "test")

    if kind == "train":
        return _parse_batch(train_path)
    elif kind in ("t10k", "test"):
        return _parse_batch(test_path)
    elif kind is None:
        X_train, y_train = _parse_batch(train_path)
        X_test, y_test = _parse_batch(test_path)
        return X_train, y_train, X_test, y_test
    else:
        raise ValueError(
            f"Unknown kind '{kind}'. Use 'train', 'test', 't10k', or None."
        )


def load_cifar100_meta(data_dir: Optional[str] = None) -> List[str]:
    """
    Loads the 100 fine label class names from CIFAR-100 'meta' file.
    """
    if data_dir is None:
        data_dir = CIFAR100_DATA_DIR

    meta_path = _find_cifar100_file(data_dir, "meta")
    meta_dict = _unpickle(meta_path)
    raw_names = meta_dict[b"fine_label_names"]
    return [name.decode("utf-8") for name in raw_names]


def create_cifar100_dataset(
    data_dir: Optional[str] = None,
    batch_size: int = CNN_BATCH_SIZE,
) -> Any:
    """
    Creates an optimized tf.data.Dataset pipeline for CIFAR-100 training data.
    """
    import tensorflow as tf
    X_train, y_train = load_cifar100_raw(data_dir=data_dir, kind="train")
    dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    return dataset.shuffle(10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
