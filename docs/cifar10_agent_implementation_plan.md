# CIFAR-10 Expansion: Agent Implementation Plan

> **Purpose:** Step-by-step instructions for an LLM agent to implement CIFAR-10 support in the projeto-cnn repository.
> **Project Root:** `c:\Users\sanfr\Desktop\projetos-gecad\projeto-cnn`
> **User Decisions Applied:** Move MNIST outputs → `outputs/mnist/`, 4-conv CNN, same hyperparams as MNIST, independent per-channel Gaussian noise, no combined dashboard yet.

---

## PHASE 0: Output Directory Reorganization

**Goal:** Move all existing MNIST-generated outputs into `outputs/mnist/` to establish dataset-scoped output isolation. **No code changes in this phase.**

### Step 0.1 — Create the new directory structure

```bash
mkdir -p outputs/mnist/checkpoints
mkdir -p outputs/mnist/logs
mkdir -p outputs/cifar10/checkpoints
mkdir -p outputs/cifar10/logs
```

### Step 0.2 — Move existing MNIST outputs

Move the following files from `outputs/` into `outputs/mnist/`. Preserve the exact filenames.

**Checkpoint files (move into `outputs/mnist/checkpoints/`):**
- `outputs/checkpoints/` → entire contents → `outputs/mnist/checkpoints/`

**Log files (move into `outputs/mnist/logs/`):**
- `outputs/logs/` → entire contents → `outputs/mnist/logs/`

**Top-level output files (move into `outputs/mnist/`):**
- `outputs/eaai_metrics.csv` → `outputs/mnist/eaai_metrics.csv`
- `outputs/eaai_metrics.json` → `outputs/mnist/eaai_metrics.json`
- `outputs/eaai_evaluation_dashboard.png` → `outputs/mnist/eaai_evaluation_dashboard.png`
- `outputs/train_rl_simulation_log.csv` → `outputs/mnist/train_rl_simulation_log.csv`
- `outputs/mahalanobis_pp_profiles.npz` → `outputs/mnist/mahalanobis_pp_profiles.npz`
- `outputs/mahalanobis_profiles.npz` → `outputs/mnist/mahalanobis_profiles.npz`
- `outputs/knn_memory_bank.npz` → `outputs/mnist/knn_memory_bank.npz`
- `outputs/knn_memory_bank_128d.npz` → `outputs/mnist/knn_memory_bank_128d.npz`
- `outputs/knn_memory_mapping.json` → `outputs/mnist/knn_memory_mapping.json`
- `outputs/rl_agent_weights-1.data-00000-of-00001` → `outputs/mnist/rl_agent_weights-1.data-00000-of-00001`
- `outputs/rl_agent_weights-1.index` → `outputs/mnist/rl_agent_weights-1.index`
- `outputs/checkpoint` → `outputs/mnist/checkpoint`

**PNG visualization files (move into `outputs/mnist/`):**
- All `.png` files in `outputs/` → `outputs/mnist/`

> [!CAUTION]
> Use **copy then delete** (not `mv`) to avoid data loss if the operation fails partway. Verify file counts match before deleting originals.

### Step 0.3 — Verification

```bash
# Verify no data loss: count files
ls outputs/mnist/ | wc -l     # Should be ~28 files + 2 subdirs
ls outputs/mnist/checkpoints/  # Should contain modelo_dissecado-* and rl_agent files
```

**Success Criteria:** All previously generated outputs are in `outputs/mnist/`. The `outputs/` root is clean except for the `mnist/` and `cifar10/` subdirectories (and possibly `__pycache__`). No files were lost.

---

## PHASE 1: Configuration System Update

**Goal:** Update `src/config.py` to support dataset-parameterized paths via a `DATASET` environment variable.

### Step 1.1 — Edit `src/config.py`

**File:** `c:\Users\sanfr\Desktop\projetos-gecad\projeto-cnn\src\config.py`

Replace the ENTIRE file content with the following. This preserves ALL existing constants and adds dataset-aware path resolution:

```python
# src/config.py — Centralized project configuration
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Random seed for reproducibility
RANDOM_SEED = 42

# =============================================================================
# DATASET SELECTION
# =============================================================================
# Set via environment variable DATASET="mnist" or DATASET="cifar10"
# Defaults to "mnist" for full backward compatibility.
DATASET = os.environ.get("DATASET", "mnist").lower()

# Mahalanobis threshold for CNN <-> RL routing
MAHALANOBIS_THRESHOLD = 12.5

# Noise sweep levels for oracle seeding
NOISE_SWEEP = [0.0, 0.2, 0.4, 0.6, 0.8]

# Threshold sweep range for global evaluation
THRESHOLD_SWEEP_START = 5.0
THRESHOLD_SWEEP_END = 30.0
THRESHOLD_SWEEP_STEP = 2.5

# =============================================================================
# DATA PATHS (Dataset-Conditional)
# =============================================================================
MNIST_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "MNIST", "raw")
CIFAR10_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "CIFAR10", "raw")

DATA_DIR = CIFAR10_DATA_DIR if DATASET == "cifar10" else MNIST_DATA_DIR

# =============================================================================
# OUTPUT PATHS (Dataset-Scoped)
# =============================================================================
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", DATASET)
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")

# Agent defaults
KNN_K = 30
KNN_N_ACTIONS = 10

# --- Fase 1: Memoria Episodica ---
MEMORY_CAPACITY = 5000
LATENT_DIM = 128
KNN_K_NEIGHBORS = 30

# --- Fase 2: Agente RL (pre-declarados para evitar imports circulares) ---
RL_STATE_DIM = 5
RL_N_ACTIONS = 4
CURRICULUM_ALPHA_DECAY = 0.995
SLIDING_VALIDATION_BUFFER_SIZE = 100
REPLAY_BUFFER_CAPACITY = 10000

# CNN training defaults (identical for both datasets)
CNN_BATCH_SIZE = 256
CNN_EPOCHS = 10
CNN_LEARNING_RATE = 0.01

# --- Fase 3: Simulacao Online e Caos ---
SIMULATION_STEPS = 50000
SIMULATION_NOISE_RATE = 0.1
SIMULATION_NOISE_LEVEL = 0.6
SIMULATION_LOG_INTERVAL = 500
SIMULATION_CSV_PATH = os.path.join(OUTPUT_DIR, "train_rl_simulation_log.csv")
RL_AGENT_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "rl_agent_phase3.pt")
MAHALANOBIS_PP_PROFILES_PATH = os.path.join(OUTPUT_DIR, "mahalanobis_pp_profiles.npz")

# Memory bank paths
MEMORY_BANK_10D_PATH = os.path.join(OUTPUT_DIR, "knn_memory_bank.npz")
MEMORY_BANK_128D_PATH = os.path.join(OUTPUT_DIR, "knn_memory_bank_128d.npz")
MAHALANOBIS_PROFILES_PATH = os.path.join(OUTPUT_DIR, "mahalanobis_profiles.npz")
MEMORY_MAPPING_PATH = os.path.join(OUTPUT_DIR, "knn_memory_mapping.json")

# =============================================================================
# DATASET METADATA (used by scripts to select model and loader)
# =============================================================================
DATASET_INPUT_SHAPE = {
    "mnist": (28, 28, 1),
    "cifar10": (32, 32, 3),
}

DATASET_NUM_CLASSES = {
    "mnist": 10,
    "cifar10": 10,
}

INPUT_SHAPE = DATASET_INPUT_SHAPE.get(DATASET, (28, 28, 1))
NUM_CLASSES = DATASET_NUM_CLASSES.get(DATASET, 10)
```

### Step 1.2 — Verification

```bash
cd c:\Users\sanfr\Desktop\projetos-gecad\projeto-cnn
python -c "from src.config import DATASET, DATA_DIR, OUTPUT_DIR, INPUT_SHAPE; print(f'DATASET={DATASET}, DATA_DIR={DATA_DIR}, OUTPUT_DIR={OUTPUT_DIR}, INPUT_SHAPE={INPUT_SHAPE}')"
```

**Expected output:** `DATASET=mnist, DATA_DIR=...data\MNIST\raw, OUTPUT_DIR=...outputs\mnist, INPUT_SHAPE=(28, 28, 1)`

```bash
set DATASET=cifar10 && python -c "from src.config import DATASET, DATA_DIR, OUTPUT_DIR, INPUT_SHAPE; print(f'DATASET={DATASET}, DATA_DIR={DATA_DIR}, OUTPUT_DIR={OUTPUT_DIR}, INPUT_SHAPE={INPUT_SHAPE}')"
```

**Expected output:** `DATASET=cifar10, DATA_DIR=...data\CIFAR10\raw, OUTPUT_DIR=...outputs\cifar10, INPUT_SHAPE=(32, 32, 3)`

**Success Criteria:** Both invocations print correct dataset-scoped paths. No import errors. All downstream modules that `from src.config import ...` continue to resolve without modification.

---

## PHASE 2: Data Loading Infrastructure

**Goal:** Add a CIFAR-10 data loader and create a unified loader interface.

### Step 2.1 — Download CIFAR-10 dataset

Create a download script at `scripts/download_cifar10.py`:

```python
"""
Downloads and extracts the CIFAR-10 dataset into data/CIFAR10/raw/.
Uses the official Python/pickle distribution from https://www.cs.toronto.edu/~kriz/cifar.html
"""

import os
import sys
import urllib.request
import tarfile

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
TARGET_DIR = os.path.join(PROJECT_ROOT, "data", "CIFAR10", "raw")

CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
ARCHIVE_NAME = "cifar-10-python.tar.gz"


def main():
    os.makedirs(TARGET_DIR, exist_ok=True)
    archive_path = os.path.join(TARGET_DIR, ARCHIVE_NAME)

    # 1. Download
    if not os.path.exists(archive_path):
        print(f"Downloading CIFAR-10 from {CIFAR10_URL}...")
        urllib.request.urlretrieve(CIFAR10_URL, archive_path)
        print(f"Downloaded to {archive_path}")
    else:
        print(f"Archive already exists: {archive_path}")

    # 2. Extract
    print("Extracting...")
    with tarfile.open(archive_path, 'r:gz') as tar:
        tar.extractall(path=TARGET_DIR)
    print(f"Extracted to {TARGET_DIR}")

    # 3. Verify
    batch_dir = os.path.join(TARGET_DIR, "cifar-10-batches-py")
    expected_files = ["data_batch_1", "data_batch_2", "data_batch_3",
                      "data_batch_4", "data_batch_5", "test_batch", "batches.meta"]
    for f in expected_files:
        path = os.path.join(batch_dir, f)
        if os.path.exists(path):
            print(f"  OK: {f}")
        else:
            print(f"  MISSING: {f}")
            sys.exit(1)

    print("\nCIFAR-10 dataset ready.")


if __name__ == "__main__":
    main()
```

Run it:
```bash
python scripts/download_cifar10.py
```

### Step 2.2 — Create CIFAR-10 data loader

**Create file:** `c:\Users\sanfr\Desktop\projetos-gecad\projeto-cnn\src\data\cifar10_loader.py`

```python
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
```

### Step 2.3 — Update existing `src/data/loader.py` with unified interface

**Edit file:** `c:\Users\sanfr\Desktop\projetos-gecad\projeto-cnn\src\data\loader.py`

**ADD the following functions at the bottom of the file** (after the existing `create_dataset` function). Do NOT modify or remove the existing `load_mnist_raw` or `create_dataset` functions:

```python
# --- Unified Dataset Interface ---

def load_dataset_raw(dataset_name: str, data_dir: str, kind: str = 'train') -> tuple:
    """
    Unified loader dispatching to MNIST or CIFAR-10.

    Args:
        dataset_name: 'mnist' or 'cifar10'.
        data_dir: Path to the raw data directory.
        kind: 'train' or 't10k'.

    Returns:
        Tuple of (images, labels) as numpy arrays.
    """
    if dataset_name == 'cifar10':
        from src.data.cifar10_loader import load_cifar10_raw
        return load_cifar10_raw(data_dir, kind=kind)
    else:
        return load_mnist_raw(data_dir, kind=kind)


def create_dataset_for(dataset_name: str, data_dir: str, batch_size: int = CNN_BATCH_SIZE):
    """
    Unified dataset pipeline factory.

    Args:
        dataset_name: 'mnist' or 'cifar10'.
        data_dir: Path to the raw data directory.
        batch_size: Training batch size.

    Returns:
        tf.data.Dataset pipeline.
    """
    if dataset_name == 'cifar10':
        from src.data.cifar10_loader import create_cifar10_dataset
        return create_cifar10_dataset(data_dir, batch_size=batch_size)
    else:
        return create_dataset(data_dir, batch_size=batch_size)
```

### Step 2.4 — Verification

```bash
cd c:\Users\sanfr\Desktop\projetos-gecad\projeto-cnn

# Test CIFAR-10 loader
python -c "
from src.data.cifar10_loader import load_cifar10_raw
from src.config import CIFAR10_DATA_DIR
imgs, lbls = load_cifar10_raw(CIFAR10_DATA_DIR, kind='train')
print(f'Train shape: {imgs.shape}, labels: {lbls.shape}, dtype: {imgs.dtype}')
imgs_t, lbls_t = load_cifar10_raw(CIFAR10_DATA_DIR, kind='t10k')
print(f'Test shape: {imgs_t.shape}, labels: {lbls_t.shape}')
print(f'Label range: {lbls.min()} - {lbls.max()}')
"
```

**Expected output:**
```
Train shape: (50000, 32, 32, 3), labels: (50000,), dtype: uint8
Test shape: (10000, 32, 32, 3), labels: (10000,)
Label range: 0 - 9
```

**Success Criteria:** CIFAR-10 loads with correct shape `(N, 32, 32, 3)`, 10 classes (0–9), both train and test partitions. Existing MNIST loader still works unchanged.

---

## PHASE 3: CIFAR-10 CNN Model

**Goal:** Create a CIFAR-10 CNN that outputs the IDENTICAL latent contract: `{"latent_features": (N, 128), "probabilities": (N, 10)}`.

### Step 3.1 — Create the CIFAR-10 CNN model

**Create file:** `c:\Users\sanfr\Desktop\projetos-gecad\projeto-cnn\src\models\custom_cnn_cifar10.py`

This file MUST follow these exact patterns from the existing `custom_cnn.py`:
- Use `tf.Module` (NOT `tf.keras.Model`)
- Import from `src.scratch.layers` and `src.scratch.activations`
- Return a `Dict[str, tf.Tensor]` with keys `"latent_features"` and `"probabilities"`
- The `latent_dense` must output exactly 128 dimensions
- The `classifier_dense` must output exactly 10 dimensions

```python
"""
CIFAR-10 CNN Feature Extractor.
A custom Convolutional Neural Network adapted for 32x32x3 RGB images,
built from scratch using the same low-level TensorFlow primitives as the MNIST model.

Maintains the identical 128D latent space contract required by the downstream
episodic memory, Mahalanobis++ OOD detector, and RL agent.
"""

import tensorflow as tf
from typing import Dict
from src.scratch.layers import DenseLayer, Conv2DLayer, MaxPool2DLayer
from src.scratch.activations import relu, softmax


class RawModelCIFAR10(tf.Module):
    def __init__(self, name: str = "custom_cnn_cifar10"):
        super().__init__(name=name)

        # --- CONVOLUTIONAL BLOCK 1 ---
        # Input: 32x32x3 RGB image. Two conv layers before pooling for richer features.
        self.conv1 = Conv2DLayer(in_channels=3, out_channels=32, kernel_size=3, name="conv1")
        # 32x32x3 -> Conv1(3x3,VALID) -> 30x30x32
        self.conv2 = Conv2DLayer(in_channels=32, out_channels=32, kernel_size=3, name="conv2")
        # 30x30x32 -> Conv2(3x3,VALID) -> 28x28x32
        self.pool1 = MaxPool2DLayer(pool_size=2, stride=2, name="pool1")
        # 28x28x32 -> Pool1(2x2) -> 14x14x32

        # --- CONVOLUTIONAL BLOCK 2 ---
        # Deeper features: edges -> textures -> object parts.
        self.conv3 = Conv2DLayer(in_channels=32, out_channels=64, kernel_size=3, name="conv3")
        # 14x14x32 -> Conv3(3x3,VALID) -> 12x12x64
        self.conv4 = Conv2DLayer(in_channels=64, out_channels=64, kernel_size=3, name="conv4")
        # 12x12x64 -> Conv4(3x3,VALID) -> 10x10x64
        self.pool2 = MaxPool2DLayer(pool_size=2, stride=2, name="pool2")
        # 10x10x64 -> Pool2(2x2) -> 5x5x64

        # --- TRANSITION ---
        self.flatten = tf.keras.layers.Flatten()

        # --- DENSE BLOCK (Latent Space and Classification) ---
        # Spatial dimensions after conv/pool chain:
        # 32 -> Conv1 -> 30 -> Conv2 -> 28 -> Pool1 -> 14
        # 14 -> Conv3 -> 12 -> Conv4 -> 10 -> Pool2 -> 5
        # 5 x 5 x 64 = 1600 (same flattened dim as the MNIST model)
        self.latent_dense = DenseLayer(in_features=5 * 5 * 64, out_features=128, name="latent_space")
        self.classifier_dense = DenseLayer(in_features=128, out_features=10, name="classifier")

    def __call__(self, x: tf.Tensor) -> Dict[str, tf.Tensor]:
        # --- Phase 1: Spatial Feature Extraction ---
        x = self.conv1(x)
        x = relu(x)
        x = self.conv2(x)
        x = relu(x)
        x = self.pool1(x)

        x = self.conv3(x)
        x = relu(x)
        x = self.conv4(x)
        x = relu(x)
        x = self.pool2(x)

        # --- Phase 2: Latent Space ---
        x_flat = self.flatten(x)
        raw_latent = self.latent_dense(x_flat)
        latent_features = relu(raw_latent)

        # --- Phase 3: Decision ---
        logits = self.classifier_dense(latent_features)
        probabilities = softmax(logits)

        return {
            "latent_features": latent_features,
            "probabilities": probabilities
        }
```

### Step 3.2 — Verification

```bash
cd c:\Users\sanfr\Desktop\projetos-gecad\projeto-cnn
python -c "
import tensorflow as tf
import numpy as np
from src.models.custom_cnn_cifar10 import RawModelCIFAR10

model = RawModelCIFAR10()
dummy = tf.convert_to_tensor(np.random.rand(2, 32, 32, 3).astype(np.float32))
out = model(dummy)
print(f'latent_features shape: {out[\"latent_features\"].shape}')  # Must be (2, 128)
print(f'probabilities shape: {out[\"probabilities\"].shape}')      # Must be (2, 10)
print(f'probabilities sum: {float(tf.reduce_sum(out[\"probabilities\"][0]))}')  # Must be ~1.0
print(f'Total params: {sum(tf.size(v).numpy() for v in model.trainable_variables)}')
"
```

**Expected output:**
```
latent_features shape: (2, 128)
probabilities shape: (2, 10)
probabilities sum: 1.0
Total params: 271786
```

**Success Criteria:** Output shapes are `(N, 128)` and `(N, 10)`. Softmax sums to 1.0. No import errors.

---

## PHASE 4: Training Script Parameterization

**Goal:** Make all 5 pipeline scripts accept a `--dataset` argument to select between MNIST and CIFAR-10.

### Step 4.1 — Update `scripts/train_cnn.py`

**Edit file:** `c:\Users\sanfr\Desktop\projetos-gecad\projeto-cnn\scripts\train_cnn.py`

Apply the following changes (preserve all existing code structure, comments, and docstrings):

**Change 1 — Add argparse and dataset-aware imports (after line 11, before the existing config imports):**

Add `import argparse` to the imports section.

**Change 2 — Replace the hardcoded config imports with dataset-aware resolution.**

The `main()` function must:
1. Accept `--dataset` argument (`mnist` or `cifar10`, default `mnist`).
2. Set `os.environ["DATASET"]` BEFORE importing from `src.config`.
3. Select the correct model class: `RawModel` for mnist, `RawModelCIFAR10` for cifar10.
4. Select the correct dataset creator: `create_dataset` for mnist, `create_dataset_for` for cifar10.

Here is the full replacement for `scripts/train_cnn.py`:

```python
"""
Main training script for the CNN (Single-GPU).
Trains the custom CNN model using GPU if available.
Supports both MNIST and CIFAR-10 datasets via --dataset argument.
"""

import sys
import os
import time
import datetime
import logging
import argparse
from typing import Tuple
import tensorflow as tf

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from src.scratch.losses import categorical_crossentropy
from src.scratch.optimizers import SGD

# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def calculate_accuracy(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Calculates accuracy by comparing the predicted class with the true class."""
    predicted_classes = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
    correct_predictions = tf.equal(predicted_classes, y_true)
    return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

def main() -> None:
    parser = argparse.ArgumentParser(description="Train the custom CNN model.")
    parser.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "cifar10"],
                        help="Dataset to train on (default: mnist).")
    args = parser.parse_args()

    # Set DATASET env var BEFORE importing config
    os.environ["DATASET"] = args.dataset

    from src.config import (
        CNN_BATCH_SIZE, CNN_EPOCHS, CNN_LEARNING_RATE,
        DATA_DIR, CHECKPOINT_DIR, LOG_DIR, DATASET
    )

    # Select model and dataset loader based on dataset
    if args.dataset == "cifar10":
        from src.models.custom_cnn_cifar10 import RawModelCIFAR10 as ModelClass
        from src.data.cifar10_loader import create_cifar10_dataset
        train_dataset = create_cifar10_dataset(DATA_DIR, batch_size=CNN_BATCH_SIZE)
    else:
        from src.models.custom_cnn import RawModel as ModelClass
        from src.data.loader import create_dataset
        train_dataset = create_dataset(DATA_DIR, batch_size=CNN_BATCH_SIZE)

    # 1. Configure Hardware
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    logger.info("\n=========================================")
    logger.info(f"Starting CNN training [{DATASET.upper()}] on {'GPU 0' if gpus else 'CPU'}")
    logger.info("=========================================\n")

    # 2. Allocation on GPU if available
    device_name = '/GPU:0' if gpus else '/CPU:0'
    with tf.device(device_name):
        # Instantiate Model and Optimizer
        model = ModelClass()
        optimizer = SGD(learning_rate=CNN_LEARNING_RATE)

        # Setup TensorBoard Summary Writer
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        os.makedirs(LOG_DIR, exist_ok=True)
        summary_log_dir = os.path.join(LOG_DIR, current_time)
        summary_writer = tf.summary.create_file_writer(summary_log_dir)

        # 3. Training Step compiled via tf.function
        @tf.function
        def train_step(x_batch: tf.Tensor, y_batch: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
            with tf.GradientTape() as tape:
                outputs = model(x_batch)
                predictions = outputs["probabilities"]
                loss = categorical_crossentropy(y_batch, predictions, CNN_BATCH_SIZE)

            # Backpropagation
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(model.trainable_variables, gradients)

            # Metrics computation
            accuracy = calculate_accuracy(y_batch, predictions)
            return loss, accuracy

        # 4. Training Loop
        logger.info("Starting training loop...")
        for epoch in range(CNN_EPOCHS):
            start_time = time.time()
            total_loss = 0.0
            total_acc = 0.0
            steps = 0

            for step, (x_batch, y_batch) in enumerate(train_dataset):
                loss, acc = train_step(x_batch, y_batch)

                total_loss += float(loss)
                total_acc += float(acc)
                steps += 1

                if step % 50 == 0 and step > 0:
                    logger.info(f"  [Epoch {epoch+1} | Batch {step}] Loss: {float(loss):.4f} | Acc: {float(acc):.4f}")

            # End of epoch metrics
            avg_loss = total_loss / steps
            avg_acc = total_acc / steps
            epoch_time = time.time() - start_time

            logger.info(f"-> END OF EPOCH {epoch+1}: Time: {epoch_time:.2f}s | Avg Loss: {avg_loss:.4f} | Avg Acc: {avg_acc:.4f}\n")

            # Write to TensorBoard
            with summary_writer.as_default():
                tf.summary.scalar('Loss/Train', avg_loss, step=epoch)
                tf.summary.scalar('Accuracy/Train', avg_acc, step=epoch)

        # Save checkpoint at the end of training
        logger.info("Saving model checkpoint...")
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        ckpt = tf.train.Checkpoint(model=model)
        ckpt.save(os.path.join(CHECKPOINT_DIR, "modelo_dissecado"))
        logger.info(f"Training completed! Checkpoint saved to {CHECKPOINT_DIR}")

if __name__ == "__main__":
    main()
```

### Step 4.2 — Update `scripts/profile_latent.py`

**Edit file:** `c:\Users\sanfr\Desktop\projetos-gecad\projeto-cnn\scripts\profile_latent.py`

Apply the same pattern: add `--dataset` argument, set `os.environ["DATASET"]` before importing config, select correct model class and loader.

Key changes:
1. Add `argparse` with `--dataset` argument.
2. Replace `from src.models.custom_cnn import RawModel` → dataset-conditional import.
3. Replace `from src.data.loader import load_mnist_raw` → use `load_dataset_raw` from the unified interface.

```python
"""
Latent Space Profiling Script (Mahalanobis Distance Setup).
Passes the training dataset through the CNN to calculate the geometric center (Mean)
and dispersion (Covariance matrix) of the 10 classes in the 128D latent space.
Supports both MNIST and CIFAR-10 datasets via --dataset argument.
"""

import sys
import os
import logging
import argparse
import numpy as np
import tensorflow as tf

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Profile the CNN latent space.")
    parser.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "cifar10"],
                        help="Dataset to profile (default: mnist).")
    args = parser.parse_args()

    os.environ["DATASET"] = args.dataset

    from src.config import DATA_DIR, CHECKPOINT_DIR, MAHALANOBIS_PROFILES_PATH, DATASET
    from src.data.loader import load_dataset_raw

    if args.dataset == "cifar10":
        from src.models.custom_cnn_cifar10 import RawModelCIFAR10 as ModelClass
    else:
        from src.models.custom_cnn import RawModel as ModelClass

    # 1. Configure Hardware
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    # 2. Load Model and Restore Weights
    logger.info(f"Loading CNN model [{DATASET.upper()}]...")
    model = ModelClass()
    ckpt = tf.train.Checkpoint(model=model)

    latest_ckpt = tf.train.latest_checkpoint(CHECKPOINT_DIR)
    if not latest_ckpt:
        logger.error(f"Checkpoint not found in {CHECKPOINT_DIR}. Train the model first!")
        return
    ckpt.restore(latest_ckpt).expect_partial()
    logger.info("Weights restored successfully.")

    # 3. Load Training Dataset
    logger.info("Loading training dataset for extraction...")
    x_train, y_train = load_dataset_raw(args.dataset, DATA_DIR, kind='train')

    num_samples = 20000
    x_train = x_train[:num_samples].astype(np.float32) / 255.0
    y_train = y_train[:num_samples]

    # 4. Batch Feature Extraction
    logger.info("Extracting 128D latent features...")
    batch_size = 500
    features_list = []

    for i in range(0, len(x_train), batch_size):
        batch_x = x_train[i : i + batch_size]
        batch_tensor = tf.convert_to_tensor(batch_x)

        outputs = model(batch_tensor)
        features_list.append(outputs["latent_features"].numpy())

    all_features = np.vstack(features_list)

    # 5. Calculate Centroid and Covariance for each class
    logger.info("Calculating centroid and covariance for each class...")

    mahalanobis_profiles = {}

    for digit in range(10):
        indices = np.where(y_train == digit)[0]
        digit_features = all_features[indices]

        mu = np.mean(digit_features, axis=0)
        sigma = np.cov(digit_features, rowvar=False)

        epsilon = 1e-6
        sigma += np.eye(sigma.shape[0]) * epsilon

        inv_sigma = np.linalg.inv(sigma)

        mahalanobis_profiles[str(digit)] = {
            "mu": mu,
            "inv_sigma": inv_sigma
        }
        logger.info(f"  -> Profile for class {digit} mapped successfully.")

    # 6. Save profiles to disk
    os.makedirs(os.path.dirname(MAHALANOBIS_PROFILES_PATH), exist_ok=True)
    np.savez_compressed(MAHALANOBIS_PROFILES_PATH, **mahalanobis_profiles)

    logger.info("==================================================")
    logger.info(f"SUCCESS! Mahalanobis profiles saved to: {MAHALANOBIS_PROFILES_PATH}")
    logger.info("==================================================")

if __name__ == "__main__":
    main()
```

### Step 4.3 — Update `scripts/train_rl.py`

Apply the same `--dataset` pattern:
1. Add `--dataset` argument.
2. Set `os.environ["DATASET"]` before config import.
3. Use `load_dataset_raw()` instead of `load_mnist_raw()`.
4. Select correct model class.

The key changes to the existing file (preserve all other logic):

- Add `parser.add_argument("--dataset", ...)` alongside existing `--latent_dim` and `--k`.
- Set `os.environ["DATASET"] = args.dataset` right after parsing.
- Move all `from src.config import ...` and `from src.models... import ...` to AFTER `os.environ` is set.
- Replace `load_mnist_raw(DATA_DIR, kind='train')` with `load_dataset_raw(args.dataset, DATA_DIR, kind='train')`.
- Replace `cnn = RawModel()` with dataset-conditional model instantiation.

### Step 4.4 — Update `training/train_rl_online_simulation.py`

This is the most complex file. The changes needed are:

1. **In `OnlineStreamPipeline.__init__`** (line 345–363):
   - Accept `dataset_name` parameter (default: `"mnist"`).
   - Select model class based on dataset.
   - Use `load_dataset_raw()` instead of `load_mnist_raw()`.

2. **In `TrainRLOnlineSimulation.__init__`** (line 423+):
   - Accept and pass through `dataset_name`.

3. **In `run_simulation` and `main`** (line 777+):
   - Accept `--dataset` argument.
   - Set `os.environ["DATASET"]`.

**Critical constraint:** The `inject_noise()` function (line 83–104) already works for ANY shape — it uses `size=batch_float.shape`, so it automatically handles both `(N, 28, 28, 1)` and `(N, 32, 32, 3)`. **DO NOT modify `inject_noise()`.**

### Step 4.5 — Update `evaluate_hybrid_global.py`

Apply the same pattern:
1. Add `--dataset` argument to `main()`.
2. Set `os.environ["DATASET"]` before config imports.
3. Select correct model class and data loader.
4. The `OnlineStreamPipeline` must receive the dataset parameter.

### Step 4.6 — Verification

```bash
# Verify MNIST pipeline still works with explicit --dataset flag
python scripts/train_cnn.py --dataset mnist --help
# Should show --dataset argument

# Verify CIFAR-10 model loads correctly
set DATASET=cifar10 && python -c "
from src.models.custom_cnn_cifar10 import RawModelCIFAR10
import tensorflow as tf, numpy as np
m = RawModelCIFAR10()
o = m(tf.convert_to_tensor(np.random.rand(1,32,32,3).astype(np.float32)))
print('OK:', o['latent_features'].shape, o['probabilities'].shape)
"
```

**Success Criteria:** All scripts accept `--dataset` flag. Running with `--dataset mnist` produces identical behavior to the original. No existing tests are broken.

---

## PHASE 5: CIFAR-10 Full Pipeline Execution

**Goal:** Run the complete 5-step pipeline for CIFAR-10.

### Step 5.1 — Train CIFAR-10 CNN

```bash
python scripts/train_cnn.py --dataset cifar10
```

**Expected:** ~10 epochs (same as MNIST). Final accuracy should be ~65-75% (CIFAR-10 is harder). Checkpoint saved to `outputs/cifar10/checkpoints/`.

### Step 5.2 — Profile CIFAR-10 latent space

```bash
python scripts/profile_latent.py --dataset cifar10
```

**Expected:** Mahalanobis profiles saved to `outputs/cifar10/mahalanobis_profiles.npz`.

### Step 5.3 — Seed episodic memory

```bash
python scripts/train_rl.py --dataset cifar10 --latent_dim 128 --k 30
```

**Expected:** Memory bank saved to `outputs/cifar10/knn_memory_bank_128d.npz`.

### Step 5.4 — Online RL simulation

```bash
python -m training.train_rl_online_simulation --dataset cifar10
```

**Expected:** RL agent trained and saved to `outputs/cifar10/checkpoints/rl_agent_phase3.pt`. Mahalanobis++ profiles saved.

### Step 5.5 — 5-baseline evaluation

```bash
python evaluate_hybrid_global.py --dataset cifar10 --samples-per-level 1000
```

**Expected outputs in `outputs/cifar10/`:**
- `eaai_metrics.json`
- `eaai_metrics.csv`
- `eaai_evaluation_dashboard.png`

**Success Criteria:** All 5 steps complete without errors. CIFAR-10 metrics JSON and dashboard are generated. MNIST outputs in `outputs/mnist/` remain untouched.

---

## PHASE 6: Documentation Update

**Goal:** Update documentation to reflect the dual-dataset capability.

### Step 6.1 — Create CIFAR-10 baseline comparison

**Create file:** `c:\Users\sanfr\Desktop\projetos-gecad\projeto-cnn\docs\results\baseline-comparison-cifar10.md`

Structure: Follow the exact same format as `baseline-comparison.md` but with CIFAR-10 results. Reference the CIFAR-10 metrics from `outputs/cifar10/eaai_metrics.json`.

### Step 6.2 — Create cross-dataset analysis

**Create file:** `c:\Users\sanfr\Desktop\projetos-gecad\projeto-cnn\docs\results\cross-dataset-analysis.md`

This document should:
1. Present a side-by-side MNIST vs CIFAR-10 results table.
2. Analyze whether the RL advantage (B4 vs B2/B3) transfers across datasets.
3. Discuss the impact of RGB color channels on noise resilience.
4. Validate the generalization thesis.

### Step 6.3 — Update documentation index

**Edit:** `docs/results/README.md` — Add links to the new CIFAR-10 and cross-dataset documents.
**Edit:** `docs/README.md` — Add CIFAR-10 entry to the documentation map table.

**Success Criteria:** All documentation links resolve. New documents follow the same academic tone, formatting, and navigation structure as existing docs.

---

## CRITICAL RULES FOR THE EXECUTING AGENT

1. **NEVER modify files in `src/models/knn_bandit_agent.py`, `src/models/rl_agent.py`, or `src/models/reward_manager.py`.** These are dataset-agnostic and must remain untouched.

2. **NEVER modify `src/scratch/` files** (layers.py, activations.py, losses.py, optimizers.py). These are the from-scratch TF primitives.

3. **Always set `os.environ["DATASET"]` BEFORE importing from `src.config`** in any script that uses the `--dataset` flag. Config resolves paths at import time.

4. **Preserve all existing MNIST results.** Files in `outputs/mnist/` must never be overwritten or deleted.

5. **The `inject_noise()` function already handles any input shape.** Do NOT modify it. It uses `size=batch_float.shape` which works for both `(N, 28, 28, 1)` and `(N, 32, 32, 3)`.

6. **The 128D latent contract is sacred.** Both CNN models MUST output `{"latent_features": (N, 128), "probabilities": (N, 10)}`.

7. **Use the SAME hyperparameters for CIFAR-10 as MNIST:** 10 epochs, lr=0.01, batch_size=256, same noise sweep [0.0, 0.2, 0.4, 0.6, 0.8].

8. **Run `python -c "..."` verification checks after each phase** before proceeding to the next.

9. **All existing comments and docstrings unrelated to the changes must be preserved.**

10. **Execute phases sequentially.** Each phase depends on artifacts from the previous one.
