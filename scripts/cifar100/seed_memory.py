"""
Episodic Memory Seeding and k-NN Tuning Script for CIFAR-100.
Anchored in Deep Semiparametric Learning (Jain & Lindsey, ICLR 2018; Pritzel et al., ICML 2017).

Extracts invariant latent representations from the converged CIFAR-100 CNN backbone (ResNet-14)
for clean in-distribution training exemplars (balanced across all 100 classes).
Fits the Dual Uncertainty Arbiter (Ledoit-Wolf shrinkage and predictive Shannon entropy),
populates the KNNBanditAgent128D episodic memory bank, executes systematic k-NN hyperparameter
tuning across candidate k in [1, 3, 5, 7, 10, 15, 20, 30], evaluates real-time query latency
and noise stress robustness, and serializes artifacts to outputs/cifar100/.
"""

from __future__ import annotations
import sys
import os
import time
import logging
import argparse
from typing import Tuple, Dict, Any, List
import numpy as np

# Ensure project root in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import CIFAR100_DATA_DIR, RANDOM_SEED, MEMORY_CAPACITY, LATENT_DIM

# Ensure Python with TensorFlow is used if launched from base Anaconda
try:
    import tensorflow as tf
except ModuleNotFoundError:
    tf_python = r"C:\Users\sanfr\.conda\envs\tf_l40s\python.exe"
    if os.path.exists(tf_python) and sys.executable.lower() != tf_python.lower():
        import subprocess
        res = subprocess.run([tf_python] + sys.argv)
        sys.exit(res.returncode)
    raise

from src.data.cifar100_loader import load_cifar100_raw
from src.cifar100.model import load_cifar100_backbone, CIFAR100_MEAN, CIFAR100_STD
from src.cifar100.ood_arbiter import DualUncertaintyArbiter
from src.models.knn_bandit_agent import KNNBanditAgent128D

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def extract_features_batch(
    model: Any,
    images: np.ndarray,
    batch_size: int = 500,
    standardize: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts latent features, predicted labels, and softmax probabilities.

    Args:
        model: Restored CIFAR-100 CNN backbone.
        images: Normalized image array of shape (N, 32, 32, 3) in [0.0, 1.0].
        batch_size: Forward pass chunk size.
        standardize: Whether to apply CIFAR-100 channel Z-Score standardization.

    Returns:
        Tuple of:
            - latent_features: Array of shape (N, latent_dim)
            - predicted_labels: Array of shape (N,)
            - probabilities: Array of shape (N, 100)
    """
    n_samples = len(images)
    features_list: List[np.ndarray] = []
    probs_list: List[np.ndarray] = []
    preds_list: List[np.ndarray] = []

    for i in range(0, n_samples, batch_size):
        batch_x = images[i : i + batch_size]
        batch_tensor = tf.convert_to_tensor(batch_x, dtype=tf.float32)
        if standardize:
            batch_tensor = (batch_tensor - CIFAR100_MEAN) / CIFAR100_STD
        out = model(batch_tensor, training=False)
        latent = out["latent_features"].numpy()
        probs = out["probabilities"].numpy()
        preds = np.argmax(probs, axis=1)

        features_list.append(latent)
        probs_list.append(probs)
        preds_list.append(preds)

    all_features = np.vstack(features_list).astype(np.float32)
    all_probs = np.vstack(probs_list).astype(np.float32)
    all_preds = np.concatenate(preds_list).astype(np.int32)

    return all_features, all_preds, all_probs


def inject_noise_batch(images: np.ndarray, noise_level: float) -> np.ndarray:
    """Injects zero-mean Gaussian noise into an image batch, clipping to [0.0, 1.0]."""
    if noise_level <= 0.0:
        return images.copy()
    noise = np.random.normal(loc=0.0, scale=noise_level, size=images.shape)
    return np.clip(images + noise, 0.0, 1.0).astype(np.float32)


def select_balanced_prototypes(
    images: np.ndarray,
    labels: np.ndarray,
    num_prototypes: int,
    n_classes: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Selects class-balanced prototypes across all n_classes.
    """
    samples_per_class = max(1, num_prototypes // n_classes)
    selected_indices: List[int] = []

    for c in range(n_classes):
        class_idxs = np.where(labels == c)[0]
        if len(class_idxs) < samples_per_class:
            chosen = class_idxs
        else:
            chosen = class_idxs[:samples_per_class]
        selected_indices.extend(chosen)

    # If rounding leaves room for extra samples, fill from remaining
    if len(selected_indices) < num_prototypes:
        remaining = np.setdiff1d(np.arange(len(labels)), selected_indices)
        needed = num_prototypes - len(selected_indices)
        selected_indices.extend(remaining[:needed])

    selected_indices = np.array(selected_indices[:num_prototypes], dtype=np.int32)
    return images[selected_indices], labels[selected_indices]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Seed CIFAR-100 Episodic Memory and perform k-NN hyperparameter tuning."
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="outputs/cifar100/checkpoints",
        help="Directory containing converged CIFAR-100 model checkpoints.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/cifar100",
        help="Directory to save memory bank and arbiter profiles.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=CIFAR100_DATA_DIR,
        help="Directory containing CIFAR-100 raw pickle files.",
    )
    parser.add_argument(
        "--num-prototypes",
        type=int,
        default=5000,
        help="Number of clean reference prototypes to seed (default: 5000).",
    )
    parser.add_argument(
        "--capacity",
        type=int,
        default=5000,
        help="Total episodic memory capacity (default: 5000).",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Primary number of nearest neighbors for retrieval (default: 10).",
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=128,
        help="Latent representation dimension (default: 128).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Batch size for forward feature extraction (default: 500).",
    )
    parser.add_argument(
        "--eval-samples",
        type=int,
        default=1000,
        help="Number of held-out test samples for k-NN evaluation (default: 1000).",
    )
    parser.add_argument(
        "--skip-tuning",
        action="store_true",
        help="Skip the k-NN hyperparameter tuning grid search.",
    )
    args = parser.parse_args()

    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    profiles_path = os.path.join(output_dir, "arbiter_profiles.npz")
    memory_bank_path = os.path.join(output_dir, "knn_memory_bank.npz")

    logger.info("=" * 70)
    logger.info("CIFAR-100 EPISODIC MEMORY SEEDING & DUAL UNCERTAINTY CALIBRATION")
    logger.info("Theoretical Basis: Jain & Lindsey (ICLR 2018); Pritzel et al. (ICML 2017)")
    logger.info("=" * 70)
    logger.info(f"Target Memory Prototypes: {args.num_prototypes:,}")
    logger.info(f"Memory Capacity:          {args.capacity:,}")
    logger.info(f"Latent Dimension:         {args.latent_dim}")
    logger.info(f"Selected k (Neighbors):   {args.k}")
    logger.info(f"Arbiter Profiles Path:    {profiles_path}")
    logger.info(f"Memory Bank Output Path:  {memory_bank_path}")

    # 1. Hardware Configuration
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                pass

    # 2. Load Converged CIFAR-100 Model
    checkpoint_path = os.path.abspath(args.checkpoint_dir)
    logger.info(f"Restoring converged feature extractor from: {checkpoint_path}")
    model, standardize = load_cifar100_backbone(checkpoint_path, latent_dim=args.latent_dim)
    logger.info(
        f"Backbone weights successfully restored ({args.latent_dim}D Latent Contract, Standardize: {standardize})."
    )

    # 3. Load Clean CIFAR-100 Training Dataset
    data_path = os.path.abspath(args.data_dir)
    logger.info(f"Loading training data from: {data_path}")
    x_train_raw, y_train_raw, x_test_raw, y_test_raw = load_cifar100_raw(data_dir=data_path)

    clean_subset_x, clean_subset_y = select_balanced_prototypes(
        x_train_raw, y_train_raw, num_prototypes=args.num_prototypes, n_classes=100
    )

    unique_classes, counts = np.unique(clean_subset_y, return_counts=True)
    logger.info(
        f"Prototype subset extracted: {len(clean_subset_x)} samples across {len(unique_classes)} classes."
    )
    assert len(unique_classes) == 100, f"Expected 100 classes in prototype subset, got {len(unique_classes)}"

    # 4. Extract Latent Representations for Prototypes
    logger.info(f"Extracting {args.latent_dim}D latent embeddings for {args.num_prototypes} prototypes...")
    t0_extract = time.time()
    latent_prototypes, cnn_preds, cnn_probs = extract_features_batch(
        model, clean_subset_x, batch_size=args.batch_size, standardize=standardize
    )
    extract_time = time.time() - t0_extract
    cnn_acc_train = np.mean(cnn_preds == clean_subset_y) * 100.0

    logger.info(
        f"Extracted {len(latent_prototypes):,} prototypes in {extract_time:.2f}s "
        f"({len(latent_prototypes) / max(0.001, extract_time):.1f} samples/s)."
    )
    logger.info(f"Latent feature shape: {latent_prototypes.shape} (Contract: (N, {args.latent_dim}))")
    logger.info(f"CNN Nominal Accuracy on Prototype Subset: {cnn_acc_train:.2f}%")

    assert latent_prototypes.shape == (args.num_prototypes, args.latent_dim), (
        f"Shape mismatch: expected ({args.num_prototypes}, {args.latent_dim}), got {latent_prototypes.shape}"
    )
    assert not np.isnan(latent_prototypes).any(), "NaN values detected in latent features!"
    assert not np.isinf(latent_prototypes).any(), "Inf values detected in latent features!"

    # 5. Fit & Calibrate Dual Uncertainty Arbiter (Ledoit-Wolf for 100 classes)
    logger.info("Fitting and calibrating Dual Uncertainty Arbiter (100 classes)...")
    t0_arb = time.time()
    arbiter = DualUncertaintyArbiter(n_classes=100, latent_dim=args.latent_dim)
    arbiter.fit(latent_prototypes, cnn_probs, clean_subset_y, percentile=95.0)
    arb_time = time.time() - t0_arb
    logger.info(
        f"Arbiter fitted in {arb_time:.2f}s | "
        f"tau_M (Mahalanobis): {arbiter.threshold_mahalanobis:.2f} | "
        f"tau_H (Entropy): {arbiter.threshold_entropy:.2f}"
    )

    # Save Arbiter Profiles
    arbiter.save(profiles_path)
    logger.info(f"Dual Uncertainty Arbiter profiles saved to: {profiles_path}")

    # 6. Populate KNNBanditAgent128D Episodic Memory Bank
    logger.info(
        f"Instantiating KNNBanditAgent128D (capacity={args.capacity}, k={args.k}, dim={args.latent_dim}, actions=100)..."
    )
    memory_bank = KNNBanditAgent128D(
        capacity=args.capacity,
        k=args.k,
        latent_dim=args.latent_dim,
        n_actions=100,
    )

    t0_seed = time.time()
    memory_bank.add_experience_batch(
        states=latent_prototypes,
        actions=clean_subset_y,
        rewards=np.ones(args.num_prototypes, dtype=np.float32),
    )
    seed_time = time.time() - t0_seed

    stats = memory_bank.get_memory_stats()
    logger.info(
        f"Memory bank seeded in {seed_time:.4f}s: {stats['size']}/{stats['capacity']} experiences "
        f"({stats['occupancy_pct']:.1f}% occupancy)."
    )
    assert memory_bank.size == args.num_prototypes, (
        f"Expected memory size {args.num_prototypes}, got {memory_bank.size}"
    )

    # Save Memory Bank to Disk
    memory_bank.save(memory_bank_path)
    file_size_mb = os.path.getsize(memory_bank_path) / (1024 * 1024)
    logger.info(f"Episodic memory bank successfully saved: {memory_bank_path} ({file_size_mb:.2f} MB)")

    # 7. Systematic k-NN Hyperparameter Tuning & Scientific Validation
    if not args.skip_tuning:
        logger.info("\n" + "=" * 70)
        logger.info("k-NN HYPERPARAMETER SWEEP & SCIENTIFIC VALIDATION (CIFAR-100)")
        logger.info("=" * 70)

        n_eval = min(args.eval_samples, len(x_test_raw))
        test_eval_x = x_test_raw[:n_eval].astype(np.float32)
        test_eval_y = y_test_raw[:n_eval].astype(np.int32)

        logger.info(f"Extracting {args.latent_dim}D embeddings for {n_eval} held-out test queries...")
        test_latents, cnn_test_preds, cnn_test_probs = extract_features_batch(
            model, test_eval_x, batch_size=args.batch_size, standardize=standardize
        )
        cnn_test_acc = np.mean(cnn_test_preds == test_eval_y) * 100.0
        logger.info(f"CNN Baseline Accuracy on Test Subset: {cnn_test_acc:.2f}%\n")

        candidate_ks = [1, 3, 5, 7, 10, 15, 20, 30]
        results_grid = []

        logger.info(
            f"{'k':>4} | {'Top-1 Accuracy':>15} | {'Latency (ms/query)':>20} | {'Throughput (q/s)':>18} | {'Relative vs CNN':>17}"
        )
        logger.info("-" * 82)

        best_k = args.k
        best_acc = 0.0

        for cand_k in candidate_ks:
            memory_bank.k = cand_k
            t0_query = time.time()
            knn_preds = memory_bank.get_action_batch(test_latents, epsilon=0.0)
            query_time = time.time() - t0_query

            acc = np.mean(knn_preds == test_eval_y) * 100.0
            latency_ms = (query_time / n_eval) * 1000.0
            throughput = n_eval / max(1e-6, query_time)
            delta_cnn = acc - cnn_test_acc

            if acc > best_acc:
                best_acc = acc
                best_k = cand_k

            marker = " *" if cand_k == args.k else ""
            logger.info(
                f"{cand_k:>4} | {acc:>14.2f}% | {latency_ms:>19.3f} ms | {throughput:>16.1f} q/s | {delta_cnn:>+16.2f}%{marker}"
            )
            results_grid.append({
                "k": cand_k,
                "accuracy": acc,
                "latency_ms": latency_ms,
                "throughput": throughput,
            })

        logger.info("-" * 82)
        logger.info(f"Selected hyperparameter: k={args.k}")
        logger.info(f"Highest accuracy in sweep: k={best_k} ({best_acc:.2f}%)")

        # 8. Noise Stress Testing
        logger.info("\n" + "=" * 70)
        logger.info(f"NOISE STRESS TESTING: CNN vs. k-NN EPISODIC RETRIEVAL (k={args.k})")
        logger.info("=" * 70)
        memory_bank.k = args.k

        noise_levels = [0.0, 0.2, 0.4, 0.6]
        logger.info(
            f"{'Noise (sigma)':>14} | {'CNN Accuracy':>15} | {'k-NN Accuracy':>18} | {'Memory Advantage':>18}"
        )
        logger.info("-" * 75)

        for sigma in noise_levels:
            noisy_test_x = inject_noise_batch(test_eval_x, noise_level=sigma)
            noisy_latents, noisy_cnn_preds, _ = extract_features_batch(
                model, noisy_test_x, batch_size=args.batch_size
            )
            noisy_knn_preds = memory_bank.get_action_batch(noisy_latents, epsilon=0.0)

            acc_cnn_noisy = np.mean(noisy_cnn_preds == test_eval_y) * 100.0
            acc_knn_noisy = np.mean(noisy_knn_preds == test_eval_y) * 100.0
            delta_adv = acc_knn_noisy - acc_cnn_noisy

            logger.info(
                f"{sigma:>14.1f} | {acc_cnn_noisy:>14.2f}% | {acc_knn_noisy:>17.2f}% | {delta_adv:>+17.2f}%"
            )
        logger.info("-" * 75)

    memory_bank.k = args.k
    logger.info("\nStep 2.3 Episodic memory seeding and Arbiter calibration completed successfully.")


if __name__ == "__main__":
    main()
