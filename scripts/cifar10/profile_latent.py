"""
Latent Space Profiling Script for CIFAR-10 Dual Uncertainty Arbiter.
Passes clean CIFAR-10 training data through the trained RawModelCIFAR10 backbone,
extracts invariant 128D latent bottleneck representations and softmax probabilities,
fits class-conditional Ledoit-Wolf shrinkage precision matrices, and calibrates
dual uncertainty thresholds (Mahalanobis distance + Shannon entropy) at the 95th percentile.

Outputs:
    outputs/cifar10/mahalanobis_pp_profiles.npz
"""
import sys
import os
import time
import logging
import argparse
from typing import Tuple
import numpy as np
import tensorflow as tf

# Ensure project root in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.loader import load_dataset_raw
from src.cifar10.model import RawModelCIFAR10
from src.cifar10.ood_arbiter import DualUncertaintyArbiter

# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)


def extract_features_and_probabilities(
    model: RawModelCIFAR10,
    x_data: np.ndarray,
    batch_size: int = 500
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts 128D latent features and softmax probabilities across batches.
    
    Args:
        model: Restored RawModelCIFAR10 instance.
        x_data: Normalized input array of shape (N, 32, 32, 3), float32 in [0, 1].
        batch_size: Batch size for forward passes.
        
    Returns:
        Tuple of (latent_features: (N, 128), probabilities: (N, 10))
    """
    n_samples = len(x_data)
    features_list = []
    probs_list = []

    for i in range(0, n_samples, batch_size):
        batch_x = x_data[i : i + batch_size]
        batch_tensor = tf.convert_to_tensor(batch_x, dtype=tf.float32)
        out = model(batch_tensor, training=False)
        features_list.append(out["latent_features"].numpy())
        probs_list.append(out["probabilities"].numpy())

    all_features = np.vstack(features_list)
    all_probs = np.vstack(probs_list)
    return all_features, all_probs


def main():
    parser = argparse.ArgumentParser(description="Profile CIFAR-10 latent space and calibrate Dual Uncertainty Arbiter.")
    parser.add_argument("--checkpoint-dir", type=str, default="outputs/cifar10/checkpoints",
                        help="Directory containing trained CIFAR-10 model checkpoints.")
    parser.add_argument("--output-path", type=str, default="outputs/cifar10/mahalanobis_pp_profiles.npz",
                        help="Path to save calibrated profiles and thresholds.")
    parser.add_argument("--data-dir", type=str, default="data/CIFAR10/raw",
                        help="Directory containing CIFAR-10 raw binary files.")
    parser.add_argument("--num-samples", type=int, default=50000,
                        help="Number of clean training samples to profile (default: 50000).")
    parser.add_argument("--batch-size", type=int, default=500,
                        help="Batch size for feature extraction (default: 500).")
    parser.add_argument("--percentile", type=float, default=95.0,
                        help="Calibration percentile for in-distribution thresholds (default: 95.0).")
    args = parser.parse_args()

    # Configure GPU memory growth if available
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                pass

    # 1. Load Model and Restore Latest Checkpoint
    checkpoint_path = os.path.abspath(args.checkpoint_dir)
    logger.info(f"Looking for latest checkpoint in: {checkpoint_path}")
    latest_ckpt = tf.train.latest_checkpoint(checkpoint_path)
    if not latest_ckpt:
        raise FileNotFoundError(f"No checkpoint found in {checkpoint_path}. Train CIFAR-10 model first!")

    logger.info(f"Restoring model weights from: {latest_ckpt}")
    model = RawModelCIFAR10()
    ckpt = tf.train.Checkpoint(model=model)
    ckpt.restore(latest_ckpt).expect_partial()
    logger.info("Model weights successfully restored.")

    # 2. Load Clean Training Dataset
    data_path = os.path.abspath(args.data_dir)
    logger.info(f"Loading CIFAR-10 training set from {data_path}...")
    x_train, y_train = load_dataset_raw("cifar10", data_path, kind="train")

    num_samples = min(args.num_samples, len(x_train))
    logger.info(f"Profiling using {num_samples} clean in-distribution training samples...")
    x_train = x_train[:num_samples].astype(np.float32) / 255.0
    y_train = y_train[:num_samples].astype(np.int32)

    # 3. Extract 128D Latent Embeddings & Softmax Probabilities
    start_time = time.time()
    logger.info("Extracting 128D latent bottleneck features and predictive probabilities...")
    latent_features, probabilities = extract_features_and_probabilities(model, x_train, batch_size=args.batch_size)
    extraction_time = time.time() - start_time
    logger.info(f"Feature extraction completed in {extraction_time:.2f}s ({num_samples / extraction_time:.1f} samples/sec).")
    logger.info(f"Latent shape: {latent_features.shape}, Probabilities shape: {probabilities.shape}")

    # 4. Fit Dual Uncertainty Arbiter (Centroids + Ledoit-Wolf Shrinkage)
    logger.info("Fitting class-conditional centroids and Ledoit-Wolf shrinkage precision matrices...")
    arbiter = DualUncertaintyArbiter(n_classes=10, latent_dim=128)
    fit_start = time.time()
    arbiter.fit(latent_features, probabilities, y_train, percentile=args.percentile)
    fit_time = time.time() - fit_start
    logger.info(f"Ledoit-Wolf shrinkage fitting completed in {fit_time:.2f}s.")

    # 5. Display Calibration Summary
    logger.info("==================================================")
    logger.info("DUAL UNCERTAINTY ARBITER CALIBRATION SUMMARY")
    logger.info(f"  In-Distribution Calibration Percentile: {args.percentile:.1f}%")
    logger.info(f"  Calibrated Mahalanobis Threshold (tau_M): {arbiter.threshold_mahalanobis:.4f}")
    logger.info(f"  Calibrated Shannon Entropy Threshold (tau_H): {arbiter.threshold_entropy:.4f}")
    logger.info("--------------------------------------------------")

    for c in range(10):
        mu_norm = np.linalg.norm(arbiter.profiles[c]["mu"])
        cond_num = np.linalg.cond(arbiter.profiles[c]["precision"])
        logger.info(f"  Class {c:02d}: Centroid L2 Norm = {mu_norm:.4f}, Precision Cond = {cond_num:.2f}")

    # 6. Evaluate Clean Calibration Rates
    clean_dists = arbiter.compute_mahalanobis_batch(latent_features)
    clean_ents = arbiter.compute_entropy_batch(probabilities)
    mah_rejections = np.mean(clean_dists > arbiter.threshold_mahalanobis) * 100.0
    ent_rejections = np.mean(clean_ents > arbiter.threshold_entropy) * 100.0
    dual_rejections = np.mean((clean_dists > arbiter.threshold_mahalanobis) | (clean_ents > arbiter.threshold_entropy)) * 100.0

    logger.info("--------------------------------------------------")
    logger.info(f"  Clean Training Rejection (Mahalanobis alone): {mah_rejections:.2f}% (Target: ~5.0%)")
    logger.info(f"  Clean Training Rejection (Entropy alone):     {ent_rejections:.2f}% (Target: ~5.0%)")
    logger.info(f"  Clean Training Rejection (Dual Combined):     {dual_rejections:.2f}%")

    # 7. Verification under Synthetic Noise Drift (Sensitivity Check)
    logger.info("Evaluating sensitivity under synthetic Gaussian noise drift...")
    x_noise_sample = x_train[:1000]
    for sigma in [0.2, 0.4, 0.8]:
        noisy_x = np.clip(x_noise_sample + np.random.normal(0.0, sigma, x_noise_sample.shape).astype(np.float32), 0.0, 1.0)
        n_feats, n_probs = extract_features_and_probabilities(model, noisy_x, batch_size=args.batch_size)
        n_dists = arbiter.compute_mahalanobis_batch(n_feats)
        n_ents = arbiter.compute_entropy_batch(n_probs)
        n_dual_rej = np.mean((n_dists > arbiter.threshold_mahalanobis) | (n_ents > arbiter.threshold_entropy)) * 100.0
        logger.info(f"  Noise sigma = {sigma:.1f} -> Mean d_M: {np.mean(n_dists):.2f}, Mean Entropy: {np.mean(n_ents):.2f}, OOD Rejection Rate: {n_dual_rej:.2f}%")

    # 8. Save Profiles
    output_file = os.path.abspath(args.output_path)
    logger.info(f"Saving calibrated arbiter profiles to: {output_file}")
    arbiter.save(output_file)

    # 9. Verify Saved File Reload
    arbiter_check = DualUncertaintyArbiter()
    arbiter_check.load(output_file)
    assert arbiter_check.is_fitted, "Arbiter failed to report is_fitted=True after reload!"
    assert np.isclose(arbiter_check.threshold_mahalanobis, arbiter.threshold_mahalanobis), "Threshold mismatch on reload!"
    assert np.isclose(arbiter_check.threshold_entropy, arbiter.threshold_entropy), "Threshold mismatch on reload!"

    logger.info("==================================================")
    logger.info(f"SUCCESS! CIFAR-10 Dual Uncertainty Arbiter saved to: {output_file}")
    logger.info("Verification gate: PASSED")
    logger.info("==================================================")


if __name__ == "__main__":
    main()
