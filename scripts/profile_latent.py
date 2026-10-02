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
