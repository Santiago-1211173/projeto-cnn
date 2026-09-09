"""
Latent Space Profiling Script (Mahalanobis Distance Setup).
Passes the training dataset through the CNN to calculate the geometric center (Mean) 
and dispersion (Covariance matrix) of the 10 digit classes in the 128D latent space.
"""

import sys
import os
import logging
import numpy as np
import tensorflow as tf

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from src.config import (
    DATA_DIR,
    CHECKPOINT_DIR,
    MAHALANOBIS_PROFILES_PATH
)
from src.models.custom_cnn import RawModel
from src.data.loader import load_mnist_raw

# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def main():
    # 1. Configure Hardware
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    # 2. Load Model and Restore Weights
    logger.info("Loading CNN model...")
    model = RawModel()
    ckpt = tf.train.Checkpoint(model=model)
    
    latest_ckpt = tf.train.latest_checkpoint(CHECKPOINT_DIR)
    if not latest_ckpt:
        logger.error(f"Checkpoint not found in {CHECKPOINT_DIR}. Train the model first!")
        return
    ckpt.restore(latest_ckpt).expect_partial()
    logger.info("Weights restored successfully.")

    # 3. Load Training Dataset (used to define classes boundaries)
    logger.info("Loading training dataset for extraction...")
    x_train, y_train = load_mnist_raw(DATA_DIR, kind='train')
    
    # Use a representative sample to prevent high memory usage during covariance computation
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

    # 5. Calculate Centroid and Covariance for each digit class
    logger.info("Calculating centroid and covariance for each digit class...")
    
    mahalanobis_profiles = {}
    
    for digit in range(10):
        # 5.1. Isolate vectors belonging to this digit
        indices = np.where(y_train == digit)[0]
        digit_features = all_features[indices]
        
        # 5.2. Class center (mean for each of the 128 dimensions)
        mu = np.mean(digit_features, axis=0)
        
        # 5.3. Covariance matrix (128x128)
        sigma = np.cov(digit_features, rowvar=False)
        
        # 5.4. Numerical stability: Add a tiny epsilon to the diagonal to make it invertible
        epsilon = 1e-6
        sigma += np.eye(sigma.shape[0]) * epsilon
        
        # 5.5. Inverse of Covariance
        inv_sigma = np.linalg.inv(sigma)
        
        mahalanobis_profiles[str(digit)] = {
            "mu": mu,
            "inv_sigma": inv_sigma
        }
        logger.info(f"  -> Profile for digit {digit} mapped successfully.")

    # 6. Save profiles to disk
    os.makedirs(os.path.dirname(MAHALANOBIS_PROFILES_PATH), exist_ok=True)
    np.savez_compressed(MAHALANOBIS_PROFILES_PATH, **mahalanobis_profiles)
    
    logger.info("==================================================")
    logger.info(f"SUCCESS! Mahalanobis profiles saved to: {MAHALANOBIS_PROFILES_PATH}")
    logger.info("==================================================")

if __name__ == "__main__":
    main()
