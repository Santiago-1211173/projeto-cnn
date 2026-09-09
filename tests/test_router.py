"""
Test Script for the Arbitrator Router (CNN vs RL Decision).
Uses Mahalanobis distance to decide whether the CNN is highly confident 
or if the image should be routed to the RL specialist agent.
"""

import sys
import os
import logging
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from src.config import (
    DATA_DIR,
    CHECKPOINT_DIR,
    MAHALANOBIS_PROFILES_PATH,
    OUTPUT_DIR
)
from src.models.custom_cnn import RawModel
from src.data.loader import load_mnist_raw

# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def calculate_mahalanobis(vector_x: np.ndarray, mu: np.ndarray, inv_sigma: np.ndarray) -> float:
    """Mathematical formula for Mahalanobis distance."""
    diff = vector_x - mu
    sq_dist = np.dot(np.dot(diff, inv_sigma), diff.T)
    return float(np.sqrt(sq_dist))

def add_noise(image: np.ndarray, intensity: float = 0.5) -> np.ndarray:
    """Injects noise to simulate a difficult/ambiguous outlier case."""
    noise = np.random.normal(loc=0.0, scale=intensity, size=image.shape)
    noisy_image = image + noise
    return np.clip(noisy_image, 0., 1.)

def main():
    # 1. Load CNN
    model = RawModel()
    ckpt = tf.train.Checkpoint(model=model)
    latest_ckpt = tf.train.latest_checkpoint(CHECKPOINT_DIR)
    if not latest_ckpt:
        logger.error(f"Checkpoint not found in {CHECKPOINT_DIR}. Train CNN first.")
        sys.exit(1)
    ckpt.restore(latest_ckpt).expect_partial()

    # 2. Load Mahalanobis Profiles (Arbitrator brain)
    if not os.path.exists(MAHALANOBIS_PROFILES_PATH):
        logger.error(f"Mahalanobis profiles not found at {MAHALANOBIS_PROFILES_PATH}. Profile clusters first.")
        sys.exit(1)
    island_map = np.load(MAHALANOBIS_PROFILES_PATH, allow_pickle=True)
    logger.info("Mahalanobis profiles loaded successfully.")

    # 3. Load Test Images
    x_test, y_test = load_mnist_raw(DATA_DIR, kind='t10k')
    x_test = x_test.astype(np.float32) / 255.0

    # 4. Select a single image
    idx = 0
    img_clean = x_test[idx]
    
    # Create noisy corrupted version of the same image
    img_corrupted = add_noise(img_clean, intensity=0.6)

    # Prepare batch dimensions for TensorFlow [1, 28, 28, 1]
    batch_clean = tf.convert_to_tensor(np.expand_dims(img_clean, axis=0), dtype=tf.float32)
    batch_corrupted = tf.convert_to_tensor(np.expand_dims(img_corrupted, axis=0), dtype=tf.float32)

    # 5. Extract 128D latent vectors via CNN
    vector_clean = model(batch_clean)["latent_features"].numpy()[0]
    vector_corrupted = model(batch_corrupted)["latent_features"].numpy()[0]

    # 6. Evaluate Mahalanobis distance to nearest class profile
    def evaluate_vector(vector: np.ndarray, label_type: str):
        min_distance = float('inf')
        nearest_digit = -1

        for digit in range(10):
            mu = island_map[str(digit)].item()["mu"]
            inv_sigma = island_map[str(digit)].item()["inv_sigma"]
            
            dist = calculate_mahalanobis(vector, mu, inv_sigma)
            
            if dist < min_distance:
                min_distance = dist
                nearest_digit = digit
                
        logger.info(f"--- {label_type} Image ---")
        logger.info(f"Nearest class centroid is '{nearest_digit}'.")
        logger.info(f"Mahalanobis distance: {min_distance:.2f}")
        return min_distance, nearest_digit

    dist_clean, _ = evaluate_vector(vector_clean, "CLEAN")
    dist_corr, _ = evaluate_vector(vector_corrupted, "CORRUPTED (Noisy)")

    # 7. Plot to verify
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    
    axes[0].imshow(img_clean[:,:,0], cmap='gray')
    axes[0].set_title(f"Clean\nDistance to centroid: {dist_clean:.1f}")
    axes[0].axis('off')
    
    axes[1].imshow(img_corrupted[:,:,0], cmap='gray')
    axes[1].set_title(f"Outlier\nDistance to centroid: {dist_corr:.1f}")
    axes[1].axis('off')
    
    output_path = os.path.join(OUTPUT_DIR, "teste_triagem.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    logger.info(f"\nVerification plot saved to: {output_path}")

if __name__ == "__main__":
    main()