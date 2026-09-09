"""
Production Hybrid Inference Pipeline Demonstration.
Tests the complete system:
Image -> CNN (10D Probabilities) -> Arbitrator (Mahalanobis) -> Decision (CNN or k-NN RL).
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
    MAHALANOBIS_THRESHOLD,
    MAHALANOBIS_PROFILES_PATH,
    MEMORY_BANK_10D_PATH,
    OUTPUT_DIR
)
from src.models.custom_cnn import RawModel
from src.models.knn_bandit_agent import KNNBanditAgent
from src.data.loader import load_mnist_raw

# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def calculate_mahalanobis(vector_x: np.ndarray, mu: np.ndarray, inv_sigma: np.ndarray) -> float:
    """Mathematical formula for Mahalanobis distance."""
    diff = vector_x - mu
    sq_dist = np.dot(np.dot(diff, inv_sigma), diff.T)
    return float(np.sqrt(sq_dist))

def add_noise(image: np.ndarray, intensity: float = 0.6) -> np.ndarray:
    """Injects Gaussian noise into a single image."""
    noise = np.random.normal(loc=0.0, scale=intensity, size=image.shape)
    return np.clip(image + noise, 0., 1.)

def main():
    # 1. Configure Hardware
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    logger.info("==================================================")
    logger.info("INITIALIZING THE HYBRID INFERENCE PIPELINE...")
    logger.info("==================================================")

    # 2. Load CNN
    cnn = RawModel()
    ckpt_cnn = tf.train.Checkpoint(model=cnn)
    latest_ckpt = tf.train.latest_checkpoint(CHECKPOINT_DIR)
    if not latest_ckpt:
        logger.error(f"Checkpoint not found in {CHECKPOINT_DIR}. Train CNN first.")
        sys.exit(1)
    ckpt_cnn.restore(latest_ckpt).expect_partial()
    logger.info("[OK] CNN weights loaded.")

    # 3. Load Arbitrator (Mahalanobis profiles)
    if not os.path.exists(MAHALANOBIS_PROFILES_PATH):
        logger.error(f"Mahalanobis profiles not found at {MAHALANOBIS_PROFILES_PATH}. Run profiling script first.")
        sys.exit(1)
    island_map = np.load(MAHALANOBIS_PROFILES_PATH, allow_pickle=True)
    logger.info("[OK] Arbitrator profiles loaded.")

    # 4. Load Specialist Brain (k-NN Agent)
    agent = KNNBanditAgent(k=30, n_actions=10, latent_dim=10, use_pca=False)
    if not os.path.exists(MEMORY_BANK_10D_PATH):
        logger.error(f"10D Memory bank not found at {MEMORY_BANK_10D_PATH}. Run training script first.")
        sys.exit(1)
    agent.load(MEMORY_BANK_10D_PATH)
    logger.info("[OK] k-NN Specialist Agent loaded.")

    # 5. Load Test Dataset
    x_test, y_test = load_mnist_raw(DATA_DIR, kind='t10k')
    x_test = x_test.astype(np.float32) / 255.0

    # Select 6 images (3 clean, 3 noisy)
    sample_indices = [10, 42, 99]
    test_images = []
    real_labels = []
    types = []

    for idx in sample_indices:
        # Add clean version
        test_images.append(x_test[idx])
        real_labels.append(y_test[idx])
        types.append("Clean")
        
        # Add noisy version of the same image
        test_images.append(add_noise(x_test[idx], intensity=0.6))
        real_labels.append(y_test[idx])
        types.append("Noisy")

    # 6. Pipeline Inference
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    logger.info("\n--- STARTING INFERENCE DEMO ---")
    
    for i in range(len(test_images)):
        img = test_images[i]
        real_label = real_labels[i]
        
        # A. Prepare Tensor
        tensor_img = tf.convert_to_tensor(np.expand_dims(img, axis=0), dtype=tf.float32)
        
        # B. CNN feature extraction
        outputs = cnn(tensor_img)
        vector_128d = outputs["latent_features"]
        prob_cnn = outputs["probabilities"][0]
        state_10d = prob_cnn.numpy()  # Input state for k-NN
        pred_cnn = int(tf.argmax(prob_cnn).numpy())
        
        # C. Arbitrator measures distance in 128D latent space
        vector_np = vector_128d.numpy()[0]
        min_dist = float('inf')
        
        for digit in range(10):
            mu = island_map[str(digit)].item()["mu"]
            inv_sigma = island_map[str(digit)].item()["inv_sigma"]
            dist = calculate_mahalanobis(vector_np, mu, inv_sigma)
            if dist < min_dist:
                min_dist = dist

        # D. Routing Decision
        if min_dist < MAHALANOBIS_THRESHOLD:
            decision_maker = "CNN"
            final_decision = pred_cnn
        else:
            decision_maker = "k-NN RL"
            final_decision = agent.get_action(state_10d, epsilon=0.0)

        # E. Log Results
        success = "V" if final_decision == real_label else "X"
        color = "green" if success == "V" else "red"
        
        logger.info(
            f"Image {i+1} ({types[i]}): MinDist={min_dist:.1f} | "
            f"Decision={decision_maker} | Pred={final_decision} (True={real_label}) [{success}]"
        )

        # F. Plot
        axes[i].imshow(img[:, :, 0], cmap='gray')
        axes[i].set_title(
            f"Type: {types[i]}\nDist: {min_dist:.1f}\nDecider: {decision_maker}\nPred: {final_decision} (True: {real_label})",
            color=color, fontweight='bold'
        )
        axes[i].axis('off')

    plt.tight_layout()
    output_plot_path = os.path.join(OUTPUT_DIR, "resultado_hibrido.png")
    os.makedirs(os.path.dirname(output_plot_path), exist_ok=True)
    plt.savefig(output_plot_path, dpi=300)
    logger.info(f"\n[OK] Final plot saved to: {output_plot_path}")

if __name__ == "__main__":
    main()
