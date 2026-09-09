"""
Latent Space Collapse Visualization (t-SNE).
Compares how the CNN clusters clean images (ideal scenario) 
vs. how it scatters noisy images (chaotic scenario).
"""

import sys
import os
import logging
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from src.config import (
    DATA_DIR,
    CHECKPOINT_DIR,
    OUTPUT_DIR
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

    logger.info("==================================================")
    logger.info("STARTING VISUAL ANALYSIS (COMPARATIVE t-SNE)")
    logger.info("==================================================")

    # 2. Load CNN model
    cnn = RawModel()
    ckpt_cnn = tf.train.Checkpoint(model=cnn)
    latest_ckpt = tf.train.latest_checkpoint(CHECKPOINT_DIR)
    if not latest_ckpt:
        logger.error(f"Checkpoint not found in {CHECKPOINT_DIR}. Train the CNN first.")
        sys.exit(1)
    ckpt_cnn.restore(latest_ckpt).expect_partial()
    logger.info("[OK] CNN weights loaded.")

    # 3. Load test data (2000 images to run t-SNE in reasonable time)
    x_test, y_test = load_mnist_raw(DATA_DIR, kind='t10k')
    
    x_test_clean = x_test[:2000].astype(np.float32) / 255.0
    labels = y_test[:2000]
    
    # Generate noisy versions
    noise = np.random.normal(loc=0.0, scale=0.6, size=x_test_clean.shape)
    x_test_noisy = np.clip(x_test_clean + noise, 0., 1.)
    logger.info("[OK] Test data prepared (Clean and Noisy versions).")

    # 4. Extract 128D latent vectors
    logger.info("Extracting latent space features via CNN...")
    outputs_clean = cnn(tf.convert_to_tensor(x_test_clean, dtype=tf.float32))
    vectors_clean = outputs_clean["latent_features"].numpy()

    outputs_noisy = cnn(tf.convert_to_tensor(x_test_noisy, dtype=tf.float32))
    vectors_noisy = outputs_noisy["latent_features"].numpy()

    # 5. Apply t-SNE reductions
    logger.info("Calculating t-SNE for Clean Scenario...")
    tsne_clean = TSNE(n_components=2, random_state=42, perplexity=30)
    features_2d_clean = tsne_clean.fit_transform(vectors_clean)

    logger.info("Calculating t-SNE for Noisy Scenario...")
    tsne_noisy = TSNE(n_components=2, random_state=42, perplexity=30)
    features_2d_noisy = tsne_noisy.fit_transform(vectors_noisy)

    # 6. Plotting
    logger.info("Generating plots...")
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    sns.set_theme(style="whitegrid")
    
    colors = sns.color_palette("tab10", 10)

    # Plot A: Clean Scenario
    sns.scatterplot(
        ax=axes[0],
        x=features_2d_clean[:, 0], y=features_2d_clean[:, 1], 
        hue=labels, palette=colors, legend=False, alpha=0.7, s=40
    )
    axes[0].set_title("Scenario A: CNN in Ideal Conditions (Clean Images)\nClear clusters -> Easy classification for CNN", fontsize=14, fontweight='bold')
    axes[0].set_xlabel("t-SNE 1")
    axes[0].set_ylabel("t-SNE 2")

    # Plot B: Noisy Scenario
    sns.scatterplot(
        ax=axes[1],
        x=features_2d_noisy[:, 0], y=features_2d_noisy[:, 1], 
        hue=labels, palette=colors, legend="full", alpha=0.7, s=40
    )
    axes[1].set_title("Scenario B: CNN under Noise Effects\nCollapsed clusters -> Triggers Arbitrator and RL Specialist Agent", fontsize=14, fontweight='bold')
    axes[1].set_xlabel("t-SNE 1")
    axes[1].set_ylabel("t-SNE 2")
    axes[1].legend(title="True Digit", bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, "colapso_latente_tsne.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    logger.info(f"==================================================")
    logger.info(f"[SUCCESS] Plot saved to: {output_path}")
    logger.info("==================================================")

if __name__ == "__main__":
    main()
