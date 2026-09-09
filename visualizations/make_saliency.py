"""
Saliency Map Visualization Script (XAI / Interpretability).
Calculates gradients of predictions with respect to input pixels to identify exactly where the CNN is focusing its attention.
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
    OUTPUT_DIR
)
from src.models.custom_cnn import RawModel
from src.data.loader import load_mnist_raw

# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def compute_saliency_map(model: tf.Module, image: tf.Tensor) -> tuple:
    """Calculates saliency map using gradients of the predicted class score with respect to input pixels."""
    image_tensor = tf.convert_to_tensor(image, dtype=tf.float32)
    
    with tf.GradientTape() as tape:
        tape.watch(image_tensor)
        outputs = model(image_tensor)
        probabilities = outputs["probabilities"]
        
        predicted_class_idx = tf.argmax(probabilities[0])
        winning_score = probabilities[0, predicted_class_idx]

    # Calculate gradient of winning class score w.r.t input pixels
    gradients = tape.gradient(winning_score, image_tensor)
    
    saliency = tf.abs(gradients)
    
    # Normalize heatmap to [0, 1] range for visual presentation
    saliency_max = tf.reduce_max(saliency)
    saliency_min = tf.reduce_min(saliency)
    saliency_normalized = (saliency - saliency_min) / (saliency_max - saliency_min + 1e-10)
    
    return saliency_normalized[0].numpy(), int(predicted_class_idx.numpy())

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
        sys.exit(1)
    ckpt.restore(latest_ckpt).expect_partial()

    # 3. Load Test Images
    logger.info("Loading test dataset...")
    x_test, y_test = load_mnist_raw(DATA_DIR, kind='t10k')
    x_test = x_test.astype(np.float32) / 255.0

    # 4. Generate Saliency Maps for the first 5 images
    num_images_to_show = 5
    plt.figure(figsize=(15, 3 * num_images_to_show))
    
    logger.info("Calculating saliency maps...")
    for i in range(num_images_to_show):
        img_input = np.expand_dims(x_test[i], axis=0)
        real_label = y_test[i]
        
        saliency_map, predicted_label = compute_saliency_map(model, img_input)
        
        img_2d = img_input[0, :, :, 0]
        saliency_2d = saliency_map[:, :, 0]

        # A) Original Image
        ax1 = plt.subplot(num_images_to_show, 3, i * 3 + 1)
        ax1.imshow(img_2d, cmap='gray')
        ax1.set_title(f"Original (True Label: {real_label})")
        ax1.axis('off')

        # B) Raw Heatmap (Saliency)
        ax2 = plt.subplot(num_images_to_show, 3, i * 3 + 2)
        ax2.imshow(saliency_2d, cmap='hot')
        ax2.set_title(f"CNN Attention Focus (Pred: {predicted_label})")
        ax2.axis('off')

        # C) Overlay (Heatmap projected onto original image)
        ax3 = plt.subplot(num_images_to_show, 3, i * 3 + 3)
        ax3.imshow(img_2d, cmap='gray')
        ax3.imshow(saliency_2d, cmap='hot', alpha=0.6)
        ax3.set_title("XAI Overlay")
        ax3.axis('off')

    output_path = os.path.join(OUTPUT_DIR, "mapa_saliencia.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    
    logger.info("==================================================")
    logger.info(f"Saliency maps successfully saved to: {output_path}")
    logger.info("==================================================")

if __name__ == "__main__":
    main()
