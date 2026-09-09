"""
Visualization of the Decision-Making Process of the k-NN RL Agent.

Generates two main visualization profiles:
1. Comparative Confidence Profile — Bar charts comparing CNN probabilities and k-NN expected rewards.
2. Correction Flow Diagram — Heatmap of reclassifications and accuracy comparisons showing the k-NN agent correcting CNN mistakes.
"""

import sys
import os
import logging
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from src.config import (
    DATA_DIR,
    CHECKPOINT_DIR,
    MEMORY_BANK_10D_PATH,
    OUTPUT_DIR
)
from src.models.custom_cnn import RawModel
from src.models.knn_bandit_agent import KNNBanditAgent
from src.data.loader import load_mnist_raw

# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Colors for digits 0-9
DIGIT_COLORS = [
    '#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6',
    '#1ABC9C', '#E67E22', '#34495E', '#E91E63', '#00BCD4'
]

def load_system():
    """Loads CNN and k-NN Agent."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    # CNN
    cnn = RawModel()
    ckpt = tf.train.Checkpoint(model=cnn)
    latest = tf.train.latest_checkpoint(CHECKPOINT_DIR)
    if not latest:
        logger.error(f"CNN Checkpoint not found in {CHECKPOINT_DIR}. Train CNN first.")
        sys.exit(1)
    ckpt.restore(latest).expect_partial()

    # k-NN Agent
    agent = KNNBanditAgent(k=15, n_actions=10, latent_dim=10, use_pca=False)
    if not os.path.exists(MEMORY_BANK_10D_PATH):
        logger.error(f"10D Memory bank not found at {MEMORY_BANK_10D_PATH}. Train RL first.")
        sys.exit(1)
    agent.load(MEMORY_BANK_10D_PATH)

    return cnn, agent

def add_noise(image: np.ndarray, intensity: float = 0.6) -> np.ndarray:
    """Injects Gaussian noise into a single image."""
    noise = np.random.normal(loc=0.0, scale=intensity, size=image.shape)
    return np.clip(image + noise, 0., 1.)

def find_ambiguous_images(x_test, y_test, cnn, n=8):
    """Finds test images where the CNN has low confidence or is incorrect."""
    candidates = []
    batch_size = 500
    for i in range(0, len(x_test), batch_size):
        batch = x_test[i : i + batch_size]
        batch_t = tf.convert_to_tensor(batch, dtype=tf.float32)
        outputs = cnn(batch_t)
        probs = outputs["probabilities"].numpy()

        for j in range(len(batch)):
            idx_global = i + j
            top_prob = np.max(probs[j])
            pred = np.argmax(probs[j])
            real = y_test[idx_global]

            # Ambiguous: low confidence OR incorrect prediction
            if top_prob < 0.7 or pred != real:
                candidates.append({
                    'idx': idx_global,
                    'confidence': top_prob,
                    'pred_cnn': pred,
                    'real': real,
                    'error': pred != real
                })

    # Prioritize incorrect predictions with lower confidence
    candidates.sort(key=lambda c: (not c['error'], c['confidence']))
    return candidates[:n]

def visualize_confidence_profiles(x_test, y_test, cnn, agent, output_dir):
    """
    Visualization 1: Comparative Confidence Profiles.
    Displays side-by-side probabilities from CNN and expected rewards from k-NN.
    """
    logger.info("\n--- Generating Confidence Profiles (CNN vs k-NN) ---")

    # Find ambiguous noisy images
    x_test_noisy = np.array([add_noise(img) for img in x_test[:2000]])
    labels_noisy = y_test[:2000]
    candidates = find_ambiguous_images(x_test_noisy, labels_noisy, cnn, n=8)

    if len(candidates) == 0:
        logger.warning("No ambiguous images found!")
        return

    n_imgs = min(len(candidates), 8)
    fig, axes = plt.subplots(n_imgs, 3, figsize=(18, 4 * n_imgs))
    if n_imgs == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle('Confidence Profile: CNN vs k-NN Bandit', fontsize=18, fontweight='bold', y=0.98)
    digits = list(range(10))

    for row, cand in enumerate(candidates[:n_imgs]):
        idx = cand['idx']
        img = x_test_noisy[idx]
        label_real = labels_noisy[idx]

        # Feed to CNN
        tensor_img = tf.convert_to_tensor(np.expand_dims(img, 0), dtype=tf.float32)
        outputs = cnn(tensor_img)
        probs_cnn = outputs["probabilities"].numpy()[0]
        pred_cnn = int(np.argmax(probs_cnn))

        # Query k-NN Agent
        expected_rewards = agent.get_expected_rewards(probs_cnn)
        pred_knn = int(np.argmax(expected_rewards))

        # --- Col 1: Noisy Image ---
        ax_img = axes[row, 0]
        ax_img.imshow(img[:, :, 0], cmap='gray')
        title_color = 'green' if pred_knn == label_real else 'red'
        ax_img.set_title(
            f"True: {label_real}\nCNN: {pred_cnn} | k-NN: {pred_knn}",
            fontsize=11, fontweight='bold', color=title_color
        )
        ax_img.axis('off')

        # --- Col 2: CNN Class Probabilities ---
        ax_cnn = axes[row, 1]
        bar_colors_cnn = ['#3498DB'] * 10
        bar_colors_cnn[pred_cnn] = '#E74C3C'  # Highlight CNN choice
        if label_real < 10:
            bar_colors_cnn[label_real] = '#2ECC71'  # Correct target in green

        bars_cnn = ax_cnn.bar(digits, probs_cnn, color=bar_colors_cnn,
                              edgecolor='white', linewidth=0.5, alpha=0.85)
        ax_cnn.set_title('CNN: Probabilities (10D)', fontsize=11, fontweight='bold')
        ax_cnn.set_xlabel('Digit')
        ax_cnn.set_ylabel('Probability')
        ax_cnn.set_xticks(digits)
        ax_cnn.set_ylim(0, max(probs_cnn) * 1.25)
        ax_cnn.axhline(y=0, color='gray', linewidth=0.5)

        for bar, val in zip(bars_cnn, probs_cnn):
            if val > 0.01:
                ax_cnn.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{val:.2f}', ha='center', va='bottom', fontsize=8)

        # --- Col 3: k-NN Expected Rewards ---
        ax_knn = axes[row, 2]
        bar_colors_knn = ['#F39C12'] * 10
        bar_colors_knn[pred_knn] = '#E74C3C'  # Highlight k-NN choice
        if label_real < 10:
            bar_colors_knn[label_real] = '#2ECC71'  # Correct target in green

        bars_knn = ax_knn.bar(digits, expected_rewards, color=bar_colors_knn,
                              edgecolor='white', linewidth=0.5, alpha=0.85)
        ax_knn.set_title('k-NN: Expected Reward', fontsize=11, fontweight='bold')
        ax_knn.set_xlabel('Digit (Action)')
        ax_knn.set_ylabel('Expected Reward')
        ax_knn.set_xticks(digits)
        y_min = min(0, min(expected_rewards) * 1.3)
        y_max = max(expected_rewards) * 1.3 if max(expected_rewards) > 0 else 0.5
        ax_knn.set_ylim(y_min, y_max)
        ax_knn.axhline(y=0, color='gray', linewidth=0.8, linestyle='--')

        for bar, val in zip(bars_knn, expected_rewards):
            if abs(val) > 0.01:
                offset = 0.02 if val >= 0 else -0.06
                ax_knn.text(bar.get_x() + bar.get_width()/2, bar.get_height() + offset,
                           f'{val:+.2f}', ha='center', va='bottom', fontsize=8)

    # Global Legend
    legend_patches = [
        mpatches.Patch(color='#2ECC71', label='True Class'),
        mpatches.Patch(color='#E74C3C', label='Model Prediction'),
        mpatches.Patch(color='#3498DB', label='CNN (Others)'),
        mpatches.Patch(color='#F39C12', label='k-NN (Others)'),
    ]
    fig.legend(handles=legend_patches, loc='lower center', ncol=4,
               fontsize=11, frameon=True, bbox_to_anchor=(0.5, 0.0))

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    path = os.path.join(output_dir, "perfil_confianca_cnn_vs_knn.png")
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    logger.info(f"[OK] Confidence profiles saved to: {path}")

def visualize_correction_flow(x_test, y_test, cnn, agent, output_dir):
    """
    Visualization 2: Correction Flow Map.
    Tracks where k-NN agent overwrites the CNN's predictions.
    """
    logger.info("\n--- Generating Correction Flow Heatmap ---")

    # Evaluate noisy test images
    x_test_noisy = np.array([add_noise(img, 0.6) for img in x_test[:5000]])
    labels = y_test[:5000]

    batch_size = 500
    all_probs = []
    all_preds_cnn = []

    for i in range(0, len(x_test_noisy), batch_size):
        batch = x_test_noisy[i : i + batch_size]
        batch_t = tf.convert_to_tensor(batch, dtype=tf.float32)
        outputs = cnn(batch_t)
        probs = outputs["probabilities"].numpy()
        all_probs.append(probs)
        all_preds_cnn.append(np.argmax(probs, axis=1))

    all_probs = np.vstack(all_probs)
    preds_cnn = np.concatenate(all_preds_cnn)
    preds_knn = agent.get_action_batch(all_probs, epsilon=0.0)

    # Track corrections
    corrections = preds_cnn != preds_knn
    num_corrected = np.sum(corrections)
    logger.info(f"Total corrections by k-NN: {num_corrected} / {len(labels)}")

    # Correction flow matrix (CNN -> k-NN)
    flow = np.zeros((10, 10), dtype=int)
    cnn_correct_knn_wrong = 0
    cnn_wrong_knn_correct = 0

    for i in range(len(labels)):
        if corrections[i]:
            flow[preds_cnn[i], preds_knn[i]] += 1
            if preds_cnn[i] == labels[i] and preds_knn[i] != labels[i]:
                cnn_correct_knn_wrong += 1
            elif preds_cnn[i] != labels[i] and preds_knn[i] == labels[i]:
                cnn_wrong_knn_correct += 1

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), gridspec_kw={'width_ratios': [1.3, 1]})
    fig.suptitle('Correction Flow: CNN -> k-NN Bandit', fontsize=16, fontweight='bold', y=1.02)

    # --- Heatmap ---
    flow_display = flow.copy().astype(float)
    np.fill_diagonal(flow_display, np.nan)

    im = ax1.imshow(flow_display, cmap='YlOrRd', interpolation='nearest', aspect='equal')

    ax1.set_xlabel('Final Decision (k-NN)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Original Decision (CNN)', fontsize=12, fontweight='bold')
    ax1.set_title('Reclassifications Map', fontsize=13, fontweight='bold')
    ax1.set_xticks(range(10))
    ax1.set_yticks(range(10))
    ax1.set_xticklabels(range(10), fontsize=11)
    ax1.set_yticklabels(range(10), fontsize=11)

    for i in range(10):
        for j in range(10):
            if i != j and flow[i, j] > 0:
                ax1.text(j, i, str(flow[i, j]), ha='center', va='center',
                        fontsize=9, fontweight='bold',
                        color='white' if flow[i, j] > flow.max() * 0.5 else 'black')

    plt.colorbar(im, ax=ax1, label='Number of Reclassified Images', shrink=0.8)

    # --- Summary bar charts ---
    acc_cnn = np.mean(preds_cnn == labels) * 100
    acc_knn = np.mean(preds_knn == labels) * 100

    # Hybrid system baseline
    cnn_conf = np.max(all_probs, axis=1)
    conf_threshold = 0.8
    preds_hybrid = np.where(cnn_conf > conf_threshold, preds_cnn, preds_knn)
    acc_hybrid = np.mean(preds_hybrid == labels) * 100

    categories = ['CNN\nAlone', 'k-NN\nAlone', 'Hybrid\nSystem']
    values = [acc_cnn, acc_knn, acc_hybrid]
    colors = ['#3498DB', '#F39C12', '#2ECC71']

    bars = ax2.bar(categories, values, color=colors, edgecolor='white', linewidth=2, width=0.6, alpha=0.9)

    for bar, val in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=14, fontweight='bold')

    ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Accuracy Comparison (Noisy Conditions)', fontsize=13, fontweight='bold')
    ax2.set_ylim(0, 100)
    ax2.axhline(y=acc_cnn, color='#3498DB', linestyle='--', alpha=0.3)

    stats_text = (
        f"Total Corrections: {num_corrected}\n"
        f"CNN Wrong -> k-NN Correct: {cnn_wrong_knn_correct}\n"
        f"CNN Correct -> k-NN Wrong: {cnn_correct_knn_wrong}\n"
        f"Net Correction Gain: {cnn_wrong_knn_correct - cnn_correct_knn_wrong}"
    )
    ax2.text(0.5, 0.35, stats_text, transform=ax2.transAxes,
             fontsize=10, verticalalignment='top', horizontalalignment='center',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', edgecolor='gray', alpha=0.8))

    plt.tight_layout()
    path = os.path.join(output_dir, "fluxo_correcoes_cnn_knn.png")
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    logger.info(f"[OK] Correction flow diagram saved to: {path}")

    # Summary logs
    logger.info(f"\nSimulation Accuracy Summary:")
    logger.info(f"  CNN alone: {acc_cnn:.1f}%")
    logger.info(f"  k-NN alone: {acc_knn:.1f}%")
    logger.info(f"  Hybrid System: {acc_hybrid:.1f}%")

def main():
    logger.info("==================================================")
    logger.info("VISUALIZING k-NN RL AGENT DECISION-MAKING")
    logger.info("==================================================")

    cnn, agent = load_system()

    # Load test data
    x_test, y_test = load_mnist_raw(DATA_DIR, kind='t10k')
    x_test = x_test.astype(np.float32) / 255.0

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Generate visual files
    visualize_confidence_profiles(x_test, y_test, cnn, agent, OUTPUT_DIR)
    visualize_correction_flow(x_test, y_test, cnn, agent, OUTPUT_DIR)

    logger.info("\n==================================================")
    logger.info("ALL DECISION-MAKING PROFILES GENERATED!")
    logger.info("==================================================")

if __name__ == "__main__":
    main()
