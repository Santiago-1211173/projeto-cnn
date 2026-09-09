"""
Comparative Benchmark: MLP (Q-Network) vs k-NN Bandit.
Runs BOTH RL agents side-by-side on the same test images and produces a detailed comparison report.

Outputs: outputs/comparacao_mlp_vs_knn.png (plot) + terminal report.
"""

import sys
import os
import time
import logging
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from src.config import (
    DATA_DIR,
    CHECKPOINT_DIR,
    MAHALANOBIS_THRESHOLD,
    MAHALANOBIS_PROFILES_PATH,
    MEMORY_BANK_10D_PATH,
    MEMORY_BANK_128D_PATH,
    OUTPUT_DIR
)
from src.models.custom_cnn import RawModel
from src.models.mlp_bandit_agent import QNetworkAgent
from src.models.knn_bandit_agent import KNNBanditAgent, KNNBanditAgent128D
from src.data.loader import load_mnist_raw

# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def calculate_mahalanobis_batch(vectors, mu, inv_sigma):
    """Vectorized Mahalanobis distance calculation."""
    diff = vectors - mu
    left = np.dot(diff, inv_sigma)
    sq_dist = np.sum(left * diff, axis=1)
    return np.sqrt(sq_dist)

def extract_features_cnn(images, cnn, batch_size=500):
    """Extracts 128D latent vectors and 10D probabilities from CNN in batches."""
    all_128d, all_10d, all_preds = [], [], []
    num_batches = len(images) // batch_size
    remainder = len(images) % batch_size

    for i in range(num_batches + (1 if remainder > 0 else 0)):
        start = i * batch_size
        end = min(start + batch_size, len(images))
        batch = tf.convert_to_tensor(images[start:end], dtype=tf.float32)

        outputs = cnn(batch)
        all_128d.append(outputs["latent_features"].numpy())
        all_10d.append(outputs["probabilities"].numpy())
        all_preds.append(tf.argmax(outputs["probabilities"], axis=1).numpy())

    return np.vstack(all_128d), np.vstack(all_10d), np.concatenate(all_preds)

def evaluate_agent_mlp(agent_mlp, vectors_128d):
    """Generates predictions from the MLP agent using 128D latent space."""
    q_values = agent_mlp(tf.convert_to_tensor(vectors_128d, dtype=tf.float32))
    return tf.argmax(q_values, axis=1).numpy()

def evaluate_agent_knn(agent_knn, probs_10d):
    """Generates predictions from the k-NN agent using 10D probabilities."""
    return agent_knn.get_action_batch(probs_10d, epsilon=0.0)

def calculate_routing(vectors_128d, island_map, threshold):
    """Calculates the Mahalanobis routing mask (True = CNN, False = RL)."""
    all_distances = np.zeros((len(vectors_128d), 10))
    for digit in range(10):
        mu = island_map[str(digit)].item()["mu"]
        inv_sigma = island_map[str(digit)].item()["inv_sigma"]
        all_distances[:, digit] = calculate_mahalanobis_batch(vectors_128d, mu, inv_sigma)
    min_distances = np.min(all_distances, axis=1)
    cnn_mask = min_distances < threshold
    return cnn_mask

def evaluate_scenario(name, images, labels, cnn, agent_mlp, agent_knn, agent_knn_128d, island_map):
    """Evaluates a complete scenario for all agents."""

    # 1. Feature extraction
    t0 = time.time()
    vectors_128d, probs_10d, preds_cnn = extract_features_cnn(images, cnn)
    tempo_cnn = time.time() - t0

    # 2. Mahalanobis Routing
    cnn_mask = calculate_routing(vectors_128d, island_map, MAHALANOBIS_THRESHOLD)
    rl_mask = ~cnn_mask

    # 3. Agent predictions
    t_mlp = time.time()
    preds_mlp = evaluate_agent_mlp(agent_mlp, vectors_128d)
    tempo_mlp = time.time() - t_mlp

    t_knn = time.time()
    preds_knn = evaluate_agent_knn(agent_knn, probs_10d)
    tempo_knn = time.time() - t_knn

    t_knn_128d = time.time()
    preds_knn_128d = agent_knn_128d.get_action_batch(vectors_128d, epsilon=0.0)
    tempo_knn_128d = time.time() - t_knn_128d

    # 4. CNN Alone metrics
    acc_cnn = np.mean(preds_cnn == labels) * 100

    # 5. Isolated Agent metrics (no routing)
    acc_mlp_isolated = np.mean(preds_mlp == labels) * 100
    acc_knn_isolated = np.mean(preds_knn == labels) * 100
    acc_knn_128d_isolated = np.mean(preds_knn_128d == labels) * 100

    # 6. Routed Agent metrics (only on subset sent to RL)
    total_rl = np.sum(rl_mask)
    total_cnn_route = np.sum(cnn_mask)

    correct_cnn_route = np.sum((preds_cnn == labels) & cnn_mask)

    if total_rl > 0:
        correct_mlp_rl = np.sum((preds_mlp == labels) & rl_mask)
        correct_knn_rl = np.sum((preds_knn == labels) & rl_mask)
        correct_knn_128d_rl = np.sum((preds_knn_128d == labels) & rl_mask)
        acc_mlp_rl = (correct_mlp_rl / total_rl) * 100
        acc_knn_rl = (correct_knn_rl / total_rl) * 100
        acc_knn_128d_rl = (correct_knn_128d_rl / total_rl) * 100
    else:
        correct_mlp_rl, correct_knn_rl, correct_knn_128d_rl = 0, 0, 0
        acc_mlp_rl, acc_knn_rl, acc_knn_128d_rl = 0.0, 0.0, 0.0

    # 7. Global Hybrid Accuracy
    acc_hybrid_mlp = ((correct_cnn_route + correct_mlp_rl) / len(labels)) * 100
    acc_hybrid_knn = ((correct_cnn_route + correct_knn_rl) / len(labels)) * 100
    acc_hybrid_knn_128d = ((correct_cnn_route + correct_knn_128d_rl) / len(labels)) * 100

    # 8. Class-wise metrics for routed subset
    acc_by_class_mlp = np.zeros(10)
    acc_by_class_knn = np.zeros(10)
    acc_by_class_knn_128d = np.zeros(10)
    count_by_class = np.zeros(10)
    for d in range(10):
        mask_d = (labels == d) & rl_mask
        n_d = np.sum(mask_d)
        count_by_class[d] = n_d
        if n_d > 0:
            acc_by_class_mlp[d] = np.mean(preds_mlp[mask_d] == labels[mask_d]) * 100
            acc_by_class_knn[d] = np.mean(preds_knn[mask_d] == labels[mask_d]) * 100
            acc_by_class_knn_128d[d] = np.mean(preds_knn_128d[mask_d] == labels[mask_d]) * 100

    return {
        "nome": name,
        "n_total": len(labels),
        "n_cnn": int(total_cnn_route),
        "n_rl": int(total_rl),
        "acc_cnn": acc_cnn,
        "acc_mlp_isolated": acc_mlp_isolated,
        "acc_knn_isolated": acc_knn_isolated,
        "acc_knn_128d_isolated": acc_knn_128d_isolated,
        "acc_mlp_rl": acc_mlp_rl,
        "acc_knn_rl": acc_knn_rl,
        "acc_knn_128d_rl": acc_knn_128d_rl,
        "acc_hybrid_mlp": acc_hybrid_mlp,
        "acc_hybrid_knn": acc_hybrid_knn,
        "acc_hybrid_knn_128d": acc_hybrid_knn_128d,
        "tempo_mlp": tempo_mlp,
        "tempo_knn": tempo_knn,
        "tempo_knn_128d": tempo_knn_128d,
        "tempo_cnn": tempo_cnn,
        "acc_by_class_mlp": acc_by_class_mlp,
        "acc_by_class_knn": acc_by_class_knn,
        "acc_by_class_knn_128d": acc_by_class_knn_128d,
        "count_by_class": count_by_class,
    }

def print_report(r):
    """Prints formatted scenario results."""
    logger.info("=" * 85)
    logger.info(f"  {r['nome'].upper()}")
    logger.info("=" * 85)
    logger.info(f"  Images: {r['n_total']} | Routed CNN: {r['n_cnn']} | Routed RL: {r['n_rl']}")
    logger.info("-" * 85)
    logger.info(f"  {'Metric':<35} {'MLP (Q-Net)':>15} {'k-NN 10D':>15} {'k-NN Improved':>15}")
    logger.info("-" * 85)
    logger.info(f"  {'CNN Alone (No Arbitrator):':<35} {r['acc_cnn']:>14.1f}% {r['acc_cnn']:>14.1f}% {r['acc_cnn']:>14.1f}%")
    logger.info(f"  {'Agent Isolated:':<35} {r['acc_mlp_isolated']:>14.1f}% {r['acc_knn_isolated']:>14.1f}% {r['acc_knn_128d_isolated']:>14.1f}%")
    logger.info(f"  {'Agent on RL Subset:':<35} {r['acc_mlp_rl']:>14.1f}% {r['acc_knn_rl']:>14.1f}% {r['acc_knn_128d_rl']:>14.1f}%")
    logger.info(f"  {'GLOBAL HYBRID SYSTEM:':<35} {r['acc_hybrid_mlp']:>14.1f}% {r['acc_hybrid_knn']:>14.1f}% {r['acc_hybrid_knn_128d']:>14.1f}%")
    logger.info("-" * 85)
    logger.info(f"  {'Inference Time:':<35} {r['tempo_mlp']*1000:>12.1f} ms {r['tempo_knn']*1000:>12.1f} ms {r['tempo_knn_128d']*1000:>12.1f} ms")
    logger.info("=" * 85)

    if r['n_rl'] > 0:
        logger.info(f"\n  Acc by Class (RL Subset, {r['n_rl']} imgs):")
        logger.info(f"  {'Digit':<8} {'N':>6} {'MLP':>10} {'k-NN 10D':>10} {'k-NN 128D':>10} {'Delta (128D-MLP)':>16}")
        for d in range(10):
            n = int(r['count_by_class'][d])
            a_mlp = r['acc_by_class_mlp'][d]
            a_knn = r['acc_by_class_knn'][d]
            a_knn_128d = r['acc_by_class_knn_128d'][d]
            delta = a_knn_128d - a_mlp
            sign = "+" if delta >= 0 else ""
            logger.info(f"  {d:<8} {n:>6} {a_mlp:>9.1f}% {a_knn:>9.1f}% {a_knn_128d:>9.1f}% {sign}{delta:>15.1f}%")
    logger.info("")

def generate_comparison_plot(results, output_path):
    """Generates comparative bar plots side-by-side."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Comparative Benchmark: MLP (Q-Network) vs k-NN Bandit",
                 fontsize=16, fontweight='bold', y=0.98)

    color_mlp = '#e74c3c'
    color_knn = '#2ecc71'
    color_cnn = '#3498db'

    for idx, r in enumerate(results):
        ax = axes[idx]

        labels_bar = ['CNN\nAlone', 'Agent\nIsolated', 'Agent on\nRL Subset', 'Hybrid\nSystem']
        vals_mlp = [r['acc_cnn'], r['acc_mlp_isolated'], r['acc_mlp_rl'], r['acc_hybrid_mlp']]
        vals_knn = [r['acc_cnn'], r['acc_knn_isolated'], r['acc_knn_rl'], r['acc_hybrid_knn']]
        vals_knn_128d = [r['acc_cnn'], r['acc_knn_128d_isolated'], r['acc_knn_128d_rl'], r['acc_hybrid_knn_128d']]

        x = np.arange(len(labels_bar))
        w = 0.25

        bars_mlp = ax.bar(x - w, vals_mlp, w, label='MLP (Q-Network)', color=color_mlp, alpha=0.85, edgecolor='white')
        bars_knn = ax.bar(x, vals_knn, w, label='k-NN 10D', color=color_knn, alpha=0.85, edgecolor='white')
        bars_knn_128d = ax.bar(x + w, vals_knn_128d, w, label='k-NN 128D', color='#9b59b6', alpha=0.85, edgecolor='white')

        # Baseline CNN is identical across agents
        bars_mlp[0].set_color(color_cnn)
        bars_knn[0].set_color(color_cnn)
        bars_knn_128d[0].set_color(color_cnn)

        # Annotate bars
        for bar in list(bars_mlp) + list(bars_knn) + list(bars_knn_128d):
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width()/2., h + 0.5,
                        f'{h:.1f}%', ha='center', va='bottom', fontsize=7, fontweight='bold')

        ax.set_title(r['nome'], fontsize=13, fontweight='bold', pad=12)
        ax.set_ylabel('Accuracy (%)')
        ax.set_xticks(x)
        ax.set_xticklabels(labels_bar, fontsize=9)
        ax.set_ylim(0, 115)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(axis='y', alpha=0.3)

        delta = r['acc_hybrid_knn_128d'] - r['acc_hybrid_mlp']
        sign = "+" if delta >= 0 else ""
        color_delta = '#9b59b6' if delta >= 0 else color_mlp
        ax.annotate(f'Delta (128D vs MLP) = {sign}{delta:.1f}%',
                    xy=(3, max(r['acc_hybrid_mlp'], r['acc_hybrid_knn'], r['acc_hybrid_knn_128d']) + 4),
                    fontsize=10, fontweight='bold', color=color_delta, ha='center')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    logger.info(f"[OK] Comparison plot saved to: {output_path}")

def main():
    # 1. Configure Hardware
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    logger.info("=" * 72)
    logger.info("  COMPARATIVE BENCHMARK: MLP (Q-Network) vs k-NN Bandit")
    logger.info("=" * 72)

    # 2. Load CNN
    logger.info("\n[1/5] Loading CNN model...")
    cnn = RawModel()
    ckpt_cnn = tf.train.Checkpoint(model=cnn)
    latest_cnn = tf.train.latest_checkpoint(CHECKPOINT_DIR)
    if not latest_cnn:
         logger.error(f"Checkpoint not found in {CHECKPOINT_DIR}. Run training first.")
         sys.exit(1)
    ckpt_cnn.restore(latest_cnn).expect_partial()

    # 3. Load Mahalanobis Profiles
    logger.info("[2/5] Loading Arbitrator (Mahalanobis Profiles)...")
    if not os.path.exists(MAHALANOBIS_PROFILES_PATH):
         logger.error(f"Mahalanobis profiles not found at {MAHALANOBIS_PROFILES_PATH}. Run profile script first.")
         sys.exit(1)
    island_map = np.load(MAHALANOBIS_PROFILES_PATH, allow_pickle=True)

    # 4. Load MLP Agent
    logger.info("[3/5] Loading MLP Agent (Q-Network)...")
    agent_mlp = QNetworkAgent()
    ckpt_agent = tf.train.Checkpoint(model=agent_mlp)
    mlp_weights_path = os.path.join(OUTPUT_DIR, "rl_agent_weights-1")
    ckpt_agent.restore(mlp_weights_path).expect_partial()

    # 5. Load k-NN Agents (10D and 128D)
    logger.info("[4/5] Loading k-NN 10D Agent...")
    agent_knn = KNNBanditAgent(k=30, n_actions=10, latent_dim=10, use_pca=False)
    if not os.path.exists(MEMORY_BANK_10D_PATH):
         logger.error(f"10D Memory bank not found at {MEMORY_BANK_10D_PATH}.")
         sys.exit(1)
    agent_knn.load(MEMORY_BANK_10D_PATH)

    logger.info("[5/5] Loading k-NN 128D Agent...")
    agent_knn_128d = KNNBanditAgent128D(k=30, n_actions=10, latent_dim=128, use_pca=True, pca_components=48)
    if not os.path.exists(MEMORY_BANK_128D_PATH):
         logger.error(f"128D Memory bank not found at {MEMORY_BANK_128D_PATH}.")
         sys.exit(1)
    agent_knn_128d.load(MEMORY_BANK_128D_PATH)

    # 6. Load Test Dataset
    x_test, y_test = load_mnist_raw(DATA_DIR, kind='t10k')
    x_test_clean = x_test.astype(np.float32) / 255.0

    # Add Target Noise level 0.6
    np.random.seed(42)
    noise = np.random.normal(loc=0.0, scale=0.6, size=x_test_clean.shape)
    x_test_noisy = np.clip(x_test_clean + noise, 0., 1.)

    # 7. Evaluate both scenarios
    logger.info("\n" + "=" * 72)
    r_clean = evaluate_scenario(
        "Scenario A: Clean Images",
        x_test_clean, y_test, cnn, agent_mlp, agent_knn, agent_knn_128d, island_map
    )
    print_report(r_clean)

    r_noisy = evaluate_scenario(
        "Scenario B: Noisy Images (σ=0.6)",
        x_test_noisy, y_test, cnn, agent_mlp, agent_knn, agent_knn_128d, island_map
    )
    print_report(r_noisy)

    # 8. Generate Graphic
    plot_path = os.path.join(OUTPUT_DIR, "comparacao_mlp_vs_knn.png")
    generate_comparison_plot([r_clean, r_noisy], plot_path)

    # 9. Final Summary
    logger.info("=" * 85)
    logger.info("  FINAL SUMMARY")
    logger.info("=" * 85)
    logger.info(f"  {'Scenario':<35} {'Hybrid+MLP':>14} {'Hybrid+k-NN':>14} {'Hybrid+128D':>14} {'Delta (128D-MLP)':>14}")
    logger.info("-" * 85)
    for r in [r_clean, r_noisy]:
        d = r['acc_hybrid_knn_128d'] - r['acc_hybrid_mlp']
        s = "+" if d >= 0 else ""
        logger.info(f"  {r['nome']:<35} {r['acc_hybrid_mlp']:>13.1f}% {r['acc_hybrid_knn']:>13.1f}% {r['acc_hybrid_knn_128d']:>13.1f}% {s}{d:>11.1f}%")
    logger.info("=" * 85)

    # 10. Markdown Table for README
    logger.info("\n--- README MARKDOWN TABLE ---")
    logger.info("| Metric | MLP (Q-Network) | k-NN 10D | k-NN Improved (128D) | Gap (128D vs MLP) |")
    logger.info("|---|---|---|---|---|")
    logger.info(f"| Hybrid (Clean) | {r_clean['acc_hybrid_mlp']:.1f}% | {r_clean['acc_hybrid_knn']:.1f}% | {r_clean['acc_hybrid_knn_128d']:.1f}% | {r_clean['acc_hybrid_knn_128d'] - r_clean['acc_hybrid_mlp']:+.1f}% |")
    logger.info(f"| Hybrid (Noisy σ=0.6) | {r_noisy['acc_hybrid_mlp']:.1f}% | {r_noisy['acc_hybrid_knn']:.1f}% | {r_noisy['acc_hybrid_knn_128d']:.1f}% | {r_noisy['acc_hybrid_knn_128d'] - r_noisy['acc_hybrid_mlp']:+.1f}% |")
    logger.info(f"| Agent Isolated (Clean) | {r_clean['acc_mlp_isolated']:.1f}% | {r_clean['acc_knn_isolated']:.1f}% | {r_clean['acc_knn_128d_isolated']:.1f}% | {r_clean['acc_knn_128d_isolated'] - r_clean['acc_mlp_isolated']:+.1f}% |")
    logger.info(f"| Agent Isolated (Noisy) | {r_noisy['acc_mlp_isolated']:.1f}% | {r_noisy['acc_knn_isolated']:.1f}% | {r_noisy['acc_knn_128d_isolated']:.1f}% | {r_noisy['acc_knn_128d_isolated'] - r_noisy['acc_mlp_isolated']:+.1f}% |")
    logger.info(f"| Inference (ms) | {r_noisy['tempo_mlp']*1000:.1f} | {r_noisy['tempo_knn']*1000:.1f} | {r_noisy['tempo_knn_128d']*1000:.1f} | - |")
    logger.info(f"| Training | 10 epochs + backprop | 0 (lazy) | 0 (lazy) | - |")
    logger.info(f"| Parameters | ~8,714 (128→64→10) | 0 (non-parametric) | 0 (non-parametric) | - |")


if __name__ == "__main__":
    main()
