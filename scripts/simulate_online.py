"""
Progressive Online Simulation of the RL Agent (k-NN Bandit 128D).

Concept:
  - 10% of the MNIST dataset is reserved for "seeding" (Oracle Seeding).
  - 90% is the evaluation partition (completely disjoint, zero data leakage).
  - The 10% seeding data is divided into 10 incremental batches.
  - After ingesting each batch, the agent is evaluated on the 90% partition.
  - A publication-quality learning curve plot is generated at the end.
"""

import sys
import os
import logging
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from src.config import (
    DATA_DIR,
    CHECKPOINT_DIR,
    OUTPUT_DIR,
    RANDOM_SEED
)
from src.models.custom_cnn import RawModel
from src.models.knn_bandit_agent import KNNBanditAgent
from src.data.loader import load_mnist_raw

# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# --- Simulation Hyperparameters ---
NUM_LOTES          = 10        # Number of incremental batches
NOISE_EVAL         = 0.6       # Noise intensity for evaluation
CENARIOS_RUIDO     = [0.0, 0.2, 0.4, 0.6, 0.8]  # Noise scenarios for Oracle Seeding
BATCH_SIZE_CNN     = 1024      # Batch size for CNN feature extraction
BATCH_SIZE_RL_EVAL = 2048      # Batch size for RL agent evaluation
# ----------------------------------

def add_noise_batch(images: np.ndarray, intensity: float) -> np.ndarray:
    """Adds Gaussian noise to a batch of images."""
    noise = np.random.normal(loc=0.0, scale=intensity, size=images.shape)
    return np.clip(images + noise, 0.0, 1.0)

def extract_features_128d(images: np.ndarray, cnn, batch_size: int = BATCH_SIZE_CNN):
    """Extracts 128D latent features and CNN predictions in optimized batches."""
    all_states = []
    all_preds = []
    for i in range(0, len(images), batch_size):
        batch = images[i : i + batch_size]
        batch_tensor = tf.convert_to_tensor(batch, dtype=tf.float32)
        outputs = cnn(batch_tensor)
        latent = outputs["latent_features"].numpy()
        probs = outputs["probabilities"].numpy()
        preds = np.argmax(probs, axis=1)
        all_states.append(latent)
        all_preds.append(preds)
    return np.vstack(all_states), np.concatenate(all_preds)

def evaluate_agent(agent, features_eval: np.ndarray, labels_eval: np.ndarray,
                   batch_size: int = BATCH_SIZE_RL_EVAL) -> float:
    """Evaluates the RL agent on the evaluation partition in optimized batches."""
    correct = 0
    total = len(labels_eval)
    for i in range(0, total, batch_size):
        feats_batch = features_eval[i : i + batch_size]
        labels_batch = labels_eval[i : i + batch_size]
        preds_rl = agent.get_action_batch(feats_batch, epsilon=0.0)
        correct += int(np.sum(preds_rl == labels_batch))
    return correct / total

def generate_plot(history: list, output_path: str):
    """Generates a publication-quality plot of the learning curve."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    sizes = [h[0] for h in history]
    accuracies = [h[1] * 100 for h in history]

    # --- Color Palette & Styles ---
    COLOR_BG       = '#0D1117'
    COLOR_AREA     = '#161B22'
    COLOR_GRID     = '#21262D'
    COLOR_TEXT     = '#C9D1D9'
    COLOR_TEXT_DIM = '#8B949E'
    COLOR_LINE     = '#58A6FF'
    COLOR_GRADIENT = '#1F6FEB'
    COLOR_POINT    = '#79C0FF'
    COLOR_BORDER_PT = '#388BFD'
    COLOR_ANNOT    = '#F0F6FC'

    fig, ax = plt.subplots(figsize=(14, 7), dpi=200)
    fig.patch.set_facecolor(COLOR_BG)
    ax.set_facecolor(COLOR_AREA)

    # --- Gradient Fill ---
    ax.fill_between(sizes, accuracies, alpha=0.15, color=COLOR_GRADIENT, zorder=1)

    # --- Main Line ---
    ax.plot(sizes, accuracies,
            color=COLOR_LINE, linewidth=2.8, zorder=3,
            marker='o', markersize=9,
            markerfacecolor=COLOR_POINT, markeredgecolor=COLOR_BORDER_PT,
            markeredgewidth=1.8)

    # --- Point Annotations ---
    for i, (x, y) in enumerate(zip(sizes, accuracies)):
        offset_y = 12 if i % 2 == 0 else -18
        ax.annotate(f'{y:.1f}%',
                    xy=(x, y), xytext=(0, offset_y),
                    textcoords='offset points', ha='center', va='bottom',
                    fontsize=9, fontweight='bold', color=COLOR_ANNOT,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=COLOR_BG,
                               edgecolor=COLOR_GRID, alpha=0.85))

    # --- Axes and Titles ---
    ax.set_xlabel('Episodic Memory Size', fontsize=13, color=COLOR_TEXT,
                  fontweight='bold', labelpad=12)
    ax.set_ylabel('RL Accuracy (%)', fontsize=13, color=COLOR_TEXT,
                  fontweight='bold', labelpad=12)
    ax.set_title('RL Specialist — Online Learning Curve\n'
                 f'Evaluation on {len(sizes)} incremental seeding rounds  |  '
                 f'Test Noise = {NOISE_EVAL}',
                 fontsize=15, color=COLOR_ANNOT, fontweight='bold', pad=20)

    # --- Grid and Frame ---
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.4, color=COLOR_GRID)
    ax.tick_params(colors=COLOR_TEXT_DIM, labelsize=10)
    for spine in ax.spines.values():
        spine.set_color(COLOR_GRID)
        spine.set_linewidth(0.8)

    # Format X axis with thousands separator
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x):,}'))

    # --- Context Metadata Box ---
    mem_final = sizes[-1]
    acc_final = accuracies[-1]
    acc_initial = accuracies[0]
    delta = acc_final - acc_initial

    info_text = (f'Delta Accuracy: +{delta:.1f}pp\n'
                 f'Final Memory: {mem_final:,}\n'
                 f'Peak Accuracy: {max(accuracies):.1f}%')
    ax.text(0.02, 0.97, info_text,
            transform=ax.transAxes, fontsize=10, color=COLOR_TEXT_DIM,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor=COLOR_BG,
                      edgecolor=COLOR_GRID, alpha=0.9))

    plt.tight_layout(pad=2.0)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)
    logger.info(f"\n  Learning curve plot saved to: {output_path}")


def main():
    # 1. Configure Hardware
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logger.info(f"  GPU(s) detected: {len(gpus)}")
    else:
        logger.info("  No GPU detected - using CPU.")

    # 2. Load CNN (Feature Extractor)
    logger.info("\nLoading CNN (128D Feature Extractor)...")
    cnn = RawModel()
    ckpt = tf.train.Checkpoint(model=cnn)
    latest_ckpt = tf.train.latest_checkpoint(CHECKPOINT_DIR)
    if not latest_ckpt:
        logger.error(f"ERROR: Checkpoint not found in {CHECKPOINT_DIR}")
        return
    ckpt.restore(latest_ckpt).expect_partial()
    logger.info("  CNN weights restored successfully.")

    # 3. Instantiate RL Agent (use unified KNNBanditAgent)
    agent = KNNBanditAgent(k=30, n_actions=10, latent_dim=128, use_pca=False)

    # 4. Load Dataset
    logger.info("\nLoading MNIST dataset...")
    x_full, y_full = load_mnist_raw(DATA_DIR, kind='train')
    x_full = x_full.astype(np.float32) / 255.0

    logger.info("Creating split: 10% Seeding / 90% Evaluation...")
    x_seed, x_eval, y_seed, y_eval = train_test_split(
        x_full, y_full,
        test_size=0.90,
        random_state=RANDOM_SEED,
        shuffle=True,
        stratify=y_full
    )
    logger.info(f"  Seeding Partition (10%): {len(x_seed):,} samples")
    logger.info(f"  Evaluation Partition (90%): {len(x_eval):,} samples")

    # 5. Prepare Incremental Batches
    indices_seed = np.arange(len(x_seed))
    np.random.seed(RANDOM_SEED)
    np.random.shuffle(indices_seed)
    batches = np.array_split(indices_seed, NUM_LOTES)

    logger.info(f"  Split into {NUM_LOTES} batches (~{len(batches[0])} samples per batch)")

    # 6. Pre-calculate Evaluation Features (once)
    logger.info(f"\nPre-calculating evaluation features (noise = {NOISE_EVAL})...")
    x_eval_noisy = add_noise_batch(x_eval, NOISE_EVAL) if NOISE_EVAL > 0 else x_eval
    features_eval, _ = extract_features_128d(x_eval_noisy, cnn)
    logger.info(f"  Evaluation features shape: {features_eval.shape}")

    # 7. Incremental Seeding Loop
    history = []

    logger.info("\n" + "=" * 60)
    logger.info("  ONLINE SIMULATION — PROGRESSIVE SEEDING")
    logger.info("=" * 60)

    for batch_idx, batch_indices in enumerate(batches):
        batch_num = batch_idx + 1
        x_batch = x_seed[batch_indices]
        y_batch = y_seed[batch_indices]

        logger.info(f"\n── Batch {batch_num}/{NUM_LOTES} "
                    f"({len(batch_indices)} samples) ──────────────────────")

        # 7a. Oracle Seeding: generate noise scenarios for current batch
        for r, intensity in enumerate(CENARIOS_RUIDO):
            x_noisy = add_noise_batch(x_batch, intensity) if intensity > 0 else x_batch

            states, preds_cnn = extract_features_128d(x_noisy, cnn)

            # Positive experiences (oracle label, reward +1.0)
            agent.add_experience_batch(states, y_batch, np.ones(len(y_batch)))

            # Negative experiences (CNN error, reward -1.0)
            errors = preds_cnn != y_batch
            num_errors = int(np.sum(errors))
            if num_errors > 0:
                agent.add_experience_batch(
                    states[errors], preds_cnn[errors], np.full(num_errors, -1.0)
                )

            acc_cnn = np.mean(preds_cnn == y_batch) * 100
            logger.info(f"    Noise {intensity:.1f} → +{len(y_batch):,} pos, "
                        f"+{num_errors:,} neg | CNN Accuracy: {acc_cnn:.1f}%")

        # 7b. Rebuild k-NN index with accumulated memory
        logger.info(f"  Rebuilding k-NN index (memory size: {agent.memory_size:,})...")
        agent.build_index()

        # 7c. Evaluate on the 90% partition
        logger.info("  Evaluating on 90% unseen partition...")
        acc_rl = evaluate_agent(agent, features_eval, y_eval)
        history.append((agent.memory_size, acc_rl))

        logger.info(f"  ✓ Memory: {agent.memory_size:,} | "
                    f"RL Accuracy: {acc_rl * 100:.2f}%")

    # 8. Final Results Summary
    logger.info("\n" + "=" * 60)
    logger.info("  SIMULATION RESULTS SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  {'Batch':<6} {'Memory':>10} {'RL Acc (%)':>12}")
    logger.info("  " + "-" * 30)
    for i, (mem, acc) in enumerate(history):
        logger.info(f"  {i+1:<6} {mem:>10,} {acc * 100:>11.2f}%")

    # 9. Save Learning Curve Plot (save to outputs/)
    output_plot_path = os.path.join(OUTPUT_DIR, "rl_online_learning_curve.png")
    generate_plot(history, output_plot_path)

    logger.info("\nOnline simulation completed successfully!")


if __name__ == "__main__":
    main()
