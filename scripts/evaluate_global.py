"""
Global Evaluation of the Hybrid Vision Pipeline (CNN + RL Specialist).

Executes two independent evaluation scenarios:
  Approach A — Native t10k Test Set (10,000 samples).
  Approach B — Disjoint 90% Partition of the Training Set (54,000 samples).

For each scenario:
  1. Constructs a mixed-noise dataset (5 noise bands: 0.0 to 0.8).
  2. Sweeps Mahalanobis thresholds from config values.
  3. Executes the hybrid pipeline in GPU-optimized batches.
  4. Extracts metrics: Hybrid Acc, CNN-only Acc, RL-only Acc, and RL routing rate.
  5. Generates a markdown table in the console and a double-panel publication plot.
"""

import os
import sys
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
    MAHALANOBIS_PROFILES_PATH,
    MEMORY_BANK_128D_PATH,
    NOISE_SWEEP,
    THRESHOLD_SWEEP_START,
    THRESHOLD_SWEEP_END,
    THRESHOLD_SWEEP_STEP,
    RANDOM_SEED
)
from src.models.custom_cnn import RawModel
from src.models.knn_bandit_agent import KNNBanditAgent, KNNBanditAgent128D
from src.data.loader import load_mnist_raw

# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# --- Global Evaluation Setup ---
NOISE_LEVELS       = NOISE_SWEEP
THRESHOLDS         = np.arange(THRESHOLD_SWEEP_START, THRESHOLD_SWEEP_END + 1e-9, THRESHOLD_SWEEP_STEP)
BATCH_SIZE_CNN     = 1024
BATCH_SIZE_RL      = 2048
OUTPUT_FIGURE_PATH = os.path.join(OUTPUT_DIR, "hybrid_global_evaluation.png")
# ------------------------------

def add_noise_batch(images: np.ndarray, intensity: float) -> np.ndarray:
    """Adds Gaussian noise to a batch of images."""
    if intensity <= 0.0:
        return images.copy()
    noise = np.random.normal(loc=0.0, scale=intensity, size=images.shape)
    return np.clip(images + noise, 0.0, 1.0).astype(np.float32)

def build_mixed_noise_dataset(x: np.ndarray, y: np.ndarray, levels: list) -> tuple:
    """Splits x/y into balanced partitions, applies respective noise intensity, and concatenates."""
    n = len(x)
    n_levels = len(levels)
    slice_size = n // n_levels

    # Shuffle with fixed seed for reproducibility
    rng = np.random.RandomState(RANDOM_SEED)
    indices = rng.permutation(n)

    x_slices, y_slices = [], []
    for i, level in enumerate(levels):
        start = i * slice_size
        end = start + slice_size if i < n_levels - 1 else n
        idx = indices[start:end]
        x_slices.append(add_noise_batch(x[idx], level))
        y_slices.append(y[idx])

    return np.concatenate(x_slices, axis=0), np.concatenate(y_slices, axis=0)

def extract_features_and_predictions(images: np.ndarray, cnn, batch_size: int = BATCH_SIZE_CNN):
    """Extracts 128D features, probabilities, and predictions from CNN in batches."""
    all_latent = []
    all_preds = []
    all_probs = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size]
        tensor = tf.convert_to_tensor(batch, dtype=tf.float32)
        outputs = cnn(tensor)
        latent = outputs["latent_features"].numpy()
        probs = outputs["probabilities"].numpy()
        preds = np.argmax(probs, axis=1)
        all_latent.append(latent)
        all_preds.append(preds)
        all_probs.append(probs)
    return (np.vstack(all_latent),
            np.concatenate(all_preds),
            np.vstack(all_probs))

def calculate_mahalanobis_batch(vectors: np.ndarray, profiles: dict) -> np.ndarray:
    """Calculates minimum Mahalanobis distance of each vector to the closest class center."""
    n = len(vectors)
    min_dists = np.full(n, np.inf, dtype=np.float64)

    for digit in range(10):
        info = profiles[str(digit)].item()
        mu = info["mu"]
        inv_sigma = info["inv_sigma"]
        diff = vectors - mu
        left = diff @ inv_sigma
        dist_sq = np.sum(left * diff, axis=1)
        dist = np.sqrt(np.maximum(dist_sq, 0.0))
        min_dists = np.minimum(min_dists, dist)

    return min_dists

def evaluate_rl_batch(agent, features: np.ndarray, labels: np.ndarray,
                      batch_size: int = BATCH_SIZE_RL) -> np.ndarray:
    """Obtains RL predictions in batches."""
    preds_list = []
    for i in range(0, len(features), batch_size):
        f_batch = features[i:i + batch_size]
        preds_list.append(agent.get_action_batch(f_batch, epsilon=0.0))
    return np.concatenate(preds_list)

def evaluate_approach(name: str, x: np.ndarray, y: np.ndarray,
                      cnn, agent, profiles: dict) -> dict:
    """Executes the complete global evaluation sweep for a dataset approach."""
    logger.info(f"\n{'═' * 70}")
    logger.info(f"  {name}")
    logger.info(f"{'═' * 70}")

    # 1. Build mixed noise dataset
    logger.info(f"  Constructing mixed noise dataset ({len(x):,} samples, {len(NOISE_LEVELS)} bands)...")
    x_mixed, y_mixed = build_mixed_noise_dataset(x, y, NOISE_LEVELS)
    logger.info(f"  Mixed dataset: {len(x_mixed):,} samples.")

    # 2. Extract features
    logger.info("  Extracting 128D features and CNN predictions...")
    features_128d, preds_cnn, _ = extract_features_and_predictions(x_mixed, cnn)
    logger.info("  CNN baseline calculated.")

    # 3. Calculate Mahalanobis distances
    logger.info("  Calculating Mahalanobis distances...")
    dists = calculate_mahalanobis_batch(features_128d, profiles)
    logger.info(f"  Distances: min={dists.min():.2f} | median={np.median(dists):.2f} | max={dists.max():.2f}")

    # 4. Extract RL predictions (once, reusable across all thresholds)
    logger.info("  Generating RL Specialist predictions...")
    preds_rl = evaluate_rl_batch(agent, features_128d, y_mixed)
    logger.info("  RL predictions calculated.")

    # 5. Sweep thresholds
    results = {
        "thresholds": [],
        "hybrid_acc": [],
        "cnn_acc": [],
        "rl_acc": [],
        "rl_rate": [],
    }

    acc_cnn_global = float(np.mean(preds_cnn == y_mixed) * 100)
    acc_rl_global = float(np.mean(preds_rl == y_mixed) * 100)

    for t in THRESHOLDS:
        mask_cnn = dists < t
        mask_rl = ~mask_cnn

        # Hybrid routing decision
        preds_hybrid = np.where(mask_cnn, preds_cnn, preds_rl)
        acc_hybrid = float(np.mean(preds_hybrid == y_mixed) * 100)
        rate_rl = float(np.mean(mask_rl) * 100)

        results["thresholds"].append(float(t))
        results["hybrid_acc"].append(acc_hybrid)
        results["cnn_acc"].append(acc_cnn_global)
        results["rl_acc"].append(acc_rl_global)
        results["rl_rate"].append(rate_rl)

    # 6. Print Markdown table
    logger.info(f"\n  ### {name} — Results by Threshold\n")
    logger.info(f"  | {'Threshold':>10} | {'Hybrid Acc':>11} | {'CNN Acc':>9} | {'RL Acc':>8} | {'RL Rate':>9} |")
    logger.info(f"  |{'-' * 12}|{'-' * 13}|{'-' * 11}|{'-' * 10}|{'-' * 11}|")
    for i, t in enumerate(results["thresholds"]):
        logger.info(
            f"  | {t:>10.1f} | {results['hybrid_acc'][i]:>10.2f}% "
            f"| {results['cnn_acc'][i]:>8.2f}% "
            f"| {results['rl_acc'][i]:>7.2f}% "
            f"| {results['rl_rate'][i]:>8.2f}% |"
        )

    return results

def generate_comparative_plot(res_a: dict, res_b: dict, output_path: str):
    """Generates double-panel comparative learning and thresholding curves."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Theme colors (GitHub Dark Mode aesthetic)
    BG_OUTER    = '#0D1117'
    BG_INNER    = '#161B22'
    GRID_CLR    = '#21262D'
    TEXT_CLR    = '#C9D1D9'
    TEXT_DIM    = '#8B949E'
    TEXT_BRIGHT = '#F0F6FC'

    # Lines: Scenario A
    CLR_A_HYBRID = '#58A6FF'
    CLR_A_CNN    = '#F78166'
    CLR_A_RL     = '#7EE787'
    CLR_A_RATE   = '#D2A8FF'

    # Lines: Scenario B
    CLR_B_HYBRID = '#79C0FF'
    CLR_B_CNN    = '#FFA198'
    CLR_B_RL     = '#AFFFB5'
    CLR_B_RATE   = '#E8D5FF'

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 9), dpi=200)
    fig.patch.set_facecolor(BG_OUTER)

    for ax in (ax1, ax2):
        ax.set_facecolor(BG_INNER)
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.4, color=GRID_CLR)
        ax.tick_params(colors=TEXT_DIM, labelsize=10)
        for spine in ax.spines.values():
            spine.set_color(GRID_CLR)
            spine.set_linewidth(0.8)

    thresholds = res_a["thresholds"]
    marker_kw = dict(markersize=7, markeredgewidth=1.5)

    # --- PANEL 1: Accuracy vs Threshold ---
    # Approach A (solid markers)
    ax1.plot(thresholds, res_a["hybrid_acc"],
             color=CLR_A_HYBRID, linewidth=2.5, marker='o',
             markerfacecolor=CLR_A_HYBRID, markeredgecolor=BG_OUTER,
             label='Hybrid — A (t10k)', **marker_kw)
    ax1.plot(thresholds, res_a["cnn_acc"],
             color=CLR_A_CNN, linewidth=2.0, marker='s',
             markerfacecolor=CLR_A_CNN, markeredgecolor=BG_OUTER,
             label='CNN-only — A', linestyle='--', **marker_kw)
    ax1.plot(thresholds, res_a["rl_acc"],
             color=CLR_A_RL, linewidth=2.0, marker='^',
             markerfacecolor=CLR_A_RL, markeredgecolor=BG_OUTER,
             label='RL-only — A', linestyle=':', **marker_kw)

    # Approach B (open markers)
    ax1.plot(thresholds, res_b["hybrid_acc"],
             color=CLR_B_HYBRID, linewidth=2.5, marker='D',
             markerfacecolor=CLR_B_HYBRID, markeredgecolor=BG_OUTER,
             label='Hybrid — B (90%)', **marker_kw)
    ax1.plot(thresholds, res_b["cnn_acc"],
             color=CLR_B_CNN, linewidth=2.0, marker='v',
             markerfacecolor=CLR_B_CNN, markeredgecolor=BG_OUTER,
             label='CNN-only — B', linestyle='--', **marker_kw)
    ax1.plot(thresholds, res_b["rl_acc"],
             color=CLR_B_RL, linewidth=2.0, marker='P',
             markerfacecolor=CLR_B_RL, markeredgecolor=BG_OUTER,
             label='RL-only — B', linestyle=':', **marker_kw)

    ax1.fill_between(thresholds, res_a["hybrid_acc"], alpha=0.10, color=CLR_A_HYBRID)
    ax1.fill_between(thresholds, res_b["hybrid_acc"], alpha=0.08, color=CLR_B_HYBRID)

    ax1.set_xlabel('Mahalanobis Threshold', fontsize=13, color=TEXT_CLR, fontweight='bold', labelpad=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=13, color=TEXT_CLR, fontweight='bold', labelpad=12)
    ax1.set_title('Panel 1 — Accuracy vs. Mahalanobis Threshold', fontsize=14, color=TEXT_BRIGHT, fontweight='bold', pad=18)

    leg1 = ax1.legend(loc='lower right', fontsize=9.5, frameon=True,
                      facecolor=BG_INNER, edgecolor=GRID_CLR,
                      labelcolor=TEXT_CLR, framealpha=0.95)
    leg1.get_frame().set_linewidth(0.8)

    # Best Accuracy annotations
    idx_best_a = int(np.argmax(res_a["hybrid_acc"]))
    ax1.annotate(
        f'Best A: {res_a["hybrid_acc"][idx_best_a]:.2f}%\n(τ={thresholds[idx_best_a]:.1f})',
        xy=(thresholds[idx_best_a], res_a["hybrid_acc"][idx_best_a]),
        xytext=(20, 25), textcoords='offset points',
        fontsize=9, fontweight='bold', color=TEXT_BRIGHT,
        arrowprops=dict(arrowstyle='->', color=CLR_A_HYBRID, lw=1.5),
        bbox=dict(boxstyle='round,pad=0.4', facecolor=BG_OUTER, edgecolor=CLR_A_HYBRID, alpha=0.9))

    idx_best_b = int(np.argmax(res_b["hybrid_acc"]))
    ax1.annotate(
        f'Best B: {res_b["hybrid_acc"][idx_best_b]:.2f}%\n(τ={thresholds[idx_best_b]:.1f})',
        xy=(thresholds[idx_best_b], res_b["hybrid_acc"][idx_best_b]),
        xytext=(-60, -35), textcoords='offset points',
        fontsize=9, fontweight='bold', color=TEXT_BRIGHT,
        arrowprops=dict(arrowstyle='->', color=CLR_B_HYBRID, lw=1.5),
        bbox=dict(boxstyle='round,pad=0.4', facecolor=BG_OUTER, edgecolor=CLR_B_HYBRID, alpha=0.9))

    # --- PANEL 2: RL Intervention Rate vs Threshold ---
    ax2.plot(thresholds, res_a["rl_rate"],
             color=CLR_A_RATE, linewidth=2.5, marker='o',
             markerfacecolor=CLR_A_RATE, markeredgecolor=BG_OUTER,
             label='RL Rate — A (t10k)', **marker_kw)
    ax2.plot(thresholds, res_b["rl_rate"],
             color=CLR_B_RATE, linewidth=2.5, marker='D',
             markerfacecolor=CLR_B_RATE, markeredgecolor=BG_OUTER,
             label='RL Rate — B (90%)', **marker_kw)

    ax2.fill_between(thresholds, res_a["rl_rate"], alpha=0.12, color=CLR_A_RATE)
    ax2.fill_between(thresholds, res_b["rl_rate"], alpha=0.08, color=CLR_B_RATE)

    ax2.set_xlabel('Mahalanobis Threshold', fontsize=13, color=TEXT_CLR, fontweight='bold', labelpad=12)
    ax2.set_ylabel('RL Routing Rate (%)', fontsize=13, color=TEXT_CLR, fontweight='bold', labelpad=12)
    ax2.set_title('Panel 2 — RL Intervention Rate vs. Threshold', fontsize=14, color=TEXT_BRIGHT, fontweight='bold', pad=18)

    leg2 = ax2.legend(loc='upper right', fontsize=10, frameon=True,
                      facecolor=BG_INNER, edgecolor=GRID_CLR,
                      labelcolor=TEXT_CLR, framealpha=0.95)
    leg2.get_frame().set_linewidth(0.8)

    for res, clr in [(res_a, CLR_A_RATE), (res_b, CLR_B_RATE)]:
        ax2.annotate(f'{res["rl_rate"][0]:.1f}%',
                     xy=(thresholds[0], res["rl_rate"][0]),
                     xytext=(10, 12), textcoords='offset points',
                     fontsize=8.5, fontweight='bold', color=clr)
        ax2.annotate(f'{res["rl_rate"][-1]:.1f}%',
                     xy=(thresholds[-1], res["rl_rate"][-1]),
                     xytext=(-30, -18), textcoords='offset points',
                     fontsize=8.5, fontweight='bold', color=clr)

    fig.suptitle(
        'Hybrid Vision Pipeline — Global Benchmark\n'
        'Mahalanobis Threshold Sweep  ·  Mixed-Noise Dataset  ·  '
        f'Noise Bands: {NOISE_LEVELS}',
        fontsize=16, color=TEXT_BRIGHT, fontweight='bold',
        y=1.02)

    plt.tight_layout(pad=3.0)
    fig.savefig(output_path, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
    plt.close(fig)
    logger.info(f"\n  ✓ Global evaluation plot saved to: {output_path}")

def print_methodological_conclusion(res_a: dict, res_b: dict):
    """Prints a comparative analysis summarizing conclusions from A vs B."""
    idx_opt_a = int(np.argmax(res_a["hybrid_acc"]))
    idx_opt_b = int(np.argmax(res_b["hybrid_acc"]))

    t_opt_a = res_a["thresholds"][idx_opt_a]
    t_opt_b = res_b["thresholds"][idx_opt_b]

    best_a = res_a["hybrid_acc"][idx_opt_a]
    best_b = res_b["hybrid_acc"][idx_opt_b]

    cnn_a = res_a["cnn_acc"][0]
    cnn_b = res_b["cnn_acc"][0]
    rl_a = res_a["rl_acc"][0]
    rl_b = res_b["rl_acc"][0]

    rate_a = res_a["rl_rate"][idx_opt_a]
    rate_b = res_b["rl_rate"][idx_opt_b]

    gain_a = best_a - cnn_a
    gain_b = best_b - cnn_b

    logger.info(f"\n{'═' * 70}")
    logger.info("  METHODOLOGICAL CONCLUSION — COMPARATIVE ANALYSIS")
    logger.info(f"{'═' * 70}")

    logger.info(f"""
  ┌─────────────────────────────────────────────────────────────────────┐
  │  APPROACH A — Native Test Set (t10k, 10,000 samples)              │
  │  ─────────────────────────────────────────────────────────────────  │
  │  Optimal Threshold (τ*):  {t_opt_a:>6.1f}                                │
  │  Hybrid Accuracy:         {best_a:>6.2f}%                               │
  │  CNN-only Baseline:       {cnn_a:>6.2f}%                               │
  │  RL-only Baseline:        {rl_a:>6.2f}%                               │
  │  RL Routing Rate at τ*:   {rate_a:>6.2f}%                               │
  │  Hybrid Gain over CNN:    {gain_a:>+6.2f} pp                              │
  ├─────────────────────────────────────────────────────────────────────┤
  │  APPROACH B — Disjoint 90% Partition (54,000 samples)             │
  │  ─────────────────────────────────────────────────────────────────  │
  │  Optimal Threshold (τ*):  {t_opt_b:>6.1f}                                │
  │  Hybrid Accuracy:         {best_b:>6.2f}%                               │
  │  CNN-only Baseline:       {cnn_b:>6.2f}%                               │
  │  RL-only Baseline:        {rl_b:>6.2f}%                               │
  │  RL Routing Rate at τ*:   {rate_b:>6.2f}%                               │
  │  Hybrid Gain over CNN:    {gain_b:>+6.2f} pp                              │
  └─────────────────────────────────────────────────────────────────────┘""")

    if abs(t_opt_a - t_opt_b) <= 2.5:
        agreement = "STRONG"
        msg = (f"Both scenarios converge to similar thresholds "
               f"(A: τ={t_opt_a:.1f}, B: τ={t_opt_b:.1f}), "
               f"validating the operational stability of the system.")
    else:
        agreement = "DIVERGENT"
        msg = (f"Optimal thresholds diverge "
               f"(A: τ={t_opt_a:.1f}, B: τ={t_opt_b:.1f}). "
               f"Distribution differences between partitions might explain this.")

    logger.info(f"""
  ┌─────────────────────────────────────────────────────────────────────┐
  │  CROSS ANALYSIS                                                   │
  │  ─────────────────────────────────────────────────────────────────  │
  │  Threshold Agreement:     {agreement:<15}                           │
  │  Delta Accuracy (B - A):  {best_b - best_a:>+6.2f} pp                              │
  │  Delta CNN Baseline (B - A): {cnn_b - cnn_a:>+6.2f} pp                              │
  │  Delta RL Baseline (B - A):  {rl_b - rl_a:>+6.2f} pp                              │
  │                                                                     │
  │  {msg:<65} │
  └─────────────────────────────────────────────────────────────────────┘""")

    consensus_t = (t_opt_a + t_opt_b) / 2.0
    consensus_t = round(consensus_t / 2.5) * 2.5
    logger.info(f"""
  ┌─────────────────────────────────────────────────────────────────────┐
  │  PRODUCTION RECOMMENDATION                                        │
  │  ─────────────────────────────────────────────────────────────────  │
  │  Consensus Threshold (τ): {consensus_t:>6.1f}                                │
  │  Balanced consensus value rounding to the nearest 2.5 step.        │
  │  Optimizes precision while preserving robust statistical borders.   │
  └─────────────────────────────────────────────────────────────────────┘
""")

def main():
    # 1. Configure Hardware
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logger.info(f"  GPU(s) detected: {len(gpus)}")
    else:
        logger.info("  No GPU detected - using CPU.")

    logger.info(f"\n{'═' * 70}")
    logger.info("  GLOBAL EVALUATION OF HYBRID PIPELINE — FULL SWEEP")
    logger.info(f"{'═' * 70}")
    logger.info(f"  Noise Bands:       {NOISE_LEVELS}")
    logger.info(f"  Threshold Sweep:   {THRESHOLDS[0]:.1f} → {THRESHOLDS[-1]:.1f} (step 2.5, {len(THRESHOLDS)} points)")
    logger.info(f"  CNN Batch Size:    {BATCH_SIZE_CNN}")
    logger.info(f"  RL Batch Size:     {BATCH_SIZE_RL}")

    # 2. Load CNN
    logger.info("\nLoading CNN (128D Feature Extractor)...")
    cnn = RawModel()
    ckpt = tf.train.Checkpoint(model=cnn)
    latest_ckpt = tf.train.latest_checkpoint(CHECKPOINT_DIR)
    if not latest_ckpt:
        logger.error(f"ERROR: Checkpoint not found in {CHECKPOINT_DIR}")
        sys.exit(1)
    ckpt.restore(latest_ckpt).expect_partial()
    logger.info("  ✓ CNN loaded successfully.")

    # 3. Load Mahalanobis Profiles
    logger.info("Loading Mahalanobis Profiles...")
    if not os.path.exists(MAHALANOBIS_PROFILES_PATH):
        logger.error(f"ERROR: Mahalanobis profiles not found at {MAHALANOBIS_PROFILES_PATH}")
        sys.exit(1)
    profiles = np.load(MAHALANOBIS_PROFILES_PATH, allow_pickle=True)
    logger.info("  ✓ Screening profiles loaded.")

    # 4. Load RL Agent (k-NN Bandit 128D)
    logger.info("Loading RL Agent (k-NN Bandit 128D)...")
    agent = KNNBanditAgent128D(k=30, n_actions=10, latent_dim=128, use_pca=True, pca_components=48)
    if not os.path.exists(MEMORY_BANK_128D_PATH):
        logger.error(f"ERROR: 128D memory bank not found at {MEMORY_BANK_128D_PATH}")
        sys.exit(1)
    agent.load(MEMORY_BANK_128D_PATH)
    stats = agent.get_memory_stats()
    logger.info(f"  ✓ RL Agent loaded ({stats['size']:,} experiences, reward_mean={stats['reward_mean']:.3f}).")

    # 5. APPROACH A — Dataset t10k (10,000 samples)
    logger.info("\nLoading t10k dataset...")
    x_t10k, y_t10k = load_mnist_raw(DATA_DIR, kind='t10k')
    x_t10k = x_t10k.astype(np.float32) / 255.0
    logger.info(f"  ✓ t10k dataset loaded: {len(x_t10k):,} samples.")

    res_a = evaluate_approach("APPROACH A — Native Test Set (t10k, 10,000 samples)",
                              x_t10k, y_t10k, cnn, agent, profiles)

    # 6. APPROACH B — Disjoint 90% Partition (54,000 samples)
    logger.info("\nLoading training dataset for 90% split...")
    x_full, y_full = load_mnist_raw(DATA_DIR, kind='train')
    x_full = x_full.astype(np.float32) / 255.0

    _, x_eval_90, _, y_eval_90 = train_test_split(
        x_full, y_full,
        test_size=0.90,
        random_state=RANDOM_SEED,
        shuffle=True,
        stratify=y_full
    )
    logger.info(f"  ✓ Disjoint 90% evaluation partition: {len(x_eval_90):,} samples.")

    res_b = evaluate_approach("APPROACH B — Disjoint 90% Partition (54,000 samples)",
                              x_eval_90, y_eval_90, cnn, agent, profiles)

    # 7. Generate Plot
    logger.info("\nGenerating global comparison plot...")
    generate_comparative_plot(res_a, res_b, OUTPUT_FIGURE_PATH)

    # 8. Methodological Conclusions
    print_methodological_conclusion(res_a, res_b)

    logger.info("Global evaluation benchmark sweep completed successfully!")

if __name__ == "__main__":
    main()
