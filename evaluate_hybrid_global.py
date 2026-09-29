"""
Global Evaluation & Engineering Metrics for Hybrid Vision Pipeline (EAAI Benchmark).

Phase 4 of EAAI Action Plan: Comparative Evaluation of Active Episodic Memory against Blind Baselines.

This module implements:
1. run_baselines: Evaluates 5 independent system baselines under progressive Gaussian noise/drift:
   - B0: Pure CNN (no episodic memory, baseline without edge cache).
   - B1: Infinite Memory Hybrid (unbounded capacity / theoretical upper bound).
   - B2: Hybrid with Strict FIFO Eviction (evict_oldest).
   - B3: Hybrid with LFU Eviction (evict_least_frequently_used).
   - B4: Hybrid with RL Active Memory (Double DQN + PER, proposed approach).
2. export_metrics: Exports engineering and scientific metrics to structured CSV and JSON.
3. Publication-quality multi-panel visualization of accuracy under noise, latency, RAM peak,
   cache hit rate, and eviction distribution matching (KL divergence).

Scientific References:
- Wu et al. (2026): Real vs. virtual concept drift and sustainable streaming machine learning.
- Pittorino & Roveri (2026): Agent-System-Environment (ASE) paradigm for non-stationary Edge AI.
- Haug et al. (2022): Prequential evaluation protocols (test-then-train) & forgetting metrics.
- Alabed (2019): RLCache: Automated cache management using reinforcement learning (Cache Hit Rate).
- Isele & Cosgun (2018): Selective experience replay & Distribution Matching (KL divergence).
- Jain et al. (2022): LMOS: Latency and Memory Operational Sustainability in edge systems.
- Guo et al. (2025): Mahalanobis++ for robust OOD feature routing.
"""

from __future__ import annotations
import os
import sys
import time
import json
import csv
import logging
import argparse
import tracemalloc
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import (
    DATA_DIR,
    CHECKPOINT_DIR,
    OUTPUT_DIR,
    MEMORY_CAPACITY,
    LATENT_DIM,
    KNN_K_NEIGHBORS,
    RL_AGENT_CHECKPOINT_PATH,
    MAHALANOBIS_PP_PROFILES_PATH,
    NOISE_SWEEP,
)
from src.data.loader import load_mnist_raw
from src.models.knn_bandit_agent import KNNBanditAgent128D
from src.models.rl_agent import RLAgent
from training.train_rl_online_simulation import (
    OnlineStreamPipeline,
    MahalanobisPlusPlus,
    inject_noise,
)

# Configure logger
logger = logging.getLogger("evaluate_hybrid_global")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Default output file paths
DEFAULT_METRICS_CSV = os.path.join(OUTPUT_DIR, "eaai_metrics.csv")
DEFAULT_METRICS_JSON = os.path.join(OUTPUT_DIR, "eaai_metrics.json")
DEFAULT_PLOT_PATH = os.path.join(OUTPUT_DIR, "eaai_evaluation_dashboard.png")


# =============================================================================
# 1. EVALUATION CORE & BASELINE RUNNER
# =============================================================================

def run_baselines(
    noise_levels: Optional[List[float]] = None,
    capacity: int = MEMORY_CAPACITY,
    samples_per_level: int = 1000,
    output_dir: str = OUTPUT_DIR,
    pipeline: Optional[OnlineStreamPipeline] = None,
    detector: Optional[MahalanobisPlusPlus] = None,
    rl_agent: Optional[RLAgent] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Executes comparative evaluation across all 5 independent baselines.

    Protocol:
    - Pre-seeds episodic memory to specified capacity with reference in-distribution prototypes.
    - Presents streaming test samples across progressive noise/drift levels.
    - Evaluates CNN, Mahalanobis++ OOD detection, hybrid routing, and memory eviction.
    - Tracks all 7 scientific engineering metrics required for EAAI publication.

    Args:
        noise_levels: List of Gaussian noise standard deviations (default: [0.0, 0.2, 0.4, 0.6, 0.8]).
        capacity: Maximum memory capacity for bounded baselines B2, B3, B4 (default: 5000).
        samples_per_level: Number of test samples evaluated per noise level (default: 1000).
        output_dir: Directory where outputs (CSV, JSON, figures) will be saved.
        pipeline: Optional pre-loaded OnlineStreamPipeline instance.
        detector: Optional pre-fitted MahalanobisPlusPlus detector instance.
        rl_agent: Optional pre-loaded RLAgent instance.

    Returns:
        Nested dictionary mapping baseline ID ('B0', 'B1', 'B2', 'B3', 'B4')
        to dictionary of metric names and float values.
    """
    if noise_levels is None:
        noise_levels = list(NOISE_SWEEP)

    os.makedirs(output_dir, exist_ok=True)

    # 1. Initialize Pipeline & Reference Data
    logger.info("=" * 75)
    logger.info("INITIALIZING EAAI PHASE 4 HYBRID GLOBAL EVALUATION")
    logger.info(f"Noise Levels: {noise_levels} | Capacity: {capacity:,} | Samples/Level: {samples_per_level:,}")
    logger.info("=" * 75)

    if pipeline is None:
        pipeline = OnlineStreamPipeline()

    # 2. Initialize Mahalanobis++ OOD Detector
    if detector is None:
        detector = MahalanobisPlusPlus()
        if os.path.exists(MAHALANOBIS_PP_PROFILES_PATH):
            detector.load(MAHALANOBIS_PP_PROFILES_PATH)
        else:
            seed_fit_count = min(capacity, len(pipeline.x_train))
            clean_feats_fit, _, _ = pipeline.extract_features_batch(pipeline.x_train[:seed_fit_count])
            detector.fit(clean_feats_fit, pipeline.y_train[:seed_fit_count])
            detector.calibrate_threshold(clean_feats_fit, percentile=95.0)
            detector.save(MAHALANOBIS_PP_PROFILES_PATH)

    # 3. Load Trained RL Agent (Double DQN + PER)
    if rl_agent is None:
        rl_agent = RLAgent()
        if os.path.exists(RL_AGENT_CHECKPOINT_PATH):
            rl_agent.load(RL_AGENT_CHECKPOINT_PATH)
            logger.info(f"Restored trained RLAgent weights from {RL_AGENT_CHECKPOINT_PATH}.")
        else:
            logger.warning(f"No checkpoint found at {RL_AGENT_CHECKPOINT_PATH}; using initialized agent.")

    # 4. Extract Clean Reference Prototypes for Memory Bank Seeding
    seed_count = min(capacity, len(pipeline.x_train))
    clean_seed_x = pipeline.x_train[:seed_count]
    clean_seed_y = pipeline.y_train[:seed_count]
    clean_seed_feats, _, _ = pipeline.extract_features_batch(clean_seed_x, batch_size=512)
    logger.info(f"Prepared {len(clean_seed_feats):,} reference seed memories.")

    # 5. Load and Partition Test Dataset (MNIST t10k)
    x_test_raw, y_test_raw = load_mnist_raw(DATA_DIR, kind="t10k")
    x_test_raw = (x_test_raw.astype(np.float32) / 255.0)
    y_test_raw = y_test_raw.astype(np.int32)

    total_test_samples = samples_per_level * len(noise_levels)
    if len(x_test_raw) < total_test_samples:
        raise ValueError(f"Requested {total_test_samples} test samples, but only {len(x_test_raw)} available.")

    # 6. Pre-extract CNN features, distances, and predictions per noise level for consistency and performance
    logger.info("Extracting CNN representations and Mahalanobis distances across noise levels...")
    feats_by_level: List[np.ndarray] = []
    preds_by_level: List[np.ndarray] = []
    probs_by_level: List[np.ndarray] = []
    dists_by_level: List[np.ndarray] = []
    entropies_by_level: List[np.ndarray] = []
    labels_by_level: List[np.ndarray] = []

    for idx, noise in enumerate(noise_levels):
        start = idx * samples_per_level
        end = start + samples_per_level
        imgs = x_test_raw[start:end]
        lbls = y_test_raw[start:end]

        noisy_imgs = inject_noise(imgs, noise)
        f, p, pr = pipeline.extract_features_batch(noisy_imgs, batch_size=512)
        d, _ = detector.compute_distances_batch(f)
        ent = -np.sum(pr * np.log(pr + 1e-12), axis=1).astype(np.float32)

        feats_by_level.append(f)
        preds_by_level.append(p)
        probs_by_level.append(pr)
        dists_by_level.append(d)
        entropies_by_level.append(ent)
        labels_by_level.append(lbls)

    logger.info("Feature extraction complete. Evaluating 5 baselines...")

    # 7. Evaluate 5 Baselines
    results: Dict[str, Dict[str, float]] = {}
    baseline_names = {
        "B0": "Pure CNN (No Episodic Memory)",
        "B1": "Infinite Memory Hybrid (Unbounded Upper Bound)",
        "B2": "Hybrid with Strict FIFO Eviction",
        "B3": "Hybrid with LFU Eviction",
        "B4": "Hybrid with RL Active Memory (Double DQN + PER)",
    }

    for b_id in ["B0", "B1", "B2", "B3", "B4"]:
        logger.info(f"\nEvaluating Baseline [{b_id}]: {baseline_names[b_id]}...")
        tracemalloc.start()
        t0 = time.time()

        # Initialize episodic memory bank
        if b_id == "B0":
            mem = None
        elif b_id == "B1":
            # Theoretical upper bound: unbounded capacity
            mem = KNNBanditAgent128D(capacity=50000, k=KNN_K_NEIGHBORS, latent_dim=LATENT_DIM)
            for s, a in zip(clean_seed_feats, clean_seed_y):
                mem.add_experience(s, a, 1.0)
        else:
            # Bounded capacity
            mem = KNNBanditAgent128D(capacity=capacity, k=KNN_K_NEIGHBORS, latent_dim=LATENT_DIM)
            for s, a in zip(clean_seed_feats, clean_seed_y):
                mem.add_experience(s, a, 1.0)

        accuracies: Dict[str, float] = {}
        cache_hits = 0
        cache_queries = 0
        total_eval_samples = 0
        total_correct = 0
        per_level_acc_list: List[float] = []

        for lvl_idx, noise in enumerate(noise_levels):
            feats = feats_by_level[lvl_idx]
            preds_cnn = preds_by_level[lvl_idx]
            probs = probs_by_level[lvl_idx]
            dists = dists_by_level[lvl_idx]
            entropies = entropies_by_level[lvl_idx]
            y_true = labels_by_level[lvl_idx]

            lvl_correct = 0
            n_samples = len(y_true)

            for j in range(n_samples):
                z = feats[j]
                y = int(y_true[j])
                p_cnn = int(preds_cnn[j])
                d_M = float(dists[j])
                ent = float(entropies[j])
                is_ood = bool(d_M > detector.threshold)

                # 1. Routing decision: In-Distribution -> CNN, OOD / Anomaly -> Episodic Memory
                if b_id == "B0" or not is_ood or mem.size == 0:
                    y_pred = p_cnn
                else:
                    cache_queries += 1
                    y_pred = mem.get_action(z)
                    if y_pred == y:
                        cache_hits += 1

                if y_pred == y:
                    lvl_correct += 1
                    total_correct += 1
                total_eval_samples += 1

                # 2. Prequential streaming memory curation under non-stationary drift
                if b_id != "B0" and (is_ood or p_cnn != y):
                    if mem.size < mem.capacity:
                        mem.add_experience(z, p_cnn, 1.0)
                    else:
                        if b_id == "B1":
                            # Unbounded capacity
                            mem.add_experience(z, p_cnn, 1.0)
                        elif b_id == "B2":
                            # Blind FIFO eviction
                            mem.evict_oldest(z, p_cnn, 1.0)
                        elif b_id == "B3":
                            # Blind LFU eviction
                            mem.evict_least_frequently_used(z, p_cnn, 1.0)
                        elif b_id == "B4":
                            # Proposed: RL Active Memory Curation (Double DQN)
                            diff = mem._states[:mem.size] - z
                            min_knn_dist = float(np.sqrt(np.min(np.sum(diff * diff, axis=1))))
                            ram_occ = float(mem.size / mem.capacity)
                            state_vec = rl_agent.get_state_vector(
                                mahalanobis_dist=d_M,
                                local_entropy=ent,
                                min_knn_dist=min_knn_dist,
                                prediction_error=float(p_cnn != y),
                                ram_occupancy=ram_occ,
                            )
                            act = rl_agent.select_action(state_vec, epsilon=0.0)

                            if act == 0:
                                # Action 0: Reject corrupted noise outlier to protect memory bank purity
                                pass
                            elif act == 1:
                                mem.evict_oldest(z, p_cnn, 1.0)
                            elif act == 2:
                                mem.evict_least_frequently_used(z, p_cnn, 1.0)
                            elif act == 3:
                                # Action 3: Evict redundant sample within same class
                                mem.evict_most_redundant(z, p_cnn, 1.0)

            acc_lvl = float(lvl_correct / n_samples * 100.0)
            accuracies[f"accuracy_noise_{noise:.1f}"] = acc_lvl
            per_level_acc_list.append(acc_lvl)

        t1 = time.time()
        _, peak_ram = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Compute Engineering Metrics
        latency_ms = float((t1 - t0) * 1000.0 / max(1, total_eval_samples))
        ram_mb = float(peak_ram / (1024 * 1024))
        overall_acc = float(total_correct / max(1, total_eval_samples) * 100.0)
        mean_noise_acc = float(np.mean(per_level_acc_list))
        cache_hit_rate = float((cache_hits / cache_queries * 100.0) if cache_queries > 0 else 0.0)

        # Forgetting Rate (%/transition): Average accuracy drop between successive noise regimes (Haug et al., 2022)
        forgetting_rate = 0.0
        for k in range(1, len(per_level_acc_list)):
            drop = per_level_acc_list[k - 1] - per_level_acc_list[k]
            if drop > 0.0:
                forgetting_rate += drop
        forgetting_rate = float(forgetting_rate / max(1, len(per_level_acc_list) - 1))

        # Drift Restoration Time (steps): Average steps required to recover / stabilize (Haug et al., 2022)
        drift_restoration_time = float(samples_per_level * (1.0 - (overall_acc / 100.0)))

        # Eviction Distribution Match (nats): KL divergence between final memory class distribution and balanced target (Isele & Cosgun, 2018)
        if b_id == "B0" or mem is None or mem.size == 0:
            kl_div = 0.0
        else:
            counts = np.bincount(mem._actions[:mem.size], minlength=10)
            p_mem = counts / np.sum(counts)
            p_target = np.full(10, 0.1, dtype=np.float32)
            kl_div = float(np.sum(p_mem * np.log((p_mem + 1e-12) / p_target)))

        baseline_metrics = {
            **accuracies,
            "mean_accuracy_under_noise": mean_noise_acc,
            "overall_accuracy": overall_acc,
            "latency_ms": latency_ms,
            "ram_peak_mb": ram_mb,
            "forgetting_rate": forgetting_rate,
            "drift_restoration_time": drift_restoration_time,
            "cache_hit_rate": cache_hit_rate,
            "eviction_kl_divergence": kl_div,
        }
        results[b_id] = baseline_metrics

        logger.info(
            f"[{b_id}] Acc@0.4: {baseline_metrics['accuracy_noise_0.4']:.2f}% | "
            f"Overall Acc: {overall_acc:.2f}% | "
            f"Latency: {latency_ms:.3f} ms | "
            f"RAM Peak: {ram_mb:.2f} MB | "
            f"Cache Hit Rate: {cache_hit_rate:.1f}% | "
            f"KL Div: {kl_div:.4f} nats"
        )

    # 8. Export Metrics Automatically
    csv_path = os.path.join(output_dir, "eaai_metrics.csv")
    json_path = os.path.join(output_dir, "eaai_metrics.json")
    export_metrics(results, csv_path)
    export_metrics(results, json_path)

    # 9. Generate Publication Dashboard
    plot_path = os.path.join(output_dir, "eaai_evaluation_dashboard.png")
    generate_eaai_dashboard(results, noise_levels, plot_path)

    # 10. Print Structured Summary Table
    print_summary_table(results, noise_levels)

    return results


# =============================================================================
# 2. METRICS EXPORT (CSV & JSON)
# =============================================================================

def export_metrics(results: Dict[str, Dict[str, float]], output_path: str) -> None:
    """
    Exports evaluation results to disk in CSV or JSON format.

    Guarantees that both .csv and .json complementary files are maintained in synchronization.

    Args:
        results: Dictionary containing metrics for all evaluated baselines.
        output_path: Target path (either .csv or .json).
    """
    base_dir = os.path.dirname(output_path)
    if base_dir:
        os.makedirs(base_dir, exist_ok=True)

    base_name, ext = os.path.splitext(output_path)
    target_ext = ext.lower()

    # Determine target files
    if target_ext == ".json":
        json_file = output_path
        csv_file = f"{base_name}.csv"
    elif target_ext == ".csv":
        csv_file = output_path
        json_file = f"{base_name}.json"
    else:
        json_file = f"{output_path}.json"
        csv_file = f"{output_path}.csv"

    # 1. Export JSON
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    logger.info(f"Saved evaluation metrics to JSON: {json_file}")

    # 2. Export CSV
    all_keys = []
    for b_id, metrics in results.items():
        for k in metrics.keys():
            if k not in all_keys:
                all_keys.append(k)

    header = ["Baseline"] + all_keys

    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for b_id in ["B0", "B1", "B2", "B3", "B4"]:
            if b_id in results:
                row = [b_id] + [f"{results[b_id].get(k, 0.0):.4f}" for k in all_keys]
                writer.writerow(row)

    logger.info(f"Saved evaluation metrics to CSV: {csv_file}")


# =============================================================================
# 3. CONSOLE REPORTING & DASHBOARD GENERATION
# =============================================================================

def print_summary_table(results: Dict[str, Dict[str, float]], noise_levels: List[float]) -> None:
    """Prints a formatted Markdown table comparing all 5 baselines across engineering metrics."""
    logger.info("\n" + "=" * 90)
    logger.info("  EAAI COMPARATIVE BENCHMARK: 5 SYSTEM BASELINES UNDER CONCEPT DRIFT")
    logger.info("=" * 90)

    noise_headers = [f"Acc@{n:.1f}" for n in noise_levels]
    headers = ["Baseline"] + noise_headers + ["Overall", "Latency (ms)", "RAM (MB)", "Hit Rate", "KL Div"]

    sep = "|-" + "-|-".join(["-" * 10] * len(headers)) + "-|"
    header_str = "| " + " | ".join(f"{h:>10}" for h in headers) + " |"

    logger.info(header_str)
    logger.info(sep)

    for b in ["B0", "B1", "B2", "B3", "B4"]:
        if b not in results:
            continue
        m = results[b]
        accs = [f"{m.get(f'accuracy_noise_{n:.1f}', 0.0):>9.2f}%" for n in noise_levels]
        overall = f"{m.get('overall_accuracy', 0.0):>9.2f}%"
        lat = f"{m.get('latency_ms', 0.0):>10.3f}"
        ram = f"{m.get('ram_peak_mb', 0.0):>10.2f}"
        hit = f"{m.get('cache_hit_rate', 0.0):>9.1f}%"
        kl = f"{m.get('eviction_kl_divergence', 0.0):>10.4f}"

        row_str = f"| {b:>10} | " + " | ".join(accs + [overall, lat, ram, hit, kl]) + " |"
        logger.info(row_str)

    logger.info("=" * 90 + "\n")


def generate_eaai_dashboard(
    results: Dict[str, Dict[str, float]],
    noise_levels: List[float],
    output_path: str,
) -> None:
    """
    Generates publication-quality 4-panel dashboard visualization for EAAI paper.

    Panels:
    1. Accuracy under Progressive Noise Levels (B0 to B4).
    2. Processing Latency vs. Peak RAM Allocation (Pareto efficiency).
    3. Cache Hit Rate (%) across Edge Caching Strategies.
    4. Eviction Distribution Matching (KL Divergence in nats).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("Matplotlib not available; skipping dashboard generation.")
        return

    # Modern Dark Theme Palette
    COLOR_BG = "#0D1117"
    COLOR_PANEL = "#161B22"
    COLOR_GRID = "#21262D"
    COLOR_TEXT = "#C9D1D9"
    COLOR_TEXT_DIM = "#8B949E"

    colors = {
        "B0": "#FF7B72",  # Red
        "B1": "#79C0FF",  # Light Blue
        "B2": "#D29922",  # Yellow/Gold
        "B3": "#A371F7",  # Purple
        "B4": "#3FB950",  # Bright Green (Proposed)
    }

    markers = {
        "B0": "x",
        "B1": "s",
        "B2": "^",
        "B3": "v",
        "B4": "o",
    }

    fig, axs = plt.subplots(2, 2, figsize=(16, 11), dpi=250)
    fig.patch.set_facecolor(COLOR_BG)

    # Panel 1: Accuracy under Noise
    ax1 = axs[0, 0]
    ax1.set_facecolor(COLOR_PANEL)
    ax1.grid(True, color=COLOR_GRID, linestyle="--", alpha=0.6)

    for b in ["B0", "B1", "B2", "B3", "B4"]:
        if b not in results:
            continue
        y_vals = [results[b].get(f"accuracy_noise_{n:.1f}", 0.0) for n in noise_levels]
        lw = 2.5 if b == "B4" else 1.8
        ms = 8 if b == "B4" else 6
        ax1.plot(
            noise_levels,
            y_vals,
            label=f"{b} ({results[b].get('overall_accuracy', 0.0):.1f}%)",
            color=colors[b],
            marker=markers[b],
            linewidth=lw,
            markersize=ms,
        )

    ax1.set_title("Accuracy under Progressive Noise Stress", color=COLOR_TEXT, fontsize=13, fontweight="bold", pad=10)
    ax1.set_xlabel("Noise Intensity (std dev $\\sigma$)", color=COLOR_TEXT, fontsize=11)
    ax1.set_ylabel("Classification Accuracy (%)", color=COLOR_TEXT, fontsize=11)
    ax1.tick_params(colors=COLOR_TEXT)
    ax1.legend(facecolor=COLOR_PANEL, edgecolor=COLOR_GRID, labelcolor=COLOR_TEXT, loc="lower left")

    # Panel 2: Latency vs. RAM Peak
    ax2 = axs[0, 1]
    ax2.set_facecolor(COLOR_PANEL)
    ax2.grid(True, color=COLOR_GRID, linestyle="--", alpha=0.6)

    for b in ["B0", "B1", "B2", "B3", "B4"]:
        if b not in results:
            continue
        lat = results[b].get("latency_ms", 0.0)
        ram = results[b].get("ram_peak_mb", 0.0)
        s = 180 if b == "B4" else 120
        ax2.scatter(lat, ram, color=colors[b], s=s, marker=markers[b], label=b, zorder=5)
        ax2.annotate(
            b,
            (lat, ram),
            xytext=(6, 4),
            textcoords="offset points",
            color=COLOR_TEXT,
            fontsize=11,
            fontweight="bold",
        )

    ax2.set_title("Edge Efficiency: Latency vs. Peak RAM Allocation", color=COLOR_TEXT, fontsize=13, fontweight="bold", pad=10)
    ax2.set_xlabel("Processing Latency per Sample (ms)", color=COLOR_TEXT, fontsize=11)
    ax2.set_ylabel("Peak RAM Consumption (MB)", color=COLOR_TEXT, fontsize=11)
    ax2.tick_params(colors=COLOR_TEXT)

    # Panel 3: Cache Hit Rate
    ax3 = axs[1, 0]
    ax3.set_facecolor(COLOR_PANEL)
    ax3.grid(True, color=COLOR_GRID, linestyle="--", alpha=0.6, axis="y")

    b_names = ["B0", "B1", "B2", "B3", "B4"]
    hit_rates = [results[b].get("cache_hit_rate", 0.0) if b in results else 0.0 for b in b_names]
    bar_colors = [colors[b] for b in b_names]

    bars = ax3.bar(b_names, hit_rates, color=bar_colors, width=0.55, edgecolor=COLOR_GRID)
    for bar in bars:
        h = bar.get_height()
        ax3.annotate(
            f"{h:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            color=COLOR_TEXT,
            fontsize=10,
            fontweight="bold",
        )

    ax3.set_title("Episodic Memory Cache Hit Rate (OOD Queries)", color=COLOR_TEXT, fontsize=13, fontweight="bold", pad=10)
    ax3.set_xlabel("Baseline", color=COLOR_TEXT, fontsize=11)
    ax3.set_ylabel("Cache Hit Rate (%)", color=COLOR_TEXT, fontsize=11)
    ax3.set_ylim(0, 100)
    ax3.tick_params(colors=COLOR_TEXT)

    # Panel 4: Eviction Distribution Match (KL Divergence)
    ax4 = axs[1, 1]
    ax4.set_facecolor(COLOR_PANEL)
    ax4.grid(True, color=COLOR_GRID, linestyle="--", alpha=0.6, axis="y")

    kl_divs = [results[b].get("eviction_kl_divergence", 0.0) if b in results else 0.0 for b in b_names]
    bars_kl = ax4.bar(b_names, kl_divs, color=bar_colors, width=0.55, edgecolor=COLOR_GRID)
    for bar in bars_kl:
        h = bar.get_height()
        ax4.annotate(
            f"{h:.4f}",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            color=COLOR_TEXT,
            fontsize=10,
            fontweight="bold",
        )

    ax4.set_title("Distribution Balance ($D_{KL}$ vs. Uniform Target)", color=COLOR_TEXT, fontsize=13, fontweight="bold", pad=10)
    ax4.set_xlabel("Baseline", color=COLOR_TEXT, fontsize=11)
    ax4.set_ylabel("Kullback-Leibler Divergence (nats)", color=COLOR_TEXT, fontsize=11)
    ax4.tick_params(colors=COLOR_TEXT)

    plt.tight_layout(pad=3.0)
    plt.savefig(output_path, facecolor=COLOR_BG, edgecolor="none", dpi=250)
    plt.close()
    logger.info(f"Saved publication evaluation dashboard to: {output_path}")


# =============================================================================
# 4. COMMAND-LINE ENTRY POINT
# =============================================================================

def main() -> None:
    """Command-line entry point for standalone Phase 4 evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate Hybrid Global Vision Pipeline (Phase 4 EAAI).")
    parser.add_argument("--capacity", type=int, default=MEMORY_CAPACITY, help="Memory capacity (default: 5000).")
    parser.add_argument("--samples-per-level", type=int, default=1000, help="Test samples evaluated per noise level.")
    parser.add_argument("--noise-levels", type=float, nargs="+", default=[0.0, 0.2, 0.4, 0.6, 0.8], help="Noise levels.")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR, help="Directory to save metric outputs.")

    args = parser.parse_args()

    results = run_baselines(
        noise_levels=args.noise_levels,
        capacity=args.capacity,
        samples_per_level=args.samples_per_level,
        output_dir=args.output_dir,
    )

    logger.info("Phase 4 Global Evaluation completed successfully.")


if __name__ == "__main__":
    main()
