"""
AutoML Experiment Runner CLI (CIFAR-100 Edge AI Case Study).

Main entry point for multi-objective hyperparameter optimization:
Samples configurations from SearchSpace (random or grid),
executes isolated trials with TrialRunner, records results and computes
Pareto rankings with ResultsTracker, and extracts the optimal solution
into outputs/cifar100/experiments/best_solution/.
"""

from __future__ import annotations
import os
import sys
import argparse
import logging
from typing import List, Dict, Any

# Ensure Python with TensorFlow is used if launched from base Anaconda
try:
    import tensorflow as tf
except ModuleNotFoundError:
    tf_python = r"C:\Users\sanfr\.conda\envs\tf_l40s\python.exe"
    if os.path.exists(tf_python) and sys.executable.lower() != tf_python.lower():
        import subprocess
        res = subprocess.run([tf_python] + sys.argv)
        sys.exit(res.returncode)

# Ensure project root in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.search_space import SearchSpace
from experiments.trial_runner import TrialRunner
from experiments.results_tracker import ResultsTracker

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_experiment")


def print_trials_summary_table(tracker: ResultsTracker) -> None:
    """Prints a formatted summary table of all completed trials and rankings."""
    trials = tracker.trials
    if not trials:
        return

    logger.info("\n" + "=" * 105)
    logger.info("  AUTOML EXPERIMENT SUMMARY & MULTI-OBJECTIVE PARETO RANKINGS")
    logger.info("=" * 105)

    headers = [
        "Trial",
        "Score",
        "Clean Acc",
        "Noisy Acc",
        "D_KL",
        "Latency",
        "RAM (MB)",
        "Latent",
        "Capacity",
        "k",
        "tau_M",
        "tau_H",
    ]
    header_str = (
        f"| {headers[0]:<9} | {headers[1]:<7} | {headers[2]:<9} | {headers[3]:<9} | "
        f"{headers[4]:<7} | {headers[5]:<10} | {headers[6]:<8} | {headers[7]:<6} | "
        f"{headers[8]:<8} | {headers[9]:<3} | {headers[10]:<6} | {headers[11]:<6} |"
    )
    sep_str = "|-" + "-|-".join(["-" * len(h) for h in [
        "Trial    ", "Score  ", "Clean Acc", "Noisy Acc", "D_KL   ", "Latency   ", "RAM (MB)", "Latent", "Capacity", "k  ", "tau_M ", "tau_H "
    ]]) + "-|"

    logger.info(header_str)
    logger.info(sep_str)

    best_trial_id = tracker.get_best_trial()["trial_id"]

    for t in sorted(trials, key=lambda x: float(x["score"]), reverse=True):
        tid = t["trial_id"]
        is_best = " *" if tid == best_trial_id else "  "
        m = t["metrics"]
        c = t["config"]
        row_str = (
            f"| {tid + is_best:<9} | {t['score']:<7.4f} | {m.get('clean_accuracy', 0.0)*100:<8.2f}% | "
            f"{m.get('noisy_accuracy', 0.0)*100:<8.2f}% | {m.get('d_kl_eviction', 0.0):<7.4f} | "
            f"{m.get('inference_latency_ms', 0.0):<7.2f} ms | {m.get('ram_footprint_mb', 0.0):<8.2f} | "
            f"{c.get('latent_dim', 128):<6} | {c.get('memory_capacity', 5000):<8} | "
            f"{c.get('knn_k', 10):<3} | {c.get('mahalanobis_threshold', 15.0):<6.1f} | "
            f"{c.get('entropy_threshold', 2.0):<6.1f} |"
        )
        logger.info(row_str)

    logger.info("=" * 105 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run AutoML experiments for CIFAR-100 Edge AI Case Study."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar100",
        choices=["cifar100", "cifar10", "mnist"],
        help="Dataset name (default: cifar100).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="random",
        choices=["random", "grid"],
        help="Search mode: random sampling or Cartesian grid (default: random).",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=10,
        help="Number of trials to execute (default: 10).",
    )
    parser.add_argument(
        "--fast-screening",
        action="store_true",
        default=True,
        help="Execute accelerated screening (5 CNN epochs, 2,000 RL steps). Default: True.",
    )
    parser.add_argument(
        "--full-convergence",
        action="store_true",
        help="Execute full convergence mode (50 CNN epochs, 50,000 RL steps).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Base output directory for experiments (default: outputs/{dataset}/experiments).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global random seed (default: 42).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override CNN training epochs per trial.",
    )
    parser.add_argument(
        "--sim-steps",
        type=int,
        default=None,
        help="Override streaming RL simulation steps per trial.",
    )
    parser.add_argument(
        "--train-samples",
        type=int,
        default=None,
        help="Optional training set subset size per trial for ultra-fast screening.",
    )

    args = parser.parse_args()

    # Determine execution mode
    fast_screening = not args.full_convergence
    if not fast_screening:
        logger.info("Operating in FULL CONVERGENCE mode (50 epochs, 50,000 steps).")
    else:
        logger.info("Operating in FAST SCREENING mode (5 epochs, 2,000 steps).")

    output_dir = args.output_dir or os.path.join(
        PROJECT_ROOT, "outputs", args.dataset, "experiments"
    )
    os.makedirs(output_dir, exist_ok=True)

    # 1. Initialize Search Space
    search_space = SearchSpace(seed=args.seed)
    if args.mode == "random":
        configurations = search_space.sample_n_random(args.n_trials)
    else:
        configurations = search_space.get_grid()[: args.n_trials]

    logger.info("=" * 80)
    logger.info(f"STARTING AUTOML CAMPAIGN: {len(configurations)} TRIALS")
    logger.info(f"Dataset: {args.dataset} | Mode: {args.mode} | Output: {output_dir}")
    logger.info("=" * 80)

    # 2. Results Tracker Initialization
    tracker = ResultsTracker(output_dir=output_dir)

    # 3. Trial Execution Loop
    for idx, config in enumerate(configurations):
        trial_id = f"trial_{idx + 1:03d}"
        config_with_meta = dict(config)
        config_with_meta["dataset"] = args.dataset
        config_with_meta["seed"] = args.seed + idx

        if args.epochs is not None:
            config_with_meta["epochs"] = args.epochs
        if args.sim_steps is not None:
            config_with_meta["sim_steps"] = args.sim_steps
        if args.train_samples is not None:
            config_with_meta["train_samples"] = args.train_samples

        logger.info(f"\n>>> Running Trial {idx + 1}/{len(configurations)}: {trial_id} <<<")

        runner = TrialRunner(
            trial_id=trial_id,
            config=config_with_meta,
            output_base_dir=output_dir,
        )

        try:
            trial_result = runner.run(fast_screening=fast_screening)
            tracker.add_trial(
                trial_id=trial_id,
                config=config_with_meta,
                metrics=trial_result["metrics"],
                artifacts_path=trial_result["artifacts_path"],
            )
            logger.info(f"✓ {trial_id} Finished successfully.")
        except Exception as e:
            logger.error(f"✗ {trial_id} Failed with exception: {e}", exc_info=True)

    # 4. Extract and Finalize Best Solution
    if tracker.trials:
        best_trial = tracker.extract_best()
        print_trials_summary_table(tracker)
        logger.info(f"AutoML Campaign Completed. Best solution: {best_trial['trial_id']}")
        logger.info(f"Artifacts saved in: {tracker.best_solution_dir}")
    else:
        logger.warning("No trials completed successfully.")


if __name__ == "__main__":
    main()
