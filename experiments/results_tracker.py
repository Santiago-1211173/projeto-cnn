"""
AutoML Results Tracker and Pareto Composite Scoring (CIFAR-100 Case Study).

Maintains records of all hyperparameter exploration trials, performs Min-Max
normalization across the 5 canonical objectives, computes multi-objective composite
Pareto scores with spec-defined weightings, logs summary tables, and exports the
best discovered solution with standardized schemas.
"""

from __future__ import annotations
import os
import sys
import json
import csv
import shutil
import datetime
import logging
from typing import Dict, Any, List, Optional
import numpy as np

logger = logging.getLogger(__name__)

# Canonical metric weights defined in the Master Execution Plan
METRIC_WEIGHTS: Dict[str, float] = {
    "clean_accuracy": 0.30,       # Higher is better
    "noisy_accuracy": 0.25,       # Higher is better
    "d_kl_eviction": 0.20,        # Lower is better
    "inference_latency_ms": 0.15, # Lower is better
    "ram_footprint_mb": 0.10,     # Lower is better
}

HIGHER_IS_BETTER: Dict[str, bool] = {
    "clean_accuracy": True,
    "noisy_accuracy": True,
    "d_kl_eviction": False,
    "inference_latency_ms": False,
    "ram_footprint_mb": False,
}


class ResultsTracker:
    """
    Manages AutoML trials, computes composite ranking scores, and persists best artifacts.
    """

    def __init__(self, output_dir: str = "outputs/cifar100/experiments") -> None:
        """
        Initializes the ResultsTracker.

        Args:
            output_dir: Base directory where experiments and best solution are stored.
        """
        self.output_dir: str = os.path.abspath(output_dir)
        self.best_solution_dir: str = os.path.join(self.output_dir, "best_solution")
        self.trials: List[Dict[str, Any]] = []

        os.makedirs(self.output_dir, exist_ok=True)

    def add_trial(
        self,
        trial_id: str,
        config: Dict[str, Any],
        metrics: Dict[str, float],
        artifacts_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Registers a completed trial and updates composite Pareto scores across all trials.

        Args:
            trial_id: Unique trial identifier string.
            config: Hyperparameter dictionary for this trial.
            metrics: Evaluated dictionary containing the 5 canonical metrics.
            artifacts_path: Optional path to the directory containing trial artifacts.

        Returns:
            The registered trial dictionary.
        """
        trial_entry: Dict[str, Any] = {
            "trial_id": trial_id,
            "config": dict(config),
            "metrics": dict(metrics),
            "artifacts_path": artifacts_path,
            "score": 1.0,
            "composite_score": 1.0,
            "normalized_metrics": {},
        }
        self.trials.append(trial_entry)

        # Recompute composite Pareto scores across all trials
        self._recompute_scores()

        # Save summary
        self.save_summary()

        return trial_entry

    def _recompute_scores(self) -> None:
        """
        Computes Min-Max scaled scores across all registered trials according
        to the formal multi-objective equation:

            Score = 0.30 * s_clean + 0.25 * s_noisy + 0.20 * s_dkl + 0.15 * s_lat + 0.10 * s_ram
        """
        n_trials = len(self.trials)
        if n_trials == 0:
            return

        if n_trials == 1:
            self.trials[0]["score"] = 1.0
            self.trials[0]["composite_score"] = 1.0
            self.trials[0]["normalized_metrics"] = {k: 1.0 for k in METRIC_WEIGHTS.keys()}
            return

        # Extract values for each metric
        metric_values: Dict[str, List[float]] = {}
        for m_name in METRIC_WEIGHTS.keys():
            metric_values[m_name] = [
                float(t["metrics"].get(m_name, 0.0)) for t in self.trials
            ]

        # Calculate min and max for each metric
        metric_ranges: Dict[str, Tuple[float, float]] = {}
        for m_name, vals in metric_values.items():
            metric_ranges[m_name] = (float(min(vals)), float(max(vals)))

        # Normalize and compute composite score for each trial
        for t in self.trials:
            norm_scores: Dict[str, float] = {}
            composite_score = 0.0

            for m_name, weight in METRIC_WEIGHTS.items():
                val = float(t["metrics"].get(m_name, 0.0))
                min_v, max_v = metric_ranges[m_name]
                rng = max_v - min_v

                if rng == 0.0:
                    scaled = 1.0
                else:
                    scaled = (val - min_v) / rng

                # Invert if lower is better
                if not HIGHER_IS_BETTER[m_name]:
                    s_m = 1.0 - scaled if rng != 0.0 else 1.0
                else:
                    s_m = scaled

                norm_scores[m_name] = round(float(s_m), 4)
                composite_score += float(weight * s_m)

            final_score = round(float(composite_score), 4)
            t["score"] = final_score
            t["composite_score"] = final_score
            t["normalized_metrics"] = norm_scores

    def get_best_trial(self) -> Dict[str, Any]:
        """
        Returns the trial with the highest Pareto composite score.
        Ties are broken by clean accuracy, then noisy accuracy.
        """
        if not self.trials:
            raise ValueError("No trials have been recorded in ResultsTracker.")

        def sort_key(t: Dict[str, Any]) -> Tuple[float, float, float]:
            return (
                float(t["score"]),
                float(t["metrics"].get("clean_accuracy", 0.0)),
                float(t["metrics"].get("noisy_accuracy", 0.0)),
            )

        best = max(self.trials, key=sort_key)
        return best

    def save_summary(self) -> None:
        """
        Persists summary files (JSON and CSV) of all executed trials.
        """
        # 1. Summary JSON
        json_path = os.path.join(self.output_dir, "experiments_summary.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.trials, f, indent=2)

        # 2. Summary CSV
        csv_path = os.path.join(self.output_dir, "experiments_summary.csv")
        headers = [
            "trial_id",
            "score",
            "clean_accuracy",
            "noisy_accuracy",
            "d_kl_eviction",
            "inference_latency_ms",
            "ram_footprint_mb",
            "latent_dim",
            "memory_capacity",
            "knn_k",
            "mahalanobis_threshold",
            "entropy_threshold",
            "rl_lr",
            "rl_gamma",
        ]

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            for t in self.trials:
                cfg = t.get("config", {})
                met = t.get("metrics", {})
                writer.writerow([
                    t.get("trial_id"),
                    t.get("score"),
                    met.get("clean_accuracy"),
                    met.get("noisy_accuracy"),
                    met.get("d_kl_eviction"),
                    met.get("inference_latency_ms"),
                    met.get("ram_footprint_mb"),
                    cfg.get("latent_dim"),
                    cfg.get("memory_capacity"),
                    cfg.get("knn_k"),
                    cfg.get("mahalanobis_threshold"),
                    cfg.get("entropy_threshold"),
                    cfg.get("rl_lr"),
                    cfg.get("rl_gamma"),
                ])

    def extract_best(self, target_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Extracts and copies all checkpoints and artifacts for the best-performing
        trial into the standardized best_solution/ destination directory.

        Generates best_config.json and best_metrics.json strictly conforming
        to the Case Study schema.

        Args:
            target_dir: Destination folder. Defaults to self.best_solution_dir.

        Returns:
            Dictionary representing the extracted best trial.
        """
        best = self.get_best_trial()
        dest_dir = os.path.abspath(target_dir or self.best_solution_dir)
        os.makedirs(dest_dir, exist_ok=True)

        # 1. Copy model and agent checkpoints if artifacts_path exists
        art_path = best.get("artifacts_path")
        if art_path and os.path.exists(art_path):
            for item in os.listdir(art_path):
                src_item = os.path.join(art_path, item)
                dst_item = os.path.join(dest_dir, item)

                # Skip old JSON files that will be generated freshly
                if item in ("best_config.json", "best_metrics.json", "config.json", "metrics.json"):
                    continue

                if os.path.isdir(src_item):
                    if os.path.exists(dst_item):
                        shutil.rmtree(dst_item)
                    shutil.copytree(src_item, dst_item)
                    # Also copy checkpoint files into root of best_solution/
                    if item == "checkpoints":
                        for sub_item in os.listdir(src_item):
                            sub_src = os.path.join(src_item, sub_item)
                            sub_dst = os.path.join(dest_dir, sub_item)
                            if os.path.isfile(sub_src):
                                shutil.copy2(sub_src, sub_dst)
                else:
                    shutil.copy2(src_item, dst_item)

        # 2. Generate best_config.json strictly compliant with Step 4.1 Schema
        cfg = best["config"]
        best_config = {
            "dataset": str(cfg.get("dataset", "cifar100")),
            "trial_id": str(best["trial_id"]),
            "latent_dim": int(cfg.get("latent_dim", 128)),
            "mahalanobis_threshold": float(cfg.get("mahalanobis_threshold", 15.0)),
            "entropy_threshold": float(cfg.get("entropy_threshold", 2.0)),
            "curriculum_alpha_decay": float(cfg.get("curriculum_alpha_decay", 0.995)),
            "memory_capacity": int(cfg.get("memory_capacity", 5000)),
            "rl_lr": float(cfg.get("rl_lr", 1e-3)),
            "rl_gamma": float(cfg.get("rl_gamma", 0.99)),
            "knn_k": int(cfg.get("knn_k", 10)),
            "min_alpha": float(cfg.get("min_alpha", 0.01)),
        }

        best_config_path = os.path.join(dest_dir, "best_config.json")
        with open(best_config_path, "w", encoding="utf-8") as f:
            json.dump(best_config, f, indent=2)

        # 3. Generate best_metrics.json strictly compliant with Step 4.1 Schema
        best_metrics = {
            "trial_id": str(best["trial_id"]),
            "composite_score": round(float(best["score"]), 4),
            "metrics": {
                "clean_accuracy": round(float(best["metrics"]["clean_accuracy"]), 4),
                "noisy_accuracy": round(float(best["metrics"]["noisy_accuracy"]), 4),
                "d_kl_eviction": round(float(best["metrics"]["d_kl_eviction"]), 4),
                "inference_latency_ms": round(float(best["metrics"]["inference_latency_ms"]), 2),
                "ram_footprint_mb": round(float(best["metrics"]["ram_footprint_mb"]), 2),
            },
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "seed": int(cfg.get("seed", 42)),
        }

        best_metrics_path = os.path.join(dest_dir, "best_metrics.json")
        with open(best_metrics_path, "w", encoding="utf-8") as f:
            json.dump(best_metrics, f, indent=2)

        # 4. Perform automated schema validation
        self.validate_solution_schema(dest_dir)

        logger.info(
            f"Best trial {best['trial_id']} successfully extracted to: {dest_dir} "
            f"(Score: {best['score']})"
        )

        return best

    @staticmethod
    def validate_solution_schema(solution_dir: str) -> Dict[str, Any]:
        """
        Validates that best_config.json and best_metrics.json in solution_dir
        strictly adhere to the Case Study schema specification (Step 4.1).

        Raises:
            FileNotFoundError: If required JSON files are absent.
            AssertionError: If schemas or types are invalid.

        Returns:
            Dictionary with parsed and validated config and metrics.
        """
        config_path = os.path.join(solution_dir, "best_config.json")
        metrics_path = os.path.join(solution_dir, "best_metrics.json")

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Missing best_config.json in {solution_dir}")
        if not os.path.exists(metrics_path):
            raise FileNotFoundError(f"Missing best_metrics.json in {solution_dir}")

        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        with open(metrics_path, "r", encoding="utf-8") as f:
            met = json.load(f)

        # 1. Validate best_config.json schema
        required_config_keys = {
            "dataset",
            "trial_id",
            "latent_dim",
            "mahalanobis_threshold",
            "entropy_threshold",
            "curriculum_alpha_decay",
            "memory_capacity",
            "rl_lr",
            "rl_gamma",
            "knn_k",
            "min_alpha",
        }
        assert set(cfg.keys()) == required_config_keys, (
            f"best_config.json keys mismatch! Expected: {sorted(required_config_keys)}, "
            f"Got: {sorted(cfg.keys())}"
        )
        assert isinstance(cfg["dataset"], str) and cfg["dataset"] in ("cifar100", "cifar10", "mnist"), "Invalid dataset"
        assert isinstance(cfg["trial_id"], str) and len(cfg["trial_id"]) > 0, "Invalid trial_id"
        assert isinstance(cfg["latent_dim"], int) and cfg["latent_dim"] > 0, "Invalid latent_dim"
        assert isinstance(cfg["mahalanobis_threshold"], (int, float)) and cfg["mahalanobis_threshold"] > 0, "Invalid mahalanobis_threshold"
        assert isinstance(cfg["entropy_threshold"], (int, float)) and cfg["entropy_threshold"] > 0, "Invalid entropy_threshold"
        assert isinstance(cfg["curriculum_alpha_decay"], (int, float)) and 0.0 < cfg["curriculum_alpha_decay"] <= 1.0, "Invalid curriculum_alpha_decay"
        assert isinstance(cfg["memory_capacity"], int) and cfg["memory_capacity"] > 0, "Invalid memory_capacity"
        assert isinstance(cfg["rl_lr"], (int, float)) and cfg["rl_lr"] > 0, "Invalid rl_lr"
        assert isinstance(cfg["rl_gamma"], (int, float)) and 0.0 < cfg["rl_gamma"] <= 1.0, "Invalid rl_gamma"
        assert isinstance(cfg["knn_k"], int) and cfg["knn_k"] > 0, "Invalid knn_k"
        assert isinstance(cfg["min_alpha"], (int, float)) and cfg["min_alpha"] >= 0.0, "Invalid min_alpha"

        # 2. Validate best_metrics.json schema
        required_metrics_keys = {
            "trial_id",
            "composite_score",
            "metrics",
            "timestamp",
            "seed",
        }
        assert set(met.keys()) == required_metrics_keys, (
            f"best_metrics.json keys mismatch! Expected: {sorted(required_metrics_keys)}, "
            f"Got: {sorted(met.keys())}"
        )
        assert met["trial_id"] == cfg["trial_id"], "trial_id mismatch between config and metrics"
        assert isinstance(met["composite_score"], (int, float)) and 0.0 <= met["composite_score"] <= 1.0, "Invalid composite_score"
        assert isinstance(met["seed"], int), "Invalid seed"

        required_sub_metrics = {
            "clean_accuracy",
            "noisy_accuracy",
            "d_kl_eviction",
            "inference_latency_ms",
            "ram_footprint_mb",
        }
        sub_metrics = met["metrics"]
        assert set(sub_metrics.keys()) == required_sub_metrics, (
            f"metrics object keys mismatch! Expected: {sorted(required_sub_metrics)}, "
            f"Got: {sorted(sub_metrics.keys())}"
        )
        assert 0.0 <= sub_metrics["clean_accuracy"] <= 1.0, "clean_accuracy must be in [0.0, 1.0]"
        assert 0.0 <= sub_metrics["noisy_accuracy"] <= 1.0, "noisy_accuracy must be in [0.0, 1.0]"
        assert sub_metrics["d_kl_eviction"] >= 0.0, "d_kl_eviction must be >= 0.0"
        assert sub_metrics["inference_latency_ms"] >= 0.0, "inference_latency_ms must be >= 0.0"
        assert sub_metrics["ram_footprint_mb"] >= 0.0, "ram_footprint_mb must be >= 0.0"

        # Validate ISO-8601 timestamp parseability
        datetime.datetime.fromisoformat(met["timestamp"].replace("Z", "+00:00"))

        return {"valid": True, "config": cfg, "metrics": met}

