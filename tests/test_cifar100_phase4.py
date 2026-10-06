"""
Unit and Integration Test Suite for CIFAR-100 Phase 4:
Extração e Validação do Contrato de Saída (Step 4.1).

Validates all Phase 4 specifications from docs/cifar100_agent_execution_plan.md:
1. Strict schema compliance for outputs/cifar100/experiments/best_solution/best_config.json
2. Strict schema compliance for outputs/cifar100/experiments/best_solution/best_metrics.json
3. Extraction and preservation of model, agent, and memory checkpoints into best_solution/
4. Automated schema validation and rejection of malformed or out-of-boundary configurations
"""

import os
import sys
import json
import tempfile
import pytest
import numpy as np

# Ensure project root is in path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.results_tracker import ResultsTracker


def test_schema_valid_contracts():
    """Validates that valid Step 4.1 contracts strictly pass validation."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tracker = ResultsTracker(output_dir=tmp_dir)

        canonical_config = {
            "dataset": "cifar100",
            "latent_dim": 256,
            "mahalanobis_threshold": 15.0,
            "entropy_threshold": 2.0,
            "curriculum_alpha_decay": 0.995,
            "memory_capacity": 5000,
            "rl_lr": 0.0005,
            "rl_gamma": 0.99,
            "knn_k": 20,
            "min_alpha": 0.05,
            "seed": 42,
        }

        canonical_metrics = {
            "clean_accuracy": 0.7412,
            "noisy_accuracy": 0.6280,
            "d_kl_eviction": 0.1543,
            "inference_latency_ms": 4.82,
            "ram_footprint_mb": 18.45,
        }

        tracker.add_trial("trial_003", canonical_config, canonical_metrics)
        best = tracker.extract_best()

        assert best["trial_id"] == "trial_003"
        dest_dir = tracker.best_solution_dir

        res = ResultsTracker.validate_solution_schema(dest_dir)
        assert res["valid"] is True
        cfg = res["config"]
        met = res["metrics"]

        # Validate exact keys in best_config.json
        expected_cfg_keys = {
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
        assert set(cfg.keys()) == expected_cfg_keys
        assert cfg["dataset"] == "cifar100"
        assert cfg["trial_id"] == "trial_003"
        assert cfg["latent_dim"] == 256

        # Validate exact keys in best_metrics.json
        expected_met_keys = {
            "trial_id",
            "composite_score",
            "metrics",
            "timestamp",
            "seed",
        }
        assert set(met.keys()) == expected_met_keys
        assert met["trial_id"] == "trial_003"
        assert met["seed"] == 42
        assert np.isclose(met["metrics"]["clean_accuracy"], 0.7412)
        assert np.isclose(met["metrics"]["noisy_accuracy"], 0.6280)


def test_schema_rejects_missing_keys():
    """Validates that missing required keys fail validation."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tracker = ResultsTracker(output_dir=tmp_dir)

        canonical_config = {
            "dataset": "cifar100",
            "latent_dim": 256,
            "mahalanobis_threshold": 15.0,
            "entropy_threshold": 2.0,
            "curriculum_alpha_decay": 0.995,
            "memory_capacity": 5000,
            "rl_lr": 0.0005,
            "rl_gamma": 0.99,
            "knn_k": 20,
            "min_alpha": 0.05,
        }
        canonical_metrics = {
            "clean_accuracy": 0.75,
            "noisy_accuracy": 0.60,
            "d_kl_eviction": 0.10,
            "inference_latency_ms": 4.5,
            "ram_footprint_mb": 15.0,
        }

        tracker.add_trial("trial_001", canonical_config, canonical_metrics)
        tracker.extract_best()

        config_path = os.path.join(tracker.best_solution_dir, "best_config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        # Remove a key and verify assertion fails
        del cfg["mahalanobis_threshold"]
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)

        with pytest.raises(AssertionError):
            ResultsTracker.validate_solution_schema(tracker.best_solution_dir)


def test_schema_rejects_invalid_values():
    """Validates that out-of-range metrics or invalid types are rejected."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tracker = ResultsTracker(output_dir=tmp_dir)

        canonical_config = {
            "dataset": "cifar100",
            "latent_dim": 256,
            "mahalanobis_threshold": 15.0,
            "entropy_threshold": 2.0,
            "curriculum_alpha_decay": 0.995,
            "memory_capacity": 5000,
            "rl_lr": 0.0005,
            "rl_gamma": 0.99,
            "knn_k": 20,
            "min_alpha": 0.05,
        }
        canonical_metrics = {
            "clean_accuracy": 0.75,
            "noisy_accuracy": 0.60,
            "d_kl_eviction": 0.10,
            "inference_latency_ms": 4.5,
            "ram_footprint_mb": 15.0,
        }

        tracker.add_trial("trial_001", canonical_config, canonical_metrics)
        tracker.extract_best()

        metrics_path = os.path.join(tracker.best_solution_dir, "best_metrics.json")
        with open(metrics_path, "r", encoding="utf-8") as f:
            met = json.load(f)

        # Corrupt clean_accuracy > 1.0
        met["metrics"]["clean_accuracy"] = 1.5
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(met, f, indent=2)

        with pytest.raises(AssertionError):
            ResultsTracker.validate_solution_schema(tracker.best_solution_dir)


def test_extract_best_with_checkpoint_artifacts():
    """Validates that all checkpoints and profiles are copied to best_solution."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tracker = ResultsTracker(output_dir=tmp_dir)

        trial_dir = os.path.join(tmp_dir, "trial_001")
        ckpt_dir = os.path.join(trial_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)

        # Create dummy artifacts mimicking trial output
        with open(os.path.join(trial_dir, "arbiter_profiles.npz"), "wb") as f:
            f.write(b"DUMMY_ARBITER_NPZ")
        with open(os.path.join(trial_dir, "knn_memory_bank.npz"), "wb") as f:
            f.write(b"DUMMY_KNN_NPZ")
        with open(os.path.join(ckpt_dir, "modelo_dissecado-1.index"), "wb") as f:
            f.write(b"DUMMY_CNN_INDEX")
        with open(os.path.join(ckpt_dir, "rl_agent.pt"), "wb") as f:
            f.write(b"DUMMY_RL_AGENT")

        tracker.add_trial(
            trial_id="trial_001",
            config={
                "dataset": "cifar100",
                "latent_dim": 128,
                "mahalanobis_threshold": 12.0,
                "entropy_threshold": 1.8,
                "curriculum_alpha_decay": 0.99,
                "memory_capacity": 3000,
                "rl_lr": 0.001,
                "rl_gamma": 0.95,
                "knn_k": 10,
                "min_alpha": 0.01,
                "seed": 100,
            },
            metrics={
                "clean_accuracy": 0.80,
                "noisy_accuracy": 0.65,
                "d_kl_eviction": 0.08,
                "inference_latency_ms": 3.9,
                "ram_footprint_mb": 12.5,
            },
            artifacts_path=trial_dir,
        )

        tracker.extract_best()

        sol_dir = tracker.best_solution_dir
        assert os.path.exists(os.path.join(sol_dir, "best_config.json"))
        assert os.path.exists(os.path.join(sol_dir, "best_metrics.json"))
        assert os.path.exists(os.path.join(sol_dir, "arbiter_profiles.npz"))
        assert os.path.exists(os.path.join(sol_dir, "knn_memory_bank.npz"))
        assert os.path.exists(os.path.join(sol_dir, "rl_agent.pt"))
        assert os.path.exists(os.path.join(sol_dir, "modelo_dissecado-1.index"))
        assert os.path.exists(os.path.join(sol_dir, "checkpoints", "rl_agent.pt"))


if __name__ == "__main__":
    print("=" * 70)
    print("RUNNING CIFAR-100 PHASE 4 TEST SUITE")
    print("=" * 70)
    test_schema_valid_contracts()
    print("✓ test_schema_valid_contracts PASSED")
    test_schema_rejects_missing_keys()
    print("✓ test_schema_rejects_missing_keys PASSED")
    test_schema_rejects_invalid_values()
    print("✓ test_schema_rejects_invalid_values PASSED")
    test_extract_best_with_checkpoint_artifacts()
    print("✓ test_extract_best_with_checkpoint_artifacts PASSED")
    print("=" * 70)
    print("ALL CIFAR-100 PHASE 4 TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 70)
