"""
Unit and Integration Test Suite for CIFAR-100 Promotion and Backward Compatibility:
1. Validation of promoted ResNet-18 V2 checkpoint loading and forward inference contract (128D latent, 100 classes).
2. Backward compatibility verification: loading legacy ResNet-14 V1 checkpoint.
3. Verification of EAAI benchmark results and scientific criteria in eaai_metrics.json.
"""

import os
import sys
import json
import numpy as np
import tensorflow as tf

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.cifar100.model import load_cifar100_backbone, RawModelCIFAR100V2, RawModelCIFAR100


def test_promoted_v2_backbone_forward_and_contract():
    """Validates that the promoted checkpoint in outputs/cifar100/checkpoints loads as V2 and respects 128D contract."""
    checkpoint_dir = os.path.join(PROJECT_ROOT, "outputs", "cifar100", "checkpoints")
    assert os.path.exists(checkpoint_dir), f"Directory {checkpoint_dir} not found"

    model, standardize = load_cifar100_backbone(checkpoint_dir)
    assert isinstance(model, RawModelCIFAR100V2), f"Expected RawModelCIFAR100V2, got {type(model)}"
    assert standardize is True, "Expected standardize=True for promoted V2 model"

    meta_path = os.path.join(checkpoint_dir, "model_meta.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta.get("best_test_acc") == 74.27
    assert meta.get("model_version") == "v2"

    # Test forward pass
    dummy = tf.convert_to_tensor(np.random.rand(2, 32, 32, 3).astype(np.float32))
    out = model(dummy, training=False)
    assert "latent_features" in out, "Missing 'latent_features' in model output"
    assert "probabilities" in out, "Missing 'probabilities' in model output"
    assert out["latent_features"].shape == (2, 128), f"Expected (2, 128), got {out['latent_features'].shape}"
    assert out["probabilities"].shape == (2, 100), f"Expected (2, 100), got {out['probabilities'].shape}"

    # Probabilities sum to 1
    prob_sums = tf.reduce_sum(out["probabilities"], axis=-1).numpy()
    assert np.allclose(prob_sums, 1.0, atol=1e-5), f"Probabilities do not sum to 1: {prob_sums}"


def test_legacy_v1_backbone_backward_compatibility():
    """Validates backward compatibility: legacy V1 checkpoint can still be loaded seamlessly."""
    legacy_dir = os.path.join(PROJECT_ROOT, "outputs", "cifar100", "checkpoints_v1_legacy")
    if not os.path.exists(legacy_dir):
        print("  Legacy checkpoint directory not present, skipping legacy test.")
        return

    model, standardize = load_cifar100_backbone(legacy_dir)
    assert isinstance(model, RawModelCIFAR100), f"Expected RawModelCIFAR100, got {type(model)}"
    assert standardize is False, "Expected standardize=False for legacy V1 model"

    dummy = tf.convert_to_tensor(np.random.rand(2, 32, 32, 3).astype(np.float32))
    out = model(dummy, training=False)
    assert out["latent_features"].shape == (2, 128)
    assert out["probabilities"].shape == (2, 100)


def test_eaai_metrics_scientific_criteria():
    """Validates that outputs/cifar100/eaai_metrics.json satisfies all core scientific theses."""
    metrics_path = os.path.join(PROJECT_ROOT, "outputs", "cifar100", "eaai_metrics.json")
    assert os.path.exists(metrics_path), f"Metrics file {metrics_path} not found"

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    for baseline in ["B0", "B1", "B2", "B3", "B4"]:
        assert baseline in metrics, f"Missing {baseline} in eaai_metrics.json"

    b0 = metrics["B0"]
    b2 = metrics["B2"]
    b3 = metrics["B3"]
    b4 = metrics["B4"]

    # 1. Clean accuracy dividend: B4 > B2
    assert b4["accuracy_noise_0.0"] > b2["accuracy_noise_0.0"], "B4 clean accuracy must exceed FIFO B2"
    assert np.isclose(b4["accuracy_noise_0.0"], 71.8)
    assert np.isclose(b2["accuracy_noise_0.0"], 67.8)

    # 2. Noise onset shield: B4 > B0, B4 > B2, B4 > B3
    assert b4["accuracy_noise_0.2"] > b0["accuracy_noise_0.2"], "B4 must exceed B0 under noise 0.2"
    assert b4["accuracy_noise_0.2"] > b2["accuracy_noise_0.2"], "B4 must exceed B2 under noise 0.2"
    assert b4["accuracy_noise_0.2"] > b3["accuracy_noise_0.2"], "B4 must exceed B3 under noise 0.2"
    assert np.isclose(b4["accuracy_noise_0.2"], 3.4)

    # 3. Overall stream top performer: B4 is #1
    assert b4["overall_accuracy"] >= b0["overall_accuracy"], "B4 overall accuracy must be >= B0"
    assert b4["overall_accuracy"] > b2["overall_accuracy"], "B4 overall accuracy must exceed B2"
    assert b4["overall_accuracy"] > b3["overall_accuracy"], "B4 overall accuracy must exceed B3"
    assert np.isclose(b4["overall_accuracy"], 15.68)

    # 4. Elimination of class starvation: D_KL(B4) near zero, D_KL(B2) > 2.0
    assert b4["eviction_kl_divergence"] < 1e-4, f"B4 D_KL must be near zero: {b4['eviction_kl_divergence']}"
    assert b2["eviction_kl_divergence"] > 2.0, f"B2 D_KL must be > 2.0: {b2['eviction_kl_divergence']}"

    # 5. Hardware edge bounds (LMOS)
    assert b4["ram_peak_mb"] <= 15.0, f"Peak RAM must be <= 15 MB: {b4['ram_peak_mb']}"
    assert b4["latency_ms"] <= 33.3, f"Query latency must support >= 30 FPS: {b4['latency_ms']}"


if __name__ == "__main__":
    print("Running test_promoted_v2_backbone_forward_and_contract...")
    test_promoted_v2_backbone_forward_and_contract()
    print("  -> Passed.")

    print("Running test_legacy_v1_backbone_backward_compatibility...")
    test_legacy_v1_backbone_backward_compatibility()
    print("  -> Passed.")

    print("Running test_eaai_metrics_scientific_criteria...")
    test_eaai_metrics_scientific_criteria()
    print("  -> Passed.")

    print("\nAll promotion and backward compatibility tests PASSED successfully!")
