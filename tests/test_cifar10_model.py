"""
Unit and Acceptance Tests for CIFAR-10 CNN Architecture (Phase 3).
"""

import os
import sys
import numpy as np
import tensorflow as tf

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.custom_cnn_cifar10 import RawModelCIFAR10
from src.models.custom_cnn import RawModel


def test_model_forward_pass():
    print("Testing RawModelCIFAR10 forward pass...")
    model = RawModelCIFAR10()
    dummy = tf.convert_to_tensor(np.random.rand(2, 32, 32, 3).astype(np.float32))
    out = model(dummy)

    assert "latent_features" in out, "Missing key 'latent_features' in model output"
    assert "probabilities" in out, "Missing key 'probabilities' in model output"

    latent_shape = out["latent_features"].shape
    prob_shape = out["probabilities"].shape
    print(f"  latent_features shape: {latent_shape}")
    print(f"  probabilities shape: {prob_shape}")

    assert latent_shape == (2, 128), f"Expected latent shape (2, 128), got {latent_shape}"
    assert prob_shape == (2, 10), f"Expected probabilities shape (2, 10), got {prob_shape}"

    prob_sums = tf.reduce_sum(out["probabilities"], axis=-1).numpy()
    print(f"  probabilities sum: {prob_sums[0]:.6f}")
    assert np.allclose(prob_sums, 1.0, atol=1e-5), f"Probabilities do not sum to 1.0: {prob_sums}"
    print("  -> Forward pass tests passed.")


def test_model_parameter_count():
    print("Testing RawModelCIFAR10 parameter count...")
    model = RawModelCIFAR10()
    # Trigger graph/variable initialization if needed (tf.Module variables are created in __init__)
    total_params = sum(tf.size(v).numpy() for v in model.trainable_variables)
    print(f"  Total params: {total_params}")
    assert total_params == 271786, f"Expected 271786 params, got {total_params}"
    print("  -> Parameter count test passed.")


def test_model_contract_compatibility():
    print("Comparing CIFAR-10 and MNIST models contract compatibility...")
    mnist_model = RawModel()
    cifar_model = RawModelCIFAR10()

    dummy_mnist = tf.convert_to_tensor(np.random.rand(4, 28, 28, 1).astype(np.float32))
    dummy_cifar = tf.convert_to_tensor(np.random.rand(4, 32, 32, 3).astype(np.float32))

    out_mnist = mnist_model(dummy_mnist)
    out_cifar = cifar_model(dummy_cifar)

    assert out_mnist["latent_features"].shape == out_cifar["latent_features"].shape == (4, 128)
    assert out_mnist["probabilities"].shape == out_cifar["probabilities"].shape == (4, 10)
    print("  -> Contract compatibility tests passed.")


if __name__ == "__main__":
    print("=" * 70)
    print("RUNNING CIFAR-10 CNN MODEL TESTS (PHASE 3)")
    print("=" * 70)
    test_model_forward_pass()
    test_model_parameter_count()
    test_model_contract_compatibility()
    print("=" * 70)
    print("ALL CIFAR-10 MODEL TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 70)
