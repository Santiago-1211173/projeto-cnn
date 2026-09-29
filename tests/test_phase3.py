"""
Unit and Integration Test Suite for Phase 3:
Simulacao de Caos e Stress Test (TrainRLOnlineSimulation, Mahalanobis++, inject_noise).

Validates all Phase 3 requirements and acceptance criteria from docs/Plano_de_Acao_EAAI.md:
1. inject_noise: perturbation properties, boundedness in [0.0, 1.0], float32 dtype.
2. MahalanobisPlusPlus: L2 normalization, Ledoit-Wolf Shrinkage covariance, OOD discrimination.
3. run_simulation:
   - Invariant: memory size never exceeds capacity (agent.size <= capacity).
   - Rewards returned and non-empty.
   - Epsilon annealing: epsilon final < epsilon initial.
   - Structured CSV logging: file exists and contains [step, size, reward_mean, alpha, loss, epsilon].
   - Double DQN training and PER stability without OOM.
"""

import os
import sys
import tempfile
import numpy as np

# Ensure project root is in path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from training.train_rl_online_simulation import (
    inject_noise,
    MahalanobisPlusPlus,
    TrainRLOnlineSimulation,
    run_simulation,
)
from src.config import MAHALANOBIS_THRESHOLD


def test_inject_noise():
    print("Testing inject_noise...")
    clean_batch = np.full((10, 28, 28, 1), 0.5, dtype=np.float32)

    # 1. Zero noise returns identical array
    zero_noisy = inject_noise(clean_batch, 0.0)
    assert np.allclose(clean_batch, zero_noisy)
    assert zero_noisy.dtype == np.float32

    # 2. Positive noise alters values and remains strictly bounded [0.0, 1.0]
    corrupted = inject_noise(clean_batch, 0.8)
    assert not np.allclose(clean_batch, corrupted)
    assert corrupted.shape == clean_batch.shape
    assert corrupted.dtype == np.float32
    assert np.all(corrupted >= 0.0)
    assert np.all(corrupted <= 1.0)
    print("  -> inject_noise verified successfully.")


def test_mahalanobis_plus_plus():
    print("Testing MahalanobisPlusPlus (Ledoit-Wolf & L2 Normalization)...")
    detector = MahalanobisPlusPlus(n_classes=3, latent_dim=16, threshold=5.0)

    # Generate synthetic clustered latent features
    np.random.seed(42)
    n_per_class = 40
    dim = 16
    features = []
    labels = []

    for c in range(3):
        center = np.zeros(dim)
        center[c * 4 : (c + 1) * 4] = 3.0
        cluster = center + np.random.randn(n_per_class, dim) * 0.2
        features.append(cluster)
        labels.extend([c] * n_per_class)

    features = np.vstack(features).astype(np.float32)
    labels = np.array(labels, dtype=np.int32)

    detector.fit(features, labels)
    assert detector.is_fitted
    assert len(detector.profiles) == 3

    # Check that in-distribution samples have smaller distance than far outliers
    in_dist_sample = features[0]
    d_in = detector.compute_distance(in_dist_sample)

    far_outlier = np.random.randn(dim).astype(np.float32) * 50.0
    d_out = detector.compute_distance(far_outlier)

    assert d_in < d_out
    assert d_in >= 0.0

    # Batch distance test
    dists, closest = detector.compute_distances_batch(features[:5])
    assert len(dists) == 5
    assert len(closest) == 5

    # Persistence save/load test
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        detector.save(tmp_path)
        restored = MahalanobisPlusPlus()
        restored.load(tmp_path)
        assert restored.is_fitted
        d_restored = restored.compute_distance(in_dist_sample)
        assert np.isclose(d_in, d_restored, atol=1e-5)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    print("  -> MahalanobisPlusPlus verified successfully.")


def test_simulation_micro_run():
    print("Testing TrainRLOnlineSimulation Micro Run (Active Eviction & Capacity Invariant)...")
    tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
    csv_path = tmp.name
    tmp.close()

    try:
        capacity = 50
        steps = 150

        sim = TrainRLOnlineSimulation(
            capacity=capacity,
            k_neighbors=5,
            latent_dim=128,
            buffer_size=20,
            replay_capacity=200,
        )

        results = sim.execute_simulation(
            n_episodes=1,
            n_steps=steps,
            noise_injection_rate=0.2,
            batch_size=16,
            log_interval=25,
            output_csv_path=csv_path,
            save_agent_path=None,
        )

        # 1. Output structure check
        assert "rewards" in results
        assert "losses" in results
        assert "epsilons" in results
        assert "sizes" in results
        assert len(results["rewards"]) == steps

        # 2. Capacity Invariant check: size never exceeds capacity
        assert all(s <= capacity for s in results["sizes"])
        assert max(results["sizes"]) == capacity  # Reached full capacity and sustained it

        # 3. Epsilon annealing check: final epsilon is lower than start
        assert results["epsilons"][-1] < results["epsilons"][0]

        # 4. Active eviction actions triggered
        assert len(results["actions"]) > 0

        # 5. CSV Logging verification
        assert os.path.exists(csv_path)
        with open(csv_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        assert len(lines) >= 6  # Header + log intervals
        header = lines[0].strip().split(",")
        assert header == ["step", "size", "reward_mean", "alpha", "loss", "epsilon"]

    finally:
        if os.path.exists(csv_path):
            try:
                os.remove(csv_path)
            except OSError:
                pass

    print("  -> TrainRLOnlineSimulation micro run passed successfully.")


def test_run_simulation_public_contract():
    print("Testing run_simulation Public Interface Contract...")
    tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
    csv_path = tmp.name
    tmp.close()

    try:
        # Run small contract test
        results = run_simulation(
            n_episodes=1,
            noise_injection_rate=0.1,
            capacity=100,
            n_steps=200,
            output_csv_path=csv_path,
        )

        assert "rewards" in results
        assert len(results["rewards"]) > 0
        assert max(results["sizes"]) <= 100
        assert results["epsilons"][-1] < results["epsilons"][0]

    finally:
        if os.path.exists(csv_path):
            os.remove(csv_path)

    print("  -> run_simulation public contract passed successfully.")


if __name__ == "__main__":
    print("=" * 70)
    print("RUNNING PHASE 3 VALIDATION TESTS")
    print("=" * 70)
    test_inject_noise()
    test_mahalanobis_plus_plus()
    test_simulation_micro_run()
    test_run_simulation_public_contract()
    print("=" * 70)
    print("ALL PHASE 3 UNIT TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 70)
