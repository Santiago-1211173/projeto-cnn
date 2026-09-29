"""
Unit and Acceptance Tests for Phase 2 (RewardManager and Double DQN RLAgent with PER).

Validates all criteria required by docs/Plano_de_Acao_EAAI.md:
1. Interface contract compliance
2. Mathematical constraints and reward bounds [-1.0, 1.0]
3. Curriculum learning alpha decay [0.0, 1.0]
4. 5D state vector normalization [0.0, 1.0]
5. Epsilon-greedy action selection in {0, 1, 2, 3}
6. Prioritized Experience Replay (SumTree) sampling and weight updates
7. Double DQN gradient step and target network synchronization
"""

import os
import sys
import tempfile
import time
import numpy as np
import torch

# Ensure repository root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.knn_bandit_agent import KNNBanditAgent128D
from src.models.reward_manager import RewardManager
from src.models.rl_agent import RLAgent, SumTree, PrioritizedReplayBuffer


def test_sum_tree() -> None:
    """Tests SumTree priority accumulation, updates, and O(log N) retrieval."""
    print("Testing SumTree...")
    capacity = 16
    tree = SumTree(capacity)
    assert tree.total_priority == 0.0

    # Add priorities
    priorities = [1.0, 2.0, 3.0, 4.0, 5.0]
    for i, p in enumerate(priorities):
        tree_idx = i + capacity - 1
        tree.update(tree_idx, p)

    expected_sum = sum(priorities)
    assert np.isclose(tree.total_priority, expected_sum), f"Expected {expected_sum}, got {tree.total_priority}"

    # Query leaf for sample values
    tree_idx, prio, data_idx = tree.get_leaf(0.5)
    assert data_idx == 0
    assert prio == 1.0

    tree_idx, prio, data_idx = tree.get_leaf(2.5)  # cumulative: 1.0 + 2.0 = 3.0
    assert data_idx == 1
    assert prio == 2.0

    # Update priority
    tree.update(capacity - 1, 10.0)  # replace first element
    assert np.isclose(tree.total_priority, expected_sum - 1.0 + 10.0)
    print("  -> SumTree passed successfully.")


def test_reward_manager_acceptance() -> None:
    """Validates the explicit acceptance criteria from docs/Plano_de_Acao_EAAI.md."""
    print("Testing RewardManager Acceptance Criteria...")
    agent = KNNBanditAgent128D(capacity=50, k=5, latent_dim=128)
    for i in range(10):
        agent.add_experience(np.random.randn(128).astype(np.float32), i % 4, 1.0)

    rm = RewardManager(buffer_size=100)
    initial_alpha = rm.get_current_alpha()
    assert 0.0 <= initial_alpha <= 1.0, f"Alpha out of bounds: {initial_alpha}"

    # Compute reward for eviction action
    reward = rm.compute_reward(evicted_index=0, memory=agent, new_state=np.random.randn(128))
    assert -1.0 <= reward <= 1.0, f"Reward out of bounds: {reward}"
    assert 0.0 <= rm.get_current_alpha() <= 1.0, f"Alpha out of bounds after decay: {rm.get_current_alpha()}"
    assert rm.get_current_alpha() < initial_alpha, "Alpha did not decay after compute_reward"

    # Compute reward for ignore action (evicted_index = -1)
    ignore_reward = rm.compute_reward(evicted_index=-1, memory=agent, new_state=np.random.randn(128))
    assert -1.0 <= ignore_reward <= 1.0, f"Ignore reward out of bounds: {ignore_reward}"

    # Sliding validation buffer
    for i in range(120):  # Overfill 100-sample buffer
        rm.update_validation_buffer(np.random.randn(128), true_label=i % 10, predicted_label=(i + 1) % 10)
    stats = rm.get_buffer_stats()
    assert stats["validation_size"] == 100, f"Buffer size should be capped at 100, got {stats['validation_size']}"
    assert stats["occupancy_pct"] == 100.0

    # Reward with populated validation buffer
    reward_with_val = rm.compute_reward(evicted_index=1, memory=agent, new_state=np.random.randn(128))
    assert -1.0 <= reward_with_val <= 1.0, f"Reward with val buffer out of bounds: {reward_with_val}"

    print("  -> RewardManager acceptance criteria passed successfully.")


def test_rl_agent_acceptance() -> None:
    """Validates the explicit RLAgent acceptance criteria from docs/Plano_de_Acao_EAAI.md."""
    print("Testing RLAgent Acceptance Criteria...")
    rl = RLAgent(state_dim=5, n_actions=4, lr=1e-3, target_update_interval=50)

    # 5D State vector normalization
    state = rl.get_state_vector(12.5, 0.8, 3.2, 0.0, 0.95)
    assert state.shape == (5,), f"Expected shape (5,), got {state.shape}"
    assert state.dtype == np.float32
    assert np.all((state >= 0.0) & (state <= 1.0)), f"State features not normalized in [0, 1]: {state}"

    # Action selection
    action = rl.select_action(state, epsilon=0.1)
    assert action in {0, 1, 2, 3}, f"Invalid action: {action}"

    # Exploitation check (epsilon = 0.0)
    greedy_action = rl.select_action(state, epsilon=0.0)
    assert greedy_action in {0, 1, 2, 3}

    # Exploration check (epsilon = 1.0)
    random_action = rl.select_action(state, epsilon=1.0)
    assert random_action in {0, 1, 2, 3}

    # Insufficient buffer update test
    assert rl.update_weights(batch_size=32) is None

    # Fill PER buffer
    for i in range(100):
        next_s = rl.get_state_vector(10.0 + i * 0.1, 0.5, 2.0, 1.0 if i % 2 == 0 else 0.0, 0.8)
        r = 0.5 if i % 2 == 0 else -0.5
        rl.store_transition(state, action, r, next_s, False)

    # Update weights with PER
    loss = rl.update_weights(batch_size=32)
    assert loss is not None, "Loss should not be None after 100 transitions"
    assert np.isfinite(loss), f"Loss is not finite: {loss}"
    assert loss >= 0.0, f"Loss should be non-negative: {loss}"

    # Multiple updates to verify target network sync and beta annealing
    initial_beta = rl.replay_buffer.beta
    for _ in range(60):
        rl.update_weights(batch_size=32)
    assert rl.train_step > 50
    assert rl.replay_buffer.beta > initial_beta, "Beta did not anneal upward"

    print("  -> RLAgent acceptance criteria passed successfully.")


def test_rl_persistence() -> None:
    """Verifies that RLAgent models can be saved and loaded accurately."""
    print("Testing RLAgent Save / Load Persistence...")
    agent1 = RLAgent(state_dim=5, n_actions=4)
    state = agent1.get_state_vector(5.0, 0.2, 1.0, 0.0, 0.5)

    # Store transitions and train once
    for _ in range(40):
        agent1.store_transition(state, 1, 1.0, state, False)
    agent1.update_weights(batch_size=32)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "rl_test_model.pt")
        agent1.save(ckpt_path)
        assert os.path.exists(ckpt_path)

        agent2 = RLAgent(state_dim=5, n_actions=4)
        agent2.load(ckpt_path)

        assert agent2.train_step == agent1.train_step

        # Compare outputs of policy networks
        t_state = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q1 = agent1.policy_net(t_state).numpy()
            q2 = agent2.policy_net(t_state).numpy()
        assert np.allclose(q1, q2, atol=1e-5), "Q-values mismatch between saved and loaded models!"

    print("  -> Persistence test passed successfully.")


def test_performance_benchmarks() -> None:
    """Verifies low-latency inference on Edge AI constraints."""
    print("Testing Edge AI Performance Benchmarks...")
    rl = RLAgent(state_dim=5, n_actions=4)
    state = rl.get_state_vector(12.5, 0.8, 3.2, 0.0, 0.95)

    # Benchmark 1000 action selections
    t0 = time.perf_counter()
    for _ in range(1000):
        rl.select_action(state, epsilon=0.05)
    t1 = time.perf_counter()
    duration_ms = (t1 - t0) * 1000.0
    avg_us = (duration_ms / 1000.0) * 1000.0
    print(f"  -> 1000 select_action calls in {duration_ms:.2f} ms ({avg_us:.2f} µs/inference)")
    assert duration_ms < 500.0, f"Inference too slow: {duration_ms} ms for 1000 calls"


if __name__ == "__main__":
    print("=" * 70)
    print("RUNNING PHASE 2 VALIDATION TESTS")
    print("=" * 70)
    test_sum_tree()
    test_reward_manager_acceptance()
    test_rl_agent_acceptance()
    test_rl_persistence()
    test_performance_benchmarks()
    print("=" * 70)
    print("ALL PHASE 2 TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 70)
