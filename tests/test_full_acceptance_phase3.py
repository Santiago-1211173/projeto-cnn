"""
Full Acceptance Test for Phase 3 (50,000 steps, capacity=5000, noise_rate=0.1).

Validates the full scale requirements defined in docs/Plano_de_Acao_EAAI.md:
1. Simulation runs without OOM across 50,000 steps with capacity=5000.
2. Invariant holds: max(size) <= capacity and max(size) == capacity.
3. RL agent reduces epsilon over training: epsilon_final < epsilon_initial.
4. Returns rewards array with len(rewards) > 0.
5. Generates structured CSV log.
"""

import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from training.train_rl_online_simulation import run_simulation
from src.config import OUTPUT_DIR, SIMULATION_CSV_PATH


def main():
    print("=" * 70)
    print("RUNNING PHASE 3 FULL ACCEPTANCE CRITERIA TEST (50,000 STEPS)")
    print("=" * 70)

    start_time = time.time()

    # Public contract call as specified in Plano_de_Acao_EAAI.md
    results = run_simulation(n_episodes=1, noise_injection_rate=0.1, capacity=5000)

    total_time = time.time() - start_time
    print(f"\nExecution finished in {total_time:.2f} seconds ({50000 / total_time:.1f} steps/s).")

    # 1. Output structure check
    assert "rewards" in results, "Key 'rewards' missing from results"
    assert len(results["rewards"]) > 0, "Rewards array is empty"
    print(f"[OK] assert 'rewards' in results and len(rewards) = {len(results['rewards']):,} > 0")

    # 2. Capacity invariant check: memory never exceeds capacity
    max_size = max(results["sizes"])
    assert max_size <= 5000, f"Memory exceeded capacity: {max_size} > 5000"
    assert max_size == 5000, f"Memory failed to reach full capacity: {max_size} != 5000"
    print(f"[OK] Invariant confirmed: max(size) == {max_size} <= 5000")

    # 3. Epsilon reduction check
    eps_start = results["epsilons"][0]
    eps_final = results["epsilons"][-1]
    assert eps_final < eps_start, f"Epsilon was not reduced: {eps_final} >= {eps_start}"
    print(f"[OK] Epsilon reduction confirmed: {eps_start:.4f} -> {eps_final:.4f}")

    # 4. CSV log verification
    assert os.path.exists(SIMULATION_CSV_PATH), f"CSV log not found at {SIMULATION_CSV_PATH}"
    with open(SIMULATION_CSV_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()
    assert len(lines) > 50, f"CSV log has insufficient lines: {len(lines)}"
    print(f"[OK] Structured CSV log successfully written to {SIMULATION_CSV_PATH} ({len(lines)} lines)")

    print("=" * 70)
    print("PHASE 3 FULL ACCEPTANCE TEST PASSED WITH ZERO ERRORS!")
    print("=" * 70)


if __name__ == "__main__":
    main()
