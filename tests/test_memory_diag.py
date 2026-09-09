"""
Episodic Memory Diagnosis Script.
Analyzes stored k-NN experiences to calculate quality metrics like average entropy and confidence distribution.
"""

import sys
import os
import numpy as np

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from src.config import MEMORY_BANK_10D_PATH

def main():
    if not os.path.exists(MEMORY_BANK_10D_PATH):
        print(f"10D Memory bank file not found at: {MEMORY_BANK_10D_PATH}")
        return

    data = np.load(MEMORY_BANK_10D_PATH)
    states = data["states"]
    actions = data["actions"]
    rewards = data["rewards"]

    print(f"Total experiences: {len(states):,}")
    print(f"Positive rewards: {np.sum(rewards > 0):,} ({np.mean(rewards > 0)*100:.1f}%)")
    print(f"Negative rewards: {np.sum(rewards < 0):,} ({np.mean(rewards < 0)*100:.1f}%)")

    # Calculate mean entropy of states (higher entropy = higher CNN confusion)
    entropies = -np.sum(states * np.log(states + 1e-10), axis=1)
    max_entropy = -np.log(1/10)  # Max possible entropy (uniform distribution)
    print(f"\nMean state entropy: {np.mean(entropies):.3f} / {max_entropy:.3f}")
    print(f"% high-entropy states (entropy > 2.0): {np.mean(entropies > 2.0)*100:.1f}%")

    # Calculate average maximum confidence
    max_probs = np.max(states, axis=1)
    print(f"Mean max CNN confidence: {np.mean(max_probs):.3f}")
    print(f"% low-confidence states (conf < 0.2): {np.mean(max_probs < 0.2)*100:.1f}%")

    # Analyze metrics across training epochs (split memory into three parts)
    num_parts = 3
    chunk_size = len(states) // num_parts
    for i, name in enumerate(["Epoch 1", "Epoch 2", "Epoch 3"]):
        sl = slice(i * chunk_size, (i + 1) * chunk_size)
        pct_positive = np.mean(rewards[sl] > 0) * 100
        mean_entropy = np.mean(entropies[sl])
        print(f"  {name}: {pct_positive:.1f}% positive rewards | Mean entropy: {mean_entropy:.3f}")

if __name__ == "__main__":
    main()
