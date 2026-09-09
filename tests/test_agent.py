"""
Unit Test Script for the k-NN Bandit Agent.
Validates episodic memory, k-NN search, and decision policy for both 10D and 128D (PCA) configurations.
"""

import sys
import os
import logging
import numpy as np

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from src.models.knn_bandit_agent import KNNBanditAgent
from src.config import OUTPUT_DIR

# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def test_agent_dimension(latent_dim, use_pca, pca_components=None):
    logger.info(f"\n==================================================")
    logger.info(f"Testing Agent: dim={latent_dim}, use_pca={use_pca}, pca_components={pca_components}")
    logger.info(f"==================================================")

    agent = KNNBanditAgent(k=5, n_actions=10, latent_dim=latent_dim, use_pca=use_pca, pca_components=pca_components or 48)

    # --- Test 1: Populate Memory ---
    logger.info("--- Test 1: Populate Episodic Memory ---")
    np.random.seed(42)
    for _ in range(100):
        # Create a state vector
        state = np.random.dirichlet(np.ones(latent_dim))
        true_digit = np.argmax(state) % 10
        action = true_digit
        reward = 1.0
        agent.add_experience(state, action, reward)
        
        # Add a negative experience
        wrong_action = (true_digit + np.random.randint(1, 10)) % 10
        agent.add_experience(state, wrong_action, -1.0)
    
    logger.info(f"Memory populated with {agent.memory_size} experiences")
    assert agent.memory_size == 200, f"Expected 200, got {agent.memory_size}"
    logger.info("   [OK] Memory size is correct!")

    # --- Test 2: Build Index ---
    logger.info("--- Test 2: Build k-NN Index ---")
    agent.build_index()
    logger.info("   [OK] Index built successfully!")

    # --- Test 3: Expected Rewards ---
    logger.info("--- Test 3: Query Expected Rewards ---")
    test_state = np.zeros(latent_dim, dtype=np.float32)
    test_state[7 % latent_dim] = 0.8
    test_state[1 % latent_dim] = 0.1
    
    expected_rewards = agent.get_expected_rewards(test_state)
    logger.info(f"Expected rewards: {expected_rewards}")
    logger.info(f"-> Selected action: {np.argmax(expected_rewards)}")
    logger.info("   [OK] Expected rewards successfully calculated!")

    # --- Test 4: Pure Exploitation ---
    logger.info("--- Test 4: Pure Exploitation (Epsilon = 0.0) ---")
    action = agent.get_action(test_state, epsilon=0.0)
    logger.info(f"Exploitation action: {action}")
    assert action == np.argmax(expected_rewards), "Action must be the argmax!"
    logger.info("   [OK] Greedy policy followed the highest expected reward!")

    # --- Test 5: Pure Exploration ---
    logger.info("--- Test 5: Pure Exploration (Epsilon = 1.0) ---")
    actions = [agent.get_action(test_state, epsilon=1.0) for _ in range(20)]
    logger.info(f"20 Random actions: {actions}")
    assert len(set(actions)) > 1, "Pure exploration must show diversity!"
    logger.info("   [OK] Agent explores randomly when epsilon=1.0!")

    # --- Test 6: Persistence (Save/Load) ---
    logger.info("--- Test 6: Persistence (Save/Load) ---")
    test_path = os.path.join(OUTPUT_DIR, f"test_knn_memory_{latent_dim}d.npz")
    os.makedirs(os.path.dirname(test_path), exist_ok=True)
    agent.save(test_path)
    
    agent2 = KNNBanditAgent(k=5, n_actions=10)
    agent2.load(test_path)
    
    assert agent2.memory_size == agent.memory_size, "Loaded memory bank must match original size!"
    action2 = agent2.get_action(test_state, epsilon=0.0)
    assert action2 == action, "Predictions must match after loading!"
    logger.info("   [OK] Memory saved and successfully restored!")
    
    # Cleanup
    os.remove(test_path)

    # --- Test 7: Statistics ---
    logger.info("--- Test 7: Memory Stats ---")
    stats = agent.get_memory_stats()
    logger.info(f"  Size: {stats['size']}")
    logger.info(f"  Mean Reward: {stats['reward_mean']:.3f}")
    logger.info(f"  Positive %: {stats['reward_positive_pct']:.1f}%")
    logger.info("   [OK] Memory statistics calculated!")

def main():
    # Test 10D mode (no PCA)
    test_agent_dimension(latent_dim=10, use_pca=False)
    
    # Test 128D mode with PCA enabled
    test_agent_dimension(latent_dim=128, use_pca=True, pca_components=8)

    logger.info("\n==================================================")
    logger.info("ALL k-NN AGENT TESTS COMPLETED SUCCESSFULLY!")
    logger.info("==================================================")

if __name__ == "__main__":
    main()