"""
[DEPRECATED] Reinforcement Learning Agent Module (Contextual Bandit).
This agent acts as the "Specialist" to resolve ambiguous cases rejected by the CNN.

WARNING: This module is DEPRECATED. It has been replaced by KNNBanditAgent in:
    src/models/knn_bandit_agent.py
    
The new agent uses k-Nearest Neighbors (Episodic Memory) instead of an MLP,
eliminating the need for backpropagation and heavy training epochs.

This file is maintained solely for reference, backward compatibility, and benchmark comparisons.
"""

import tensorflow as tf
import numpy as np
from src.scratch.layers import DenseLayer
from src.scratch.activations import relu

class QNetworkAgent(tf.Module):
    """
    Q-Network Agent using a Multi-Layer Perceptron (MLP).
    
    Parameters:
        state_dim (int): Dimensionality of the input state (default: 128).
        action_dim (int): Number of possible actions (default: 10).
        name (str): Name of the module.
    """
    def __init__(self, state_dim: int = 128, action_dim: int = 10, name: str = "q_network"):
        super().__init__(name=name)
        
        # --- THE AGENT'S BRAIN ---
        # A simple architecture mapping the latent features to the 10 actions.
        self.dense1 = DenseLayer(in_features=state_dim, out_features=64, name=f"{name}_fc1")
        self.dense2 = DenseLayer(in_features=64, out_features=action_dim, name=f"{name}_fc2")

    def __call__(self, state: tf.Tensor) -> tf.Tensor:
        """
        Forward pass of the agent.
        Receives the state (typically 128D) and returns the Q-values (10D).
        Note: No softmax is used at the output; values represent raw expected rewards.
        """
        x = self.dense1(state)
        x = relu(x)
        q_values = self.dense2(x)  # Output shape: [batch_size, 10]
        return q_values

    def get_action(self, state: tf.Tensor, epsilon: float) -> int:
        """
        Epsilon-greedy decision policy.
        """
        # Decide whether to explore or exploit
        if np.random.rand() < epsilon:
            # Exploration: Choose a random action
            chosen_action = np.random.randint(0, 10)
        else:
            # Exploitation: Ask the Q-Network for the most profitable action
            q_values = self(state)
            chosen_action = int(tf.argmax(q_values[0]).numpy())
            
        return chosen_action
