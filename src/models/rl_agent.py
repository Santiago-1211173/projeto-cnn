"""
Deep Reinforcement Learning Agent Module (Double DQN with Prioritized Experience Replay).

This module implements:
1. SumTree: An array-based binary sum tree for O(log N) priority updates and sampling.
2. PrioritizedReplayBuffer (PER): Proportional prioritized replay buffer with importance sampling (IS)
   corrections and beta annealing (Schaul et al., 2015).
3. QNetwork: Lightweight PyTorch MLP (2 hidden layers, 64 units each, ReLU) for low-latency Edge AI inference.
4. RLAgent: Double DQN agent (van Hasselt et al., 2016) with 5D state representation and 4 discrete
   memory eviction actions for active episodic memory curation.

Scientific References:
- Schaul et al. (2015): Prioritized Experience Replay (PER).
- van Hasselt et al. (2016): Deep Reinforcement Learning with Double Q-learning.
- Zhou et al. (2024): Catcher+ — DRL-based cache replacement with PER in cloud/edge storage.
- Staffolani et al. (2023): RLQ — Double DQN for queue and memory allocation in distributed edge environments.
"""

from __future__ import annotations
import os
import logging
from typing import Tuple, Optional, Dict, Any, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import (
    RL_STATE_DIM,
    RL_N_ACTIONS,
    REPLAY_BUFFER_CAPACITY,
)

logger = logging.getLogger(__name__)


# =============================================================================
# SUM TREE FOR PRIORITIZED EXPERIENCE REPLAY
# =============================================================================

class SumTree:
    """
    Array-based Binary SumTree for O(log N) prioritized sampling and priority updates.

    Leaves store individual transition priorities. Internal nodes store the sum
    of their respective children. The root (index 0) stores the total priority sum.

    Tree indexing convention:
        - Total capacity: C
        - Total tree array size: 2 * C - 1
        - Leaf nodes range: [C - 1, 2 * C - 2]
        - Data slot d in [0, C - 1] corresponds to tree leaf: d + C - 1
        - Parent of node i: (i - 1) // 2
        - Left child of node i: 2 * i + 1
        - Right child of node i: 2 * i + 2
    """

    def __init__(self, capacity: int) -> None:
        """
        Initializes the SumTree with fixed leaf capacity.

        Args:
            capacity: Maximum number of experiences/leaf nodes.
        """
        self.capacity: int = capacity
        self.tree: np.ndarray = np.zeros(2 * capacity - 1, dtype=np.float64)

    def update(self, tree_idx: int, priority: float) -> None:
        """
        Updates the priority of a leaf node and propagates the delta upward to the root.

        Args:
            tree_idx: Array index in tree structure.
            priority: New positive priority value.
        """
        delta = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority

        current_idx = tree_idx
        while current_idx > 0:
            current_idx = (current_idx - 1) // 2
            self.tree[current_idx] += delta

    def get_leaf(self, value: float) -> Tuple[int, float, int]:
        """
        Searches the tree for the leaf corresponding to a cumulative priority value.

        Args:
            value: Cumulative priority query in [0, total_priority].

        Returns:
            Tuple of:
                - tree_idx: Array index in tree structure.
                - priority: Priority value stored at leaf.
                - data_idx: Corresponding index in replay data buffer [0, capacity - 1].
        """
        parent_idx = 0
        while True:
            left_child = 2 * parent_idx + 1
            right_child = left_child + 1

            if left_child >= len(self.tree):
                # Leaf reached
                leaf_idx = parent_idx
                break

            if value <= self.tree[left_child]:
                parent_idx = left_child
            else:
                value -= self.tree[left_child]
                parent_idx = right_child

        data_idx = leaf_idx - (self.capacity - 1)
        data_idx = min(max(0, data_idx), self.capacity - 1)
        return leaf_idx, float(self.tree[leaf_idx]), data_idx

    @property
    def total_priority(self) -> float:
        """Returns the root sum of all priorities."""
        return float(self.tree[0])


# =============================================================================
# PRIORITIZED EXPERIENCE REPLAY BUFFER (PER)
# =============================================================================

class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay (PER) Buffer using SumTree.

    Stores transitions in pre-allocated NumPy arrays to guarantee O(1) insertion,
    zero memory re-allocations, and O(log N) prioritized sampling.
    Implements Importance Sampling (IS) weight correction and linear beta annealing.

    Attributes:
        capacity (int): Maximum transition capacity.
        state_dim (int): Dimensionality of state vectors.
        alpha (float): Priority exponent hyperparameter P(i) ~ p_i^alpha.
        beta (float): Current importance sampling weight exponent.
        beta_0 (float): Initial beta value at step 0.
        beta_increment (float): Linear increment applied to beta per sample call.
        epsilon (float): Small positive constant added to TD-error to avoid zero priority.
    """

    def __init__(
        self,
        capacity: int = REPLAY_BUFFER_CAPACITY,
        state_dim: int = RL_STATE_DIM,
        alpha: float = 0.6,
        beta_0: float = 0.4,
        beta_increment: float = 1e-4,
        epsilon: float = 1e-5,
    ) -> None:
        self.capacity: int = capacity
        self.state_dim: int = state_dim
        self.alpha: float = alpha
        self.beta: float = beta_0
        self.beta_0: float = beta_0
        self.beta_increment: float = beta_increment
        self.epsilon: float = epsilon

        self.max_priority: float = 1.0
        self.ptr: int = 0
        self.size: int = 0

        # SumTree structure
        self.tree = SumTree(capacity)

        # Pre-allocated transition buffers (NumPy contiguous memory)
        self.states: np.ndarray = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions: np.ndarray = np.zeros(capacity, dtype=np.int64)
        self.rewards: np.ndarray = np.zeros(capacity, dtype=np.float32)
        self.next_states: np.ndarray = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones: np.ndarray = np.zeros(capacity, dtype=np.bool_)

    def store(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """
        Stores a transition with maximal priority to ensure immediate exploration.

        Args:
            state: Starting state array of shape (state_dim,).
            action: Discrete action taken.
            reward: Scalar reward received.
            next_state: Resulting state array of shape (state_dim,).
            done: Terminal flag.
        """
        data_idx = self.ptr
        tree_idx = data_idx + self.capacity - 1

        self.states[data_idx] = np.asarray(state, dtype=np.float32).reshape(-1)
        self.actions[data_idx] = int(action)
        self.rewards[data_idx] = float(reward)
        self.next_states[data_idx] = np.asarray(next_state, dtype=np.float32).reshape(-1)
        self.dones[data_idx] = bool(done)

        # Assign maximum priority initially so new experiences are guaranteed to be sampled
        priority = float(self.max_priority ** self.alpha)
        self.tree.update(tree_idx, priority)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def __len__(self) -> int:
        """Returns current number of stored experiences."""
        return self.size

    def sample(
        self,
        batch_size: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Samples a prioritized mini-batch with Importance Sampling weights.

        Args:
            batch_size: Number of transitions to sample.

        Returns:
            Tuple of:
                - states: shape (batch_size, state_dim)
                - actions: shape (batch_size,)
                - rewards: shape (batch_size,)
                - next_states: shape (batch_size, state_dim)
                - dones: shape (batch_size,)
                - weights: Importance sampling weights normalized by max(w), shape (batch_size,)
                - tree_indices: Tree array indices for subsequent priority updates, shape (batch_size,)
        """
        if self.size < batch_size:
            raise ValueError(f"Not enough transitions in buffer ({self.size} < {batch_size})")

        total_p = self.tree.total_priority
        if total_p <= 0.0 or not np.isfinite(total_p):
            total_p = 1.0

        segment = total_p / batch_size

        tree_indices = np.zeros(batch_size, dtype=np.int32)
        data_indices = np.zeros(batch_size, dtype=np.int32)
        priorities = np.zeros(batch_size, dtype=np.float64)

        for i in range(batch_size):
            low = i * segment
            high = (i + 1) * segment
            v = np.random.uniform(low, high)
            tree_idx, priority, data_idx = self.tree.get_leaf(v)

            # Safety fallback for initial buffer boundary wrap-around
            if data_idx >= self.size:
                data_idx = np.random.randint(0, self.size)
                tree_idx = data_idx + self.capacity - 1
                priority = float(self.tree.tree[tree_idx])

            tree_indices[i] = tree_idx
            data_indices[i] = data_idx
            priorities[i] = max(priority, 1e-8)

        # Importance Sampling Weights: w_i = (N * P(i))^(-beta)
        probs = priorities / total_p
        weights = (float(self.size) * probs) ** (-self.beta)
        weights /= (np.max(weights) + 1e-8)  # Standard normalize by max weight
        weights = weights.astype(np.float32)

        # Beta annealing toward 1.0
        self.beta = min(1.0, self.beta + self.beta_increment)

        return (
            self.states[data_indices],
            self.actions[data_indices],
            self.rewards[data_indices],
            self.next_states[data_indices],
            self.dones[data_indices],
            weights,
            tree_indices,
        )

    def update_priorities(self, tree_indices: np.ndarray, td_errors: np.ndarray) -> None:
        """
        Updates priorities of sampled transitions based on absolute TD-errors.

        Args:
            tree_indices: Tree indices of sampled transitions.
            td_errors: Absolute TD errors (|target - current_Q|).
        """
        for tree_idx, td_err in zip(tree_indices, td_errors):
            p = float((abs(td_err) + self.epsilon) ** self.alpha)
            self.max_priority = max(self.max_priority, p)
            self.tree.update(int(tree_idx), p)


# =============================================================================
# Q-NETWORK ARCHITECTURE
# =============================================================================

class QNetwork(nn.Module):
    """
    Lightweight Multi-Layer Perceptron (MLP) for Q-value estimation.

    Consists of 2 hidden layers with 64 units each and ReLU activations,
    optimized for deterministic, sub-millisecond forward passes on Edge hardware.
    """

    def __init__(self, state_dim: int = RL_STATE_DIM, n_actions: int = RL_N_ACTIONS) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through QNetwork."""
        return self.network(x)


# =============================================================================
# DOUBLE DQN AGENT
# =============================================================================

class RLAgent:
    """
    Double DQN Agent with Prioritized Experience Replay for Active Memory Management.

    Exposes 4 discrete actions:
        - 0: Ignore (Do not insert or evict)
        - 1: Evict FIFO (evict_oldest)
        - 2: Evict LFU (evict_least_frequently_used)
        - 3: Evict Redundant (evict_most_redundant)

    Operates over a 5D normalized state vector:
        [Dist_Mahalanobis, Entropia_Local, Dist_Minima_KNN, Erro_Predicao_Atual, Ocupacao_RAM]
    """

    def __init__(
        self,
        state_dim: int = RL_STATE_DIM,
        n_actions: int = RL_N_ACTIONS,
        lr: float = 1e-3,
        gamma: float = 0.99,
        target_update_interval: int = 100,
        buffer_capacity: int = REPLAY_BUFFER_CAPACITY,
        device: Optional[str] = None,
    ) -> None:
        """
        Initializes the Double DQN agent and target network.

        Args:
            state_dim: Number of input state features (default: 5).
            n_actions: Number of discrete eviction actions (default: 4).
            lr: Learning rate for Adam optimizer.
            gamma: Discount factor for Bellman updates.
            target_update_interval: Step frequency for syncing target network weights.
            buffer_capacity: Capacity of Prioritized Experience Replay buffer.
            device: Torch execution device ('cpu' or 'cuda').
        """
        self.state_dim: int = state_dim
        self.n_actions: int = n_actions
        self.gamma: float = gamma
        self.target_update_interval: int = target_update_interval

        # Device selection
        if device is None:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        # Networks: Policy and Target
        self.policy_net = QNetwork(state_dim=self.state_dim, n_actions=self.n_actions).to(self.device)
        self.target_net = QNetwork(state_dim=self.state_dim, n_actions=self.n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)

        # Prioritized Replay Buffer
        self.replay_buffer = PrioritizedReplayBuffer(
            capacity=buffer_capacity,
            state_dim=self.state_dim,
        )

        self.train_step: int = 0

    # =========================================================================
    # STATE VECTOR CONSTRUCTION
    # =========================================================================

    def get_state_vector(
        self,
        mahalanobis_dist: float,
        local_entropy: float,
        min_knn_dist: float,
        prediction_error: float,
        ram_occupancy: float,
    ) -> np.ndarray:
        """
        Constructs and normalizes the 5D state representation into [0.0, 1.0].

        Normalization scales:
            - mahalanobis_dist: Scaled by 50.0 (covering nominal OOD spectrum)
            - local_entropy: Scaled by ln(10) ~ 2.3026 (max Shannon entropy for 10 classes)
            - min_knn_dist: Scaled by 10.0 (nominal Euclidean radius in 128D)
            - prediction_error: Binary error in {0.0, 1.0} or prob delta in [0.0, 1.0]
            - ram_occupancy: Occupancy ratio (size / capacity) in [0.0, 1.0]

        Returns:
            np.ndarray: 5D state vector with dtype np.float32 and shape (5,).
        """
        s_mahal = float(np.clip(mahalanobis_dist / 50.0, 0.0, 1.0))
        s_entropy = float(np.clip(local_entropy / 2.3026, 0.0, 1.0))
        s_dist = float(np.clip(min_knn_dist / 10.0, 0.0, 1.0))
        s_err = float(np.clip(prediction_error, 0.0, 1.0))
        s_ram = float(np.clip(ram_occupancy, 0.0, 1.0))

        return np.array([s_mahal, s_entropy, s_dist, s_err, s_ram], dtype=np.float32)

    # =========================================================================
    # ACTION SELECTION
    # =========================================================================

    def select_action(self, state: np.ndarray, epsilon: float = 0.1) -> int:
        """
        Selects an eviction action via epsilon-greedy policy.

        Args:
            state: 5D state vector of shape (5,).
            epsilon: Exploration probability in [0.0, 1.0].

        Returns:
            Discrete action index in {0, 1, 2, 3}.
        """
        if np.random.rand() < epsilon:
            return int(np.random.randint(0, self.n_actions))

        state_arr = np.asarray(state, dtype=np.float32)
        if np.any(state_arr > 1.0):
            state_arr = self.get_state_vector(
                mahalanobis_dist=float(state_arr[0]),
                local_entropy=float(state_arr[1]),
                min_knn_dist=float(state_arr[2]),
                prediction_error=float(state_arr[3]),
                ram_occupancy=float(state_arr[4]),
            )

        state_tensor = torch.as_tensor(state_arr, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
            best_action = int(torch.argmax(q_values, dim=1).item())

        return best_action

    # =========================================================================
    # TRANSITION STORAGE & DOUBLE DQN TRAINING
    # =========================================================================

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """
        Pushes transition into the Prioritized Experience Replay buffer.

        Args:
            state: Initial state array.
            action: Selected action integer.
            reward: Composite reward float.
            next_state: Resulting state array.
            done: Termination flag.
        """
        self.replay_buffer.store(state, action, reward, next_state, done)

    def update_weights(self, batch_size: int = 32) -> Optional[float]:
        """
        Performs one gradient optimization step using Double DQN with PER.

        Target formula:
            y = r + (1 - done) * gamma * target_net(s')[policy_net(s').argmax()]

        Args:
            batch_size: Number of transitions to sample.

        Returns:
            Scalar training loss float, or None if buffer has insufficient transitions.
        """
        if self.replay_buffer.size < batch_size:
            return None

        # 1. Sample prioritized mini-batch
        (
            b_states,
            b_actions,
            b_rewards,
            b_next_states,
            b_dones,
            b_weights,
            b_tree_indices,
        ) = self.replay_buffer.sample(batch_size)

        # 2. Convert to torch tensors
        states = torch.as_tensor(b_states, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(b_actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards = torch.as_tensor(b_rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.as_tensor(b_next_states, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(b_dones, dtype=torch.float32, device=self.device).unsqueeze(1)
        weights = torch.as_tensor(b_weights, dtype=torch.float32, device=self.device).unsqueeze(1)

        # 3. Current Q-values from policy network
        current_q = self.policy_net(states).gather(1, actions)

        # 4. Double DQN Target: policy_net selects action, target_net evaluates
        with torch.no_grad():
            next_policy_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
            next_target_q = self.target_net(next_states).gather(1, next_policy_actions)
            target_q = rewards + (1.0 - dones) * self.gamma * next_target_q

        # 5. TD-error calculation and PER priority updates
        td_errors = torch.abs(target_q - current_q).detach().cpu().numpy().flatten()
        self.replay_buffer.update_priorities(b_tree_indices, td_errors)

        # 6. Weighted Smooth L1 (Huber) loss
        per_sample_loss = F.smooth_l1_loss(current_q, target_q, reduction="none")
        loss = (weights * per_sample_loss).mean()

        # 7. Optimization with gradient clipping
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        # 8. Synchronize target network periodically
        self.train_step += 1
        if self.train_step % self.target_update_interval == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            logger.debug(f"Target network synced at step {self.train_step}")

        return float(loss.item())

    # Convenient alias for update_weights
    update = update_weights

    # =========================================================================
    # PERSISTENCE & STATS
    # =========================================================================

    def save(self, path: str) -> None:
        """
        Saves policy and target network weights and optimizer state to disk.

        Args:
            path: Target file path (.pt).
        """
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        save_payload = {
            "policy_net_state_dict": self.policy_net.state_dict(),
            "target_net_state_dict": self.target_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "train_step": self.train_step,
            "state_dim": self.state_dim,
            "n_actions": self.n_actions,
            "gamma": self.gamma,
        }
        torch.save(save_payload, path)
        logger.info(f"RLAgent checkpoint successfully saved to {path}")

    def load(self, path: str) -> None:
        """
        Loads policy and target network weights and optimizer state from disk.

        Args:
            path: Source file path (.pt).
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint["policy_net_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_net_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.train_step = checkpoint.get("train_step", 0)
        logger.info(f"RLAgent checkpoint successfully loaded from {path} (step {self.train_step})")

    def get_agent_stats(self) -> Dict[str, Any]:
        """
        Returns runtime statistics of the RL agent.

        Returns:
            Dictionary with training steps, buffer occupancy, and current beta.
        """
        return {
            "train_step": int(self.train_step),
            "buffer_size": int(self.replay_buffer.size),
            "buffer_capacity": int(self.replay_buffer.capacity),
            "current_beta": float(self.replay_buffer.beta),
            "max_priority": float(self.replay_buffer.max_priority),
        }
