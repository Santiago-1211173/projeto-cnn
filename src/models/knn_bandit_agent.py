"""
k-NN Bandit Agent Module (Episodic Memory).

Implements KNNBanditAgent128D, an instance-based episodic memory system
built with pre-allocated NumPy arrays for high-performance edge computing.
Eliminates dynamic Python list appending and memory fragmentation (O(1) updates).
Supports active eviction policies:
- evict_oldest: First-In-First-Out (FIFO)
- evict_least_frequently_used: Least Frequently Used (LFU)
- evict_most_redundant: Minimum distance within the same action/class
"""
from __future__ import annotations
import os
import logging
from typing import Tuple, Optional, Dict, Any, List, Union
import numpy as np

from src.config import (
    MEMORY_CAPACITY,
    LATENT_DIM,
    KNN_K_NEIGHBORS,
    KNN_N_ACTIONS,
)

logger = logging.getLogger(__name__)


class KNNBanditAgent128D:
    """
    Episodic Memory Agent using vectorized k-Nearest Neighbors with pure NumPy.

    Pre-allocates memory buffers to ensure strictly bounded RAM usage (no OOM),
    O(1) insertion/eviction mechanics, and fast vectorized k-NN search using
    np.linalg.norm and np.argpartition.

    Attributes:
        capacity (int): Maximum capacity of experiences stored in memory.
        k (int): Number of nearest neighbors to query.
        latent_dim (int): Dimensionality of the latent representation (default 128).
        n_actions (int): Number of discrete actions.
        size (int): Current number of experiences stored (0 <= size <= capacity).
        tick_counter (int): Monotonically increasing logical clock.
    """

    def __init__(
        self,
        capacity: int = MEMORY_CAPACITY,
        k: int = KNN_K_NEIGHBORS,
        latent_dim: int = LATENT_DIM,
        n_actions: int = KNN_N_ACTIONS,
        **kwargs: Any,
    ) -> None:
        self.capacity: int = capacity
        self.k: int = k
        self.latent_dim: int = latent_dim
        self.n_actions: int = n_actions

        # State properties
        self.size: int = 0
        self.tick_counter: int = 0

        # Pre-allocated memory arrays (O(1) allocation and access, zero heap fragmentation)
        self._states: np.ndarray = np.zeros((self.capacity, self.latent_dim), dtype=np.float32)
        self._actions: np.ndarray = np.zeros(self.capacity, dtype=np.int32)
        self._rewards: np.ndarray = np.zeros(self.capacity, dtype=np.float32)

        # Parallel metadata arrays for RL management policies
        self._insertion_ticks: np.ndarray = np.zeros(self.capacity, dtype=np.int64)
        self._usage_counts: np.ndarray = np.zeros(self.capacity, dtype=np.int32)

    @property
    def memory_size(self) -> int:
        """Backward-compatible alias for self.size."""
        return self.size

    # =========================================================================
    # MEMORY INSERTION
    # =========================================================================

    def add_experience(self, state: np.ndarray, action: int, reward: float) -> None:
        """
        Stores a single experience in episodic memory.

        If memory is full (size >= capacity), executes evict_oldest (FIFO) as the
        default mechanical eviction policy to preserve the capacity constraint.

        Args:
            state: Feature vector of shape (latent_dim,) or (1, latent_dim).
            action: Action index.
            reward: Scalar reward received.
        """
        state_flat = np.asarray(state, dtype=np.float32).reshape(-1)

        if self.size < self.capacity:
            idx = self.size
            self._states[idx] = state_flat
            self._actions[idx] = int(action)
            self._rewards[idx] = float(reward)
            self._insertion_ticks[idx] = self.tick_counter
            self._usage_counts[idx] = 0
            self.size += 1
            self.tick_counter += 1
        else:
            self.evict_oldest(state_flat, action, reward)

    def add_experience_batch(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
    ) -> None:
        """
        Stores a batch of experiences into episodic memory.

        Args:
            states: Array of shape (N, latent_dim).
            actions: Array of shape (N,).
            rewards: Array of shape (N,).
        """
        for s, a, r in zip(states, actions, rewards):
            self.add_experience(s, a, r)

    # =========================================================================
    # NEAREST NEIGHBORS SEARCH
    # =========================================================================

    def get_nearest_neighbors(
        self,
        query: np.ndarray,
        k: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Finds the k nearest neighbors for a query vector using pure NumPy vectorized distance.

        Uses np.argpartition on the occupied memory slice [:self.size] for O(N) selection,
        then sorts only the top-k items in ascending distance order.

        Args:
            query: Query feature vector of shape (latent_dim,) or (1, latent_dim).
            k: Number of neighbors to return. Defaults to self.k.

        Returns:
            Tuple of:
                - indices: Array of shape (k,) containing indices of nearest neighbors in memory.
                - distances: Array of shape (k,) containing Euclidean distances.

        Raises:
            RuntimeError: If episodic memory is empty.
        """
        if self.size == 0:
            raise RuntimeError("Episodic memory is empty! Cannot query nearest neighbors.")

        query_flat = np.asarray(query, dtype=np.float32).reshape(-1)
        effective_k = self.k if k is None else k
        actual_k = min(effective_k, self.size)

        diff = self._states[:self.size] - query_flat
        dists = np.linalg.norm(diff, axis=1)

        if actual_k < self.size:
            partition_idx = np.argpartition(dists, actual_k - 1)[:actual_k]
            sorted_order = np.argsort(dists[partition_idx])
            nearest_indices = partition_idx[sorted_order]
        else:
            nearest_indices = np.argsort(dists)[:actual_k]

        nearest_distances = dists[nearest_indices]
        return nearest_indices, nearest_distances

    # =========================================================================
    # EVICTION POLICIES
    # =========================================================================

    def evict_oldest(
        self,
        new_state: np.ndarray,
        new_action: int,
        new_reward: float,
    ) -> int:
        """
        Evicts the oldest experience based on insertion tick (FIFO policy).

        Overwrites the target slot with new experience, resets its usage count,
        updates the insertion tick, advances tick_counter, and returns the target index.

        Args:
            new_state: State vector to insert.
            new_action: Action label to insert.
            new_reward: Scalar reward to insert.

        Returns:
            Index of the evicted and overwritten experience slot.
        """
        if self.size == 0:
            self.add_experience(new_state, new_action, new_reward)
            return 0

        target_idx = int(np.argmin(self._insertion_ticks[:self.size]))
        state_flat = np.asarray(new_state, dtype=np.float32).reshape(-1)

        self._states[target_idx] = state_flat
        self._actions[target_idx] = int(new_action)
        self._rewards[target_idx] = float(new_reward)
        self._insertion_ticks[target_idx] = self.tick_counter
        self._usage_counts[target_idx] = 0
        self.tick_counter += 1

        return target_idx

    def evict_least_frequently_used(
        self,
        new_state: np.ndarray,
        new_action: int,
        new_reward: float,
    ) -> int:
        """
        Evicts the least frequently used experience (LFU policy).

        If multiple experiences share the minimum usage count, breaks ties
        by selecting the oldest experience (lowest insertion tick).

        Args:
            new_state: State vector to insert.
            new_action: Action label to insert.
            new_reward: Scalar reward to insert.

        Returns:
            Index of the evicted and overwritten experience slot.
        """
        if self.size == 0:
            self.add_experience(new_state, new_action, new_reward)
            return 0

        valid_usage = self._usage_counts[:self.size]
        min_usage = np.min(valid_usage)
        candidates = np.where(valid_usage == min_usage)[0]

        if len(candidates) == 1:
            target_idx = int(candidates[0])
        else:
            oldest_candidate = np.argmin(self._insertion_ticks[candidates])
            target_idx = int(candidates[oldest_candidate])

        state_flat = np.asarray(new_state, dtype=np.float32).reshape(-1)

        self._states[target_idx] = state_flat
        self._actions[target_idx] = int(new_action)
        self._rewards[target_idx] = float(new_reward)
        self._insertion_ticks[target_idx] = self.tick_counter
        self._usage_counts[target_idx] = 0
        self.tick_counter += 1

        return target_idx

    def evict_most_redundant(
        self,
        new_state: np.ndarray,
        new_action: int,
        new_reward: float,
    ) -> int:
        """
        Evicts the most redundant experience in the same class/action.

        Calculates Euclidean distances only against stored experiences with the
        exact same action/label, and replaces the closest geometric neighbor.
        If no experience with the same action exists, falls back to LFU eviction.

        Args:
            new_state: State vector to insert.
            new_action: Action label to insert.
            new_reward: Scalar reward to insert.

        Returns:
            Index of the evicted and overwritten experience slot.
        """
        if self.size == 0:
            self.add_experience(new_state, new_action, new_reward)
            return 0

        same_class_mask = (self._actions[:self.size] == new_action)
        same_class_indices = np.where(same_class_mask)[0]

        if len(same_class_indices) == 0:
            return self.evict_least_frequently_used(new_state, new_action, new_reward)

        state_flat = np.asarray(new_state, dtype=np.float32).reshape(-1)
        diff = self._states[same_class_indices] - state_flat
        dists = np.linalg.norm(diff, axis=1)
        min_idx = np.argmin(dists)
        target_idx = int(same_class_indices[min_idx])

        self._states[target_idx] = state_flat
        self._actions[target_idx] = int(new_action)
        self._rewards[target_idx] = float(new_reward)
        self._insertion_ticks[target_idx] = self.tick_counter
        self._usage_counts[target_idx] = 0
        self.tick_counter += 1

        return target_idx

    # =========================================================================
    # METADATA & REWARD INFERENCE
    # =========================================================================

    def increment_usage(self, indices: np.ndarray | List[int] | int) -> None:
        """
        Increments the usage count of specified memory slots.

        Args:
            indices: Array, list, or scalar index of slots in memory.
        """
        self._usage_counts[indices] += 1

    def build_index(self) -> None:
        """
        No-op method kept for backward compatibility with previous interface.
        Search uses dynamic, vectorized NumPy routines without external indexing.
        """
        if self.size == 0:
            raise ValueError("Episodic memory is empty! Populate it before building the index.")
        logger.info(
            f"NumPy k-NN index ready with {self.size} experiences. (capacity={self.capacity}, k={self.k})"
        )

    def get_expected_rewards(self, state: np.ndarray) -> np.ndarray:
        """
        Calculates distance-weighted expected reward for each possible action.

        Weights are inversely proportional to distance (1 / (dist + 1e-8)),
        accumulated per action across the top-k neighbors.

        Args:
            state: Query state vector.

        Returns:
            Array of shape (n_actions,) with estimated expected rewards.
        """
        if self.size == 0:
            return np.zeros(self.n_actions, dtype=np.float32)

        indices, distances = self.get_nearest_neighbors(state, k=self.k)
        weights = 1.0 / (distances + 1e-8)

        neighbor_actions = self._actions[indices]
        neighbor_rewards = self._rewards[indices]

        expected_rewards = np.zeros(self.n_actions, dtype=np.float32)
        for action in range(self.n_actions):
            mask = neighbor_actions == action
            if np.any(mask):
                expected_rewards[action] = np.sum(neighbor_rewards[mask] * weights[mask])

        return expected_rewards

    def get_expected_rewards_batch(self, states: np.ndarray) -> np.ndarray:
        """
        Calculates expected rewards for a batch of states.

        Args:
            states: Array of shape (N, latent_dim).

        Returns:
            Array of shape (N, n_actions).
        """
        n_samples = len(states)
        result = np.zeros((n_samples, self.n_actions), dtype=np.float32)
        for i in range(n_samples):
            result[i] = self.get_expected_rewards(states[i])
        return result

    def get_action(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        """
        Epsilon-greedy action selection.

        Args:
            state: Query state vector.
            epsilon: Exploration probability.

        Returns:
            Selected action index.
        """
        if np.random.rand() < epsilon:
            return int(np.random.randint(0, self.n_actions))
        expected = self.get_expected_rewards(state)
        return int(np.argmax(expected))

    def get_action_batch(self, states: np.ndarray, epsilon: float = 0.0) -> np.ndarray:
        """
        Vectorized epsilon-greedy action selection for a batch of states.

        Args:
            states: Array of shape (N, latent_dim).
            epsilon: Exploration probability.

        Returns:
            Array of shape (N,) with chosen actions.
        """
        n = len(states)
        expected = self.get_expected_rewards_batch(states)
        greedy_actions = np.argmax(expected, axis=1)

        explore_mask = np.random.rand(n) < epsilon
        random_actions = np.random.randint(0, self.n_actions, size=n)

        return np.where(explore_mask, random_actions, greedy_actions)

    # =========================================================================
    # PERSISTENCE
    # =========================================================================

    def save(self, path: str) -> None:
        """
        Saves episodic memory and metadata to disk as a compressed .npz file.

        Args:
            path: Target file path.
        """
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        save_dict = {
            "states": self._states[:self.size],
            "actions": self._actions[:self.size],
            "rewards": self._rewards[:self.size],
            "insertion_ticks": self._insertion_ticks[:self.size],
            "usage_counts": self._usage_counts[:self.size],
            "size": np.array(self.size, dtype=np.int32),
            "capacity": np.array(self.capacity, dtype=np.int32),
            "tick_counter": np.array(self.tick_counter, dtype=np.int64),
            "latent_dim": np.array(self.latent_dim, dtype=np.int32),
        }
        np.savez_compressed(path, **save_dict)
        logger.info(f"Episodic memory saved to {path} ({self.size}/{self.capacity} experiences)")

    def load(self, path: str) -> None:
        """
        Loads episodic memory and metadata from disk (.npz file).

        Args:
            path: Source file path.
        """
        data = np.load(path)
        states = data["states"]
        actions = data["actions"]
        rewards = data["rewards"]

        n_loaded = len(states)
        if n_loaded > self.capacity:
            self.capacity = n_loaded

        if n_loaded > 0:
            self.latent_dim = states.shape[1]

        self._states = np.zeros((self.capacity, self.latent_dim), dtype=np.float32)
        self._actions = np.zeros(self.capacity, dtype=np.int32)
        self._rewards = np.zeros(self.capacity, dtype=np.float32)
        self._insertion_ticks = np.zeros(self.capacity, dtype=np.int64)
        self._usage_counts = np.zeros(self.capacity, dtype=np.int32)

        self._states[:n_loaded] = states
        self._actions[:n_loaded] = actions.astype(np.int32)
        self._rewards[:n_loaded] = rewards.astype(np.float32)

        if "insertion_ticks" in data and len(data["insertion_ticks"]) == n_loaded:
            self._insertion_ticks[:n_loaded] = data["insertion_ticks"]
            self._usage_counts[:n_loaded] = data["usage_counts"]
            self.tick_counter = int(data.get("tick_counter", np.max(self._insertion_ticks[:n_loaded]) + 1))
        else:
            self._insertion_ticks[:n_loaded] = np.arange(n_loaded, dtype=np.int64)
            self._usage_counts[:n_loaded] = 0
            self.tick_counter = n_loaded

        self.size = n_loaded
        logger.info(f"Memory loaded from {path} ({self.size} experiences, dim: {self.latent_dim})")

    # =========================================================================
    # DIAGNOSTICS
    # =========================================================================

    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Returns statistical metrics of the episodic memory.

        Returns:
            Dictionary containing size, capacity, occupancy, and reward statistics.
        """
        if self.size == 0:
            return {
                "size": 0,
                "capacity": self.capacity,
                "occupancy_pct": 0.0,
                "reward_mean": 0.0,
                "reward_positive_pct": 0.0,
                "actions_distribution": {int(a): 0 for a in range(self.n_actions)},
            }

        rewards = self._rewards[:self.size]
        actions = self._actions[:self.size]

        return {
            "size": int(self.size),
            "capacity": int(self.capacity),
            "occupancy_pct": float((self.size / self.capacity) * 100.0),
            "reward_mean": float(np.mean(rewards)),
            "reward_positive_pct": float(np.mean(rewards > 0) * 100.0),
            "actions_distribution": {
                int(a): int(np.sum(actions == a)) for a in range(self.n_actions)
            },
        }


# Backward-compatible alias
KNNBanditAgent = KNNBanditAgent128D
