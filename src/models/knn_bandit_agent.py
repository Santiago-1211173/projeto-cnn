"""
k-NN Bandit Agent Module (Episodic Memory).
Replaces the old QNetworkAgent (MLP) with a non-parametric approach.

Instead of training a neural network, this agent stores triplets
(state, action, reward) and uses k-Nearest Neighbors search to estimate
the expected reward for each possible action.
"""

import os
import logging
import numpy as np
from sklearn.neighbors import NearestNeighbors
from src.config import KNN_K, KNN_N_ACTIONS

logger = logging.getLogger(__name__)


class KNNBanditAgent:
    """
    Reinforcement Learning Agent based on k-Nearest Neighbors.
    
    Architecture:
        - Episodic Memory: A database of past experiences.
        - k-NN Search: For each new state, finds the k most similar states
          observed in the past.
        - Decision: Calculates a distance-weighted score for each of the
          possible actions based on the neighbors' histories.
          Uses SUM (weighted vote) instead of MEAN so that more neighbors
          voting for an action results in a stronger signal.
    
    Parameters:
        k (int): Number of neighbors to query (default: 30).
        n_actions (int): Number of possible actions (default: 10).
        latent_dim (int): Dimensionality of the raw state features (default: 10).
        use_pca (bool): Whether to apply PCA dimensionality reduction (default: False).
        pca_components (int): Target dimension after PCA reduction (default: 48).
    """

    def __init__(
        self,
        k: int = KNN_K,
        n_actions: int = KNN_N_ACTIONS,
        latent_dim: int = 10,
        use_pca: bool = False,
        pca_components: int = 48
    ):
        self.k = k
        self.n_actions = n_actions
        self.latent_dim = latent_dim
        self.use_pca = use_pca
        self.pca_components = pca_components

        # --- EPISODIC MEMORY (Agent's Brain) ---
        self._states: list = []     # List of flattened state arrays
        self._actions: list = []    # List of integer actions taken
        self._rewards: list = []    # List of float rewards received

        # Search index and array cache
        self._nn_index: NearestNeighbors | None = None
        self._states_array: np.ndarray | None = None
        self.pca = None

    @property
    def memory_size(self) -> int:
        """Returns the total number of stored experiences."""
        return len(self._states)

    # =========================================================================
    # PHASE 1: POPULATE MEMORY
    # =========================================================================

    def add_experience(self, state: np.ndarray, action: int, reward: float) -> None:
        """
        Stores a single experience in episodic memory.
        
        Args:
            state: Array representing the state (10D or 128D).
            action: Action taken (0-9).
            reward: Reward received (+1.0 or -1.0).
        """
        self._states.append(np.asarray(state, dtype=np.float32).flatten())
        self._actions.append(int(action))
        self._rewards.append(float(reward))

    def add_experience_batch(self, states: np.ndarray, actions: np.ndarray, rewards: np.ndarray) -> None:
        """
        Stores a batch of experiences for efficiency.
        
        Args:
            states: Array of shape (N, D) representing states.
            actions: Array of shape (N,) representing actions.
            rewards: Array of shape (N,) representing rewards.
        """
        for i in range(len(states)):
            self._states.append(np.asarray(states[i], dtype=np.float32).flatten())
            self._actions.append(int(actions[i]))
            self._rewards.append(float(rewards[i]))

    # =========================================================================
    # PHASE 2: BUILD INDEX
    # =========================================================================

    def build_index(self) -> None:
        """
        Builds the k-NN index over the accumulated memory.
        Must be called after populating memory and before performing inference.
        """
        if self.memory_size == 0:
            raise ValueError("Episodic memory is empty! Populate it before building the index.")

        states_raw = np.array(self._states, dtype=np.float32)

        if self.use_pca:
            from sklearn.decomposition import PCA
            # Cap the number of components by memory size and original feature size
            n_comp = min(self.pca_components, len(states_raw), states_raw.shape[1])
            self.pca = PCA(n_components=n_comp, random_state=42)
            self._states_array = self.pca.fit_transform(states_raw)
        else:
            self._states_array = states_raw
            self.pca = None

        effective_dim = self._states_array.shape[1]
        effective_k = min(self.k, self.memory_size)

        # Select algorithm based on dimensionality (Ball Tree is efficient for <= 10D)
        algo = 'ball_tree' if effective_dim <= 10 else 'auto'
        n_jobs = -1 if effective_dim > 10 else None

        self._nn_index = NearestNeighbors(
            n_neighbors=effective_k,
            algorithm=algo,
            metric='euclidean',
            n_jobs=n_jobs
        )
        self._nn_index.fit(self._states_array)
        
        logger.info(
            f"k-NN Index built with {self.memory_size} experiences. "
            f"Effective dim: {effective_dim}, k: {effective_k}, algorithm: {algo}."
        )

    # =========================================================================
    # PHASE 3: INFERENCE (QUERYING MEMORY)
    # =========================================================================

    def _transform_query(self, state: np.ndarray) -> np.ndarray:
        """Helper to apply PCA transformation if enabled and fitted."""
        query = np.asarray(state, dtype=np.float32)
        if query.ndim == 1:
            query = query.reshape(1, -1)
        
        if self.use_pca and self.pca is not None:
            if hasattr(self.pca, "mean_") and hasattr(self.pca, "components_"):
                # Manual projection for scikit-learn version robustness
                return (query - self.pca.mean_) @ self.pca.components_.T
            else:
                return self.pca.transform(query)
        return query

    def get_expected_rewards(self, state: np.ndarray) -> np.ndarray:
        """
        Queries the k-NN index for the given state and calculates the
        distance-weighted score for each action.
        
        Uses weighted SUM so that:
        - More neighbors voting for an action increases the score.
        - Closer neighbors have higher weights (1 / distance).
        """
        if self._nn_index is None:
            raise RuntimeError("k-NN index has not been built! Call build_index() first.")

        query_transformed = self._transform_query(state)
        distances, indices = self._nn_index.kneighbors(query_transformed)
        neighbor_indices = indices[0]
        neighbor_distances = distances[0]

        # Weights inversely proportional to distance (add epsilon to avoid division by zero)
        weights = 1.0 / (neighbor_distances + 1e-8)

        neighbor_actions = np.array([self._actions[i] for i in neighbor_indices])
        neighbor_rewards = np.array([self._rewards[i] for i in neighbor_indices])

        expected_rewards = np.zeros(self.n_actions, dtype=np.float32)
        for action in range(self.n_actions):
            mask = neighbor_actions == action
            if np.any(mask):
                expected_rewards[action] = np.sum(neighbor_rewards[mask] * weights[mask])

        return expected_rewards

    def get_expected_rewards_batch(self, states: np.ndarray) -> np.ndarray:
        """Vectorized version of get_expected_rewards to process a batch of states."""
        if self._nn_index is None:
            raise RuntimeError("k-NN index has not been built! Call build_index() first.")

        states_transformed = self._transform_query(states)
        all_distances, all_indices = self._nn_index.kneighbors(states_transformed)

        actions_array = np.array(self._actions)
        rewards_array = np.array(self._rewards)

        result = np.zeros((len(states_transformed), self.n_actions), dtype=np.float32)

        for i, (neighbor_indices, neighbor_distances) in enumerate(zip(all_indices, all_distances)):
            weights = 1.0 / (neighbor_distances + 1e-8)
            neighbor_actions = actions_array[neighbor_indices]
            neighbor_rewards = rewards_array[neighbor_indices]

            for action in range(self.n_actions):
                mask = neighbor_actions == action
                if np.any(mask):
                    result[i, action] = np.sum(neighbor_rewards[mask] * weights[mask])

        return result

    def get_action(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        """Epsilon-Greedy action selection policy."""
        if np.random.rand() < epsilon:
            return int(np.random.randint(0, self.n_actions))

        expected = self.get_expected_rewards(state)
        return int(np.argmax(expected))

    def get_action_batch(self, states: np.ndarray, epsilon: float = 0.0) -> np.ndarray:
        """Vectorized Epsilon-Greedy policy for a batch of states."""
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
        """Saves episodic memory and PCA components to disk as a .npz file."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        
        save_dict = {
            "states": np.array(self._states, dtype=np.float32),
            "actions": np.array(self._actions, dtype=np.int32),
            "rewards": np.array(self._rewards, dtype=np.float32),
        }

        if self.use_pca and self.pca is not None:
            save_dict["pca_mean"] = self.pca.mean_
            save_dict["pca_components_"] = self.pca.components_

        np.savez_compressed(path, **save_dict)
        logger.info(f"Episodic memory saved to {path} ({self.memory_size} experiences)")

    def load(self, path: str) -> None:
        """Loads episodic memory and PCA components from disk, then rebuilds the index."""
        data = np.load(path)
        self._states = list(data["states"])
        self._actions = list(data["actions"].astype(int))
        self._rewards = list(data["rewards"].astype(float))

        # Auto-detect state dimension
        first_state = self._states[0] if len(self._states) > 0 else np.array([])
        self.latent_dim = first_state.shape[0] if first_state.size > 0 else 10

        # Load PCA components if present in file
        if "pca_mean" in data and "pca_components_" in data:
            self.use_pca = True
            from sklearn.decomposition import PCA
            self.pca = PCA(n_components=data["pca_components_"].shape[0])
            self.pca.mean_ = data["pca_mean"]
            self.pca.components_ = data["pca_components_"]
            self.pca.n_components_ = data["pca_components_"].shape[0]
            self.pca.n_features_in_ = data["pca_components_"].shape[1]
            self.pca_components = data["pca_components_"].shape[0]
        else:
            self.use_pca = False
            self.pca = None

        logger.info(f"Memory loaded from {path} ({self.memory_size} experiences, dim: {self.latent_dim})")
        self.build_index()

    # =========================================================================
    # DIAGNOSTICS
    # =========================================================================

    def get_memory_stats(self) -> dict:
        """Returns statistical metrics of the episodic memory."""
        if self.memory_size == 0:
            return {"size": 0}
        
        rewards = np.array(self._rewards)
        actions = np.array(self._actions)
        
        return {
            "size": self.memory_size,
            "reward_mean": float(np.mean(rewards)),
            "reward_positive_pct": float(np.mean(rewards > 0) * 100),
            "actions_distribution": {
                int(a): int(np.sum(actions == a)) for a in range(self.n_actions)
            }
        }


# Backward-compatible alias
KNNBanditAgent128D = KNNBanditAgent
