"""
Hyperparameter Search Space Definition for Edge AI AutoML Experiments.

Defines the search domain for the deep semiparametric architecture on CIFAR-100,
covering CNN latent bottleneck dimension, dual uncertainty arbiter thresholds
(Ledoit-Wolf Mahalanobis distance and Shannon predictive entropy), episodic memory
capacity, nearest neighbors k, curriculum alpha decay, and reinforcement learning
hyperparameters.
"""

from __future__ import annotations
import itertools
import random
from typing import Dict, List, Any, Optional
import numpy as np

# Canonical discrete search space specification for CIFAR-100 Edge AI Case Study
SEARCH_SPACE: Dict[str, List[Any]] = {
    "latent_dim": [128, 256, 512],
    "mahalanobis_threshold": [10.0, 15.0, 20.0, 25.0],
    "entropy_threshold": [1.5, 2.0, 3.0, 4.0],
    "curriculum_alpha_decay": [0.990, 0.995, 0.999],
    "memory_capacity": [3000, 5000, 8000],
    "rl_lr": [1e-4, 5e-4, 1e-3],
    "rl_gamma": [0.95, 0.99],
    "knn_k": [10, 20, 30],
    "min_alpha": [0.01, 0.05, 0.10],
}


class SearchSpace:
    """
    Manages the hyperparameter exploration space for AutoML trials.

    Supports deterministic grid exploration and pseudo-random sampling
    with seeded repeatability.
    """

    def __init__(
        self,
        seed: int = 42,
        custom_space: Optional[Dict[str, List[Any]]] = None,
    ) -> None:
        """
        Initializes the SearchSpace.

        Args:
            seed: Random seed for deterministic reproducibility.
            custom_space: Optional dictionary overriding default SEARCH_SPACE.
        """
        self.seed: int = seed
        self._rng: random.Random = random.Random(seed)
        self.space: Dict[str, List[Any]] = (
            dict(custom_space) if custom_space is not None else dict(SEARCH_SPACE)
        )
        self.param_names: List[str] = list(self.space.keys())

    def sample_random(self) -> Dict[str, Any]:
        """
        Samples a single configuration randomly from the discrete parameter sets.

        Returns:
            Dict[str, Any]: Configuration dictionary with one value per hyperparameter.
        """
        sample: Dict[str, Any] = {}
        for key, values in self.space.items():
            if not values:
                raise ValueError(f"Parameter list for '{key}' is empty.")
            sample[key] = self._rng.choice(values)
        return sample

    def sample_n_random(self, n: int) -> List[Dict[str, Any]]:
        """
        Samples n unique or independent random parameter configurations.

        Args:
            n: Number of samples to generate.

        Returns:
            List of hyperparameter configuration dictionaries.
        """
        return [self.sample_random() for _ in range(n)]

    def get_grid(self) -> List[Dict[str, Any]]:
        """
        Constructs the full Cartesian product grid across all search space parameters.

        Returns:
            List[Dict[str, Any]]: Exhaustive list of all candidate combinations.
        """
        keys = list(self.space.keys())
        value_lists = [self.space[k] for k in keys]
        combinations = itertools.product(*value_lists)

        grid: List[Dict[str, Any]] = []
        for combo in combinations:
            config = dict(zip(keys, combo))
            grid.append(config)
        return grid

    @property
    def total_combinations(self) -> int:
        """Computes the cardinality of the full Cartesian product grid."""
        total = 1
        for values in self.space.values():
            total *= len(values)
        return total

    def __len__(self) -> int:
        return self.total_combinations

    def __repr__(self) -> str:
        return (
            f"SearchSpace(seed={self.seed}, num_params={len(self.space)}, "
            f"total_combinations={self.total_combinations:,})"
        )
