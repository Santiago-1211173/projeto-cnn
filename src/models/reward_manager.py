"""
Reward Manager Module for Episodic Memory Management via Curriculum Learning.

This module implements the RewardManager class responsible for orchestrating
curriculum-based reward signals for active episodic memory management.
The reward blends a fast Geometric Proxy (measuring local Euclidean coverage
and redundancy avoidance in memory) with empirical accuracy measured over a
Sliding Validation Buffer of hard Out-of-Distribution (OOD) instances.

Scientific References:
- Blundell et al. (2016) / MFEC (Pritzel et al., 2017): Transition from rapid non-parametric
  episodic memory retrieval to stable parametric validation.
- Freitag et al. (2024): Multi-stage reward curriculum for complex reward functions.
- Isele & Cosgun (2018): Distribution matching and coverage optimization in finite replay buffers.
"""

from __future__ import annotations
import logging
from typing import Optional, Dict, Any, TYPE_CHECKING
import numpy as np

from src.config import (
    SLIDING_VALIDATION_BUFFER_SIZE,
    CURRICULUM_ALPHA_DECAY,
    LATENT_DIM,
)

if TYPE_CHECKING:
    from src.models.knn_bandit_agent import KNNBanditAgent128D

logger = logging.getLogger(__name__)


class RewardManager:
    """
    Curriculum Learning Reward Orchestrator for Active Episodic Memory.

    Computes composite rewards interpolating between an instantaneous Geometric Proxy
    and task accuracy evaluated over a pre-allocated Sliding Validation Buffer.
    The curriculum weighting parameter (alpha) decays exponentially over training steps:
        R_total = alpha * R_geom + (1 - alpha) * R_acc
        alpha <- max(min_alpha, alpha * alpha_decay)

    Attributes:
        buffer_size (int): Maximum capacity of the sliding validation buffer.
        alpha_decay (float): Multiplicative decay applied to alpha per reward computation.
        min_alpha (float): Lower bound for alpha to preserve residual geometric regularization.
        latent_dim (int): Dimensionality of latent feature representations.
        alpha (float): Current curriculum interpolation weight in [min_alpha, 1.0].
    """

    def __init__(
        self,
        buffer_size: int = SLIDING_VALIDATION_BUFFER_SIZE,
        alpha_decay: float = CURRICULUM_ALPHA_DECAY,
        min_alpha: float = 0.01,
        latent_dim: int = LATENT_DIM,
    ) -> None:
        """
        Initializes the RewardManager and pre-allocates validation buffer arrays.

        Args:
            buffer_size: Number of hard/OOD validation samples to maintain.
            alpha_decay: Step-wise decay factor for curriculum interpolation parameter.
            min_alpha: Minimum floor for alpha decay.
            latent_dim: Latent representation dimension (default: 128).
        """
        self.buffer_size: int = buffer_size
        self.alpha_decay: float = alpha_decay
        self.min_alpha: float = min_alpha
        self.latent_dim: int = latent_dim

        # Curriculum weighting state
        self.alpha: float = 1.0

        # Pre-allocated sliding validation buffer (O(1) updates, zero heap allocation)
        self._val_states: np.ndarray = np.zeros(
            (self.buffer_size, self.latent_dim), dtype=np.float32
        )
        self._val_labels: np.ndarray = np.zeros(self.buffer_size, dtype=np.int32)
        self._val_size: int = 0
        self._val_ptr: int = 0

    # =========================================================================
    # VALIDATION BUFFER MANAGEMENT
    # =========================================================================

    def update_validation_buffer(
        self,
        state: np.ndarray,
        true_label: int,
        predicted_label: int,
    ) -> None:
        """
        Inserts a sample into the circular sliding validation buffer.

        Typically invoked whenever an incoming query exhibits prediction error
        or high uncertainty (hard OOD cases), maintaining a fresh streaming
        ground-truth evaluation set.

        Args:
            state: Feature vector of shape (latent_dim,) or (1, latent_dim).
            true_label: Ground truth target class index.
            predicted_label: Class predicted by the system prior to memory update.
        """
        state_flat = np.asarray(state, dtype=np.float32).reshape(-1)

        self._val_states[self._val_ptr] = state_flat
        self._val_labels[self._val_ptr] = int(true_label)

        self._val_ptr = (self._val_ptr + 1) % self.buffer_size
        self._val_size = min(self._val_size + 1, self.buffer_size)

    # =========================================================================
    # REWARD COMPUTATION
    # =========================================================================

    def compute_reward(
        self,
        evicted_index: int,
        memory: KNNBanditAgent128D,
        new_state: np.ndarray,
        characteristic_dist: float = 1.0,
        max_eval_samples: int = 15,
    ) -> float:
        """
        Computes the composite curriculum reward for a memory management decision.

        Args:
            evicted_index: Index of slot overwritten in memory (>= 0), or -1 if
                           the action was 'Ignore' (no insertion/eviction performed).
            memory: Reference to KNNBanditAgent128D memory structure.
            new_state: The candidate feature vector of shape (latent_dim,).
            characteristic_dist: Scaling distance for geometric proxy reward mapping.
            max_eval_samples: Number of samples from validation buffer to evaluate (default: 15).

        Returns:
            Scalar reward bounded strictly within [-1.0, 1.0].
        """
        new_state_flat = np.asarray(new_state, dtype=np.float32).reshape(-1)

        # ---------------------------------------------------------------------
        # 1. Geometric Proxy Reward (R_geom)
        # ---------------------------------------------------------------------
        r_geom = self._compute_geometric_proxy(
            evicted_index=evicted_index,
            memory=memory,
            new_state=new_state_flat,
            characteristic_dist=characteristic_dist,
        )

        # ---------------------------------------------------------------------
        # 2. Sliding Validation Accuracy Reward (R_acc)
        # ---------------------------------------------------------------------
        if self._val_size > 0 and memory.size > 0:
            r_acc = self._compute_validation_accuracy_reward(memory, max_eval_samples=max_eval_samples)
        else:
            # Fallback to geometric proxy when validation buffer is empty
            r_acc = r_geom

        # ---------------------------------------------------------------------
        # 3. Curriculum Interpolation & Decay
        # ---------------------------------------------------------------------
        composite_reward = (self.alpha * r_geom) + ((1.0 - self.alpha) * r_acc)
        bounded_reward = float(np.clip(composite_reward, -1.0, 1.0))

        # Advance curriculum decay
        self.alpha = float(max(self.min_alpha, self.alpha * self.alpha_decay))

        return bounded_reward

    def _compute_geometric_proxy(
        self,
        evicted_index: int,
        memory: KNNBanditAgent128D,
        new_state: np.ndarray,
        characteristic_dist: float,
    ) -> float:
        """
        Evaluates the geometric quality and diversity of the memory state.

        If new_state was stored (evicted_index >= 0):
            Measures distance to nearest neighbor in remaining memory.
            Penalizes storing redundant/duplicate samples; rewards coverage expansion.
        If new_state was ignored (evicted_index < 0):
            Rewards rejection of duplicates; penalizes rejection of novel vectors.

        Returns:
            Geometric proxy reward in [-1.0, 1.0].
        """
        mem_size = memory.size
        if mem_size == 0:
            return 0.0

        tau = max(1e-4, float(characteristic_dist))

        if evicted_index >= 0:
            # Memory insertion took place at evicted_index
            if mem_size <= 1:
                # First or single vector in memory
                return 0.5

            # Mask out the newly inserted slot to compare against existing entries
            mask = np.ones(mem_size, dtype=bool)
            if evicted_index < mem_size:
                mask[evicted_index] = False

            existing_states = memory._states[:mem_size][mask]
            if len(existing_states) == 0:
                return 0.5

            diffs = existing_states - new_state
            dists = np.linalg.norm(diffs, axis=1)
            min_dist = float(np.min(dists))

            # Smooth monotonic mapping: min_dist = 0 -> -1.0; min_dist = tau -> 0.0; min_dist >> tau -> +1.0
            r_geom = 2.0 * (min_dist / (min_dist + tau)) - 1.0

        else:
            # Action 0: Ignore / Reject new_state
            existing_states = memory._states[:mem_size]
            diffs = existing_states - new_state
            dists = np.linalg.norm(diffs, axis=1)
            min_dist = float(np.min(dists))

            # Rewarded if rejecting near-duplicate (min_dist -> 0 => +1.0)
            # Penalized if rejecting novel representation (min_dist >> tau => -1.0)
            r_geom = 1.0 - 2.0 * (min_dist / (min_dist + tau))

        return float(np.clip(r_geom, -1.0, 1.0))

    def _compute_validation_accuracy_reward(
        self,
        memory: KNNBanditAgent128D,
        max_eval_samples: int = 50,
    ) -> float:
        """
        Evaluates k-NN accuracy of memory against the validation buffer.

        Subsamples up to max_eval_samples from validation buffer to ensure
        strictly bounded, predictable O(1) latency on Edge hardware.

        Returns:
            Centered accuracy reward in [-1.0, 1.0].
        """
        eval_count = min(self._val_size, max_eval_samples)
        if eval_count == 0:
            return 0.0

        val_states = self._val_states[:eval_count]
        val_targets = self._val_labels[:eval_count]

        predictions = memory.get_action_batch(val_states, epsilon=0.0)
        accuracy = float(np.mean(predictions == val_targets))

        # Centered mapping: 100% accuracy -> +1.0, 50% accuracy -> 0.0, 0% accuracy -> -1.0
        r_acc = 2.0 * accuracy - 1.0
        return float(np.clip(r_acc, -1.0, 1.0))

    # =========================================================================
    # DIAGNOSTICS & STATE
    # =========================================================================

    def get_current_alpha(self) -> float:
        """
        Returns the current curriculum interpolation parameter alpha.

        Returns:
            Alpha value in [min_alpha, 1.0].
        """
        return float(self.alpha)

    def get_buffer_stats(self) -> Dict[str, Any]:
        """
        Returns statistical summary of the sliding validation buffer.

        Returns:
            Dictionary with occupancy, capacity, and current alpha.
        """
        return {
            "validation_size": int(self._val_size),
            "buffer_capacity": int(self.buffer_size),
            "occupancy_pct": float((self._val_size / self.buffer_size) * 100.0),
            "current_alpha": float(self.alpha),
        }
