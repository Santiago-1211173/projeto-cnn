"""
Online RL Simulation under Non-Stationary Environments (Concept Drift & Noise Stress Testing).

Phase 3 of EAAI Action Plan: Active Episodic Memory Management via Deep RL.

This module implements:
1. inject_noise: Gaussian noise perturbation simulating real and virtual concept drift.
2. MahalanobisPlusPlus: Out-of-Distribution (OOD) detector using L2-normalized latent features
   and Ledoit-Wolf Shrinkage covariance regularization.
3. TrainRLOnlineSimulation & run_simulation: Streaming prequential (test-then-train) simulation
   orchestrating CNN feature extraction, Mahalanobis routing, KNNBanditAgent128D memory curation,
   Curriculum Learning RewardManager, and Double DQN RLAgent with Prioritized Experience Replay.

Scientific References:
- Wu et al. (2026): Real vs. virtual concept drift in streaming machine learning.
- Pittorino & Roveri (2026): Agent-System-Environment (ASE) paradigm for non-stationary Edge AI.
- Haug et al. (2022): Prequential evaluation (test-then-train) for evolving data streams.
- Guo et al. (2025): Mahalanobis++: Improved OOD detection via feature normalization.
- Chen et al. (2010): Shrinkage Algorithms for MMSE Covariance Estimation (Ledoit-Wolf).
- Schaul et al. (2015): Prioritized Experience Replay (PER).
- van Hasselt et al. (2016): Deep Reinforcement Learning with Double Q-learning.
"""

from __future__ import annotations
import os
import sys
import csv
import time
import logging
import argparse
from typing import Tuple, Optional, Dict, List, Any, Union
import numpy as np
import tensorflow as tf
from sklearn.covariance import LedoitWolf

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import (
    DATA_DIR,
    CHECKPOINT_DIR,
    OUTPUT_DIR,
    RANDOM_SEED,
    MEMORY_CAPACITY,
    LATENT_DIM,
    KNN_K_NEIGHBORS,
    KNN_N_ACTIONS,
    RL_STATE_DIM,
    RL_N_ACTIONS,
    CURRICULUM_ALPHA_DECAY,
    SLIDING_VALIDATION_BUFFER_SIZE,
    REPLAY_BUFFER_CAPACITY,
    MAHALANOBIS_THRESHOLD,
    SIMULATION_STEPS,
    SIMULATION_NOISE_RATE,
    SIMULATION_NOISE_LEVEL,
    SIMULATION_LOG_INTERVAL,
    SIMULATION_CSV_PATH,
    RL_AGENT_CHECKPOINT_PATH,
    MAHALANOBIS_PP_PROFILES_PATH,
)
from src.models.custom_cnn import RawModel
from src.models.knn_bandit_agent import KNNBanditAgent128D
from src.models.reward_manager import RewardManager
from src.models.rl_agent import RLAgent
from src.data.loader import load_mnist_raw

# Configure module-level logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


# =============================================================================
# 1. NOISE INJECTION (CONCEPT DRIFT SIMULATOR)
# =============================================================================

def inject_noise(batch: np.ndarray, noise_level: float) -> np.ndarray:
    """
    Injects zero-mean Gaussian perturbation noise into an image batch to simulate sensor corruption.

    Follows the non-stationary Edge AI stress-testing methodology of Pittorino & Roveri (2026),
    inducing both virtual concept drift (covariate shift) and real concept drift (boundary disruption).

    Args:
        batch: Array of input images normalized to [0.0, 1.0].
               Shape can be (N, H, W, C), (N, H, W), or individual image (H, W, C).
        noise_level: Standard deviation (intensity) of Gaussian perturbation noise.

    Returns:
        Perturbed image array clipped strictly within [0.0, 1.0], with dtype np.float32.
    """
    if noise_level <= 0.0:
        return np.asarray(batch, dtype=np.float32).copy()

    batch_float = np.asarray(batch, dtype=np.float32)
    noise = np.random.normal(loc=0.0, scale=float(noise_level), size=batch_float.shape).astype(np.float32)
    corrupted = np.clip(batch_float + noise, 0.0, 1.0)
    return corrupted


# =============================================================================
# 2. MAHALANOBIS++ OUT-OF-DISTRIBUTION DETECTOR
# =============================================================================

class MahalanobisPlusPlus:
    """
    Mahalanobis++ Out-of-Distribution (OOD) Detector with Ledoit-Wolf Shrinkage.

    Implements feature-normalized distance evaluation in latent space:
        z_norm = z / (||z||_2 + 1e-8)
        d_M(z_norm, c) = sqrt((z_norm - mu_c)^T * Sigma_c^{-1} * (z_norm - mu_c))

    Applies the Ledoit-Wolf analytic shrinkage estimator (Chen et al., 2010)
    to guarantee well-conditioned, non-singular covariance matrices for high-dimensional
    edge embeddings without manual hyperparameter grid search.

    References:
        - Guo et al. (2025): Mahalanobis++: Improved Out-of-Distribution Detection.
        - Chen et al. (2010): Shrinkage Algorithms for MMSE Covariance Estimation.
        - Lee et al. (2018): A Simple Unified Framework for Detecting OOD Samples.
    """

    def __init__(
        self,
        n_classes: int = KNN_N_ACTIONS,
        latent_dim: int = LATENT_DIM,
        threshold: float = MAHALANOBIS_THRESHOLD,
    ) -> None:
        """
        Initializes the Mahalanobis++ OOD detector.

        Args:
            n_classes: Total number of discrete semantic classes (default: 10).
            latent_dim: Latent representation dimensionality (default: 128).
            threshold: Decision distance threshold separating In-Distribution from OOD.
        """
        self.n_classes: int = n_classes
        self.latent_dim: int = latent_dim
        self.threshold: float = threshold

        # Class profiles: mapping class index -> {"mu": np.ndarray (128,), "precision": np.ndarray (128, 128)}
        self.profiles: Dict[int, Dict[str, np.ndarray]] = {}
        self.is_fitted: bool = False

    def fit(self, latent_features: np.ndarray, labels: np.ndarray) -> None:
        """
        Fits class-conditional centroids and regularized precision matrices using Ledoit-Wolf Shrinkage.

        Args:
            latent_features: Unnormalized feature matrix of shape (N, latent_dim).
            labels: Integer target labels of shape (N,).
        """
        features_flat = np.asarray(latent_features, dtype=np.float32)
        labels_flat = np.asarray(labels, dtype=np.int32).reshape(-1)

        # 1. Apply L2 normalization to project features onto the unit hypersphere
        norms = np.linalg.norm(features_flat, axis=1, keepdims=True) + 1e-8
        normalized_features = features_flat / norms

        self.profiles.clear()

        for c in range(self.n_classes):
            idx = np.where(labels_flat == c)[0]
            if len(idx) < 2:
                # Handle sparse/missing class edge cases
                mu_c = np.zeros(self.latent_dim, dtype=np.float32)
                precision_c = np.eye(self.latent_dim, dtype=np.float32)
            else:
                class_feats = normalized_features[idx]
                lw = LedoitWolf().fit(class_feats)
                mu_c = lw.location_.astype(np.float32)
                precision_c = lw.precision_.astype(np.float32)

            self.profiles[c] = {
                "mu": mu_c,
                "precision": precision_c,
            }

        self.is_fitted = True
        logger.info(f"Mahalanobis++ successfully fitted across {self.n_classes} classes (dim={self.latent_dim}).")

    def compute_distance(self, latent_feature: np.ndarray, target_class: Optional[int] = None) -> float:
        """
        Computes the Mahalanobis++ distance for a single latent feature vector.

        If target_class is None, returns the minimum distance to the closest class centroid:
            min_{c} d_M(z_norm, c)

        Args:
            latent_feature: Vector of shape (latent_dim,) or (1, latent_dim).
            target_class: Optional specific class index to evaluate.

        Returns:
            Scalar Mahalanobis++ distance (non-negative float).
        """
        if not self.is_fitted:
            raise RuntimeError("MahalanobisPlusPlus must be fitted or loaded before computing distances.")

        z = np.asarray(latent_feature, dtype=np.float32).reshape(-1)
        z_norm = z / (np.linalg.norm(z) + 1e-8)

        if target_class is not None:
            if target_class not in self.profiles:
                return float("inf")
            p = self.profiles[target_class]
            diff = z_norm - p["mu"]
            sq_dist = float(diff @ p["precision"] @ diff.T)
            return float(np.sqrt(max(0.0, sq_dist)))

        # Evaluate against all classes and select minimum
        min_dist = float("inf")
        for c in range(self.n_classes):
            p = self.profiles[c]
            diff = z_norm - p["mu"]
            sq_dist = float(diff @ p["precision"] @ diff.T)
            dist = float(np.sqrt(max(0.0, sq_dist)))
            if dist < min_dist:
                min_dist = dist

        return min_dist

    def compute_distances_batch(self, latent_features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Vectorized computation of minimum Mahalanobis++ distances and closest classes for a batch.

        Args:
            latent_features: Array of shape (N, latent_dim).

        Returns:
            Tuple of:
                - min_distances: Array of shape (N,) with minimum Mahalanobis distances.
                - closest_classes: Array of shape (N,) with the corresponding closest class index.
        """
        if not self.is_fitted:
            raise RuntimeError("MahalanobisPlusPlus must be fitted before computing batch distances.")

        feats = np.asarray(latent_features, dtype=np.float32)
        n_samples = len(feats)
        norms = np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8
        feats_norm = feats / norms

        all_distances = np.zeros((n_samples, self.n_classes), dtype=np.float32)

        for c in range(self.n_classes):
            p = self.profiles[c]
            diff = feats_norm - p["mu"]
            # diff shape: (N, 128), precision shape: (128, 128)
            sq_dists = np.sum((diff @ p["precision"]) * diff, axis=1)
            all_distances[:, c] = np.sqrt(np.maximum(0.0, sq_dists))

        min_distances = np.min(all_distances, axis=1)
        closest_classes = np.argmin(all_distances, axis=1)
        return min_distances, closest_classes

    def is_out_of_distribution(self, latent_feature: np.ndarray) -> Tuple[bool, float]:
        """
        Evaluates whether a sample is Out-of-Distribution based on calibrated threshold.

        Args:
            latent_feature: Latent vector of shape (latent_dim,).

        Returns:
            Tuple of (is_ood: bool, min_distance: float).
        """
        dist = self.compute_distance(latent_feature)
        return (dist > self.threshold), dist

    def calibrate_threshold(self, clean_features: np.ndarray, percentile: float = 95.0) -> float:
        """
        Calibrates the OOD distance threshold based on empirical percentile of clean In-Distribution features.

        Args:
            clean_features: In-distribution feature array of shape (N, latent_dim).
            percentile: Percentile cutoff (default: 95.0%).

        Returns:
            Updated threshold float.
        """
        min_dists, _ = self.compute_distances_batch(clean_features)
        self.threshold = float(np.percentile(min_dists, percentile))
        logger.info(f"Mahalanobis++ threshold calibrated to {self.threshold:.2f} (percentile={percentile}%).")
        return self.threshold

    def save(self, path: str = MAHALANOBIS_PP_PROFILES_PATH) -> None:
        """
        Saves class profiles to compressed .npz archive.

        Args:
            path: Target file path.
        """
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        save_dict: Dict[str, Any] = {
            "n_classes": np.array(self.n_classes, dtype=np.int32),
            "latent_dim": np.array(self.latent_dim, dtype=np.int32),
            "threshold": np.array(self.threshold, dtype=np.float32),
        }
        for c in range(self.n_classes):
            save_dict[f"mu_{c}"] = self.profiles[c]["mu"]
            save_dict[f"precision_{c}"] = self.profiles[c]["precision"]

        np.savez_compressed(path, **save_dict)
        logger.info(f"Mahalanobis++ profiles saved to {path}.")

    def load(self, path: str = MAHALANOBIS_PP_PROFILES_PATH) -> None:
        """
        Loads class profiles from compressed .npz archive.

        Args:
            path: Source file path.
        """
        data = np.load(path)
        self.n_classes = int(data["n_classes"])
        self.latent_dim = int(data["latent_dim"])
        self.threshold = float(data["threshold"])

        self.profiles.clear()
        for c in range(self.n_classes):
            self.profiles[c] = {
                "mu": data[f"mu_{c}"].astype(np.float32),
                "precision": data[f"precision_{c}"].astype(np.float32),
            }

        self.is_fitted = True
        logger.info(f"Mahalanobis++ profiles successfully restored from {path} (threshold={self.threshold:.2f}).")


# =============================================================================
# 3. ONLINE STREAMING & FEATURE EXTRACTION PIPELINE
# =============================================================================

class OnlineStreamPipeline:
    """
    Streaming data and feature extraction manager for online simulation.

    Loads the custom CNN model, restores weights from disk, and extracts latent representations
    and predictions in deterministic chunks to minimize CPU/GPU dispatch overhead.
    """

    def __init__(self, data_dir: str = DATA_DIR, checkpoint_dir: str = CHECKPOINT_DIR) -> None:
        self.data_dir = data_dir
        self.checkpoint_dir = checkpoint_dir

        # Initialize CNN model
        self.cnn = RawModel()
        self.ckpt = tf.train.Checkpoint(model=self.cnn)
        latest_ckpt = tf.train.latest_checkpoint(self.checkpoint_dir)
        if latest_ckpt:
            self.ckpt.restore(latest_ckpt).expect_partial()
            logger.info(f"CNN feature extractor restored from {latest_ckpt}.")
        else:
            logger.warning(f"No checkpoint found in {self.checkpoint_dir}; using initial weights.")

        # Load MNIST dataset
        self.x_train, self.y_train = load_mnist_raw(self.data_dir, kind="train")
        self.x_train = self.x_train.astype(np.float32) / 255.0
        self.y_train = self.y_train.astype(np.int32)
        logger.info(f"Loaded {len(self.x_train):,} base samples from {self.data_dir}.")

    def extract_features_batch(
        self,
        images: np.ndarray,
        batch_size: int = 512,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extracts 128D latent vectors, predicted labels, and probability distributions in chunks.

        Args:
            images: Image array of shape (N, 28, 28, 1).
            batch_size: Batch extraction chunk size.

        Returns:
            Tuple of:
                - latent_features: Array of shape (N, 128)
                - predicted_labels: Array of shape (N,)
                - probabilities: Array of shape (N, 10)
        """
        all_latent: List[np.ndarray] = []
        all_preds: List[np.ndarray] = []
        all_probs: List[np.ndarray] = []

        total = len(images)
        for i in range(0, total, batch_size):
            chunk = images[i : i + batch_size]
            tensor_chunk = tf.convert_to_tensor(chunk, dtype=tf.float32)
            outputs = self.cnn(tensor_chunk)
            latent = outputs["latent_features"].numpy()
            probs = outputs["probabilities"].numpy()
            preds = np.argmax(probs, axis=1)

            all_latent.append(latent)
            all_preds.append(preds)
            all_probs.append(probs)

        return (
            np.vstack(all_latent).astype(np.float32),
            np.concatenate(all_preds).astype(np.int32),
            np.vstack(all_probs).astype(np.float32),
        )


# =============================================================================
# 4. ACTIVE MEMORY ONLINE SIMULATION ORCHESTRATOR
# =============================================================================

class TrainRLOnlineSimulation:
    """
    Active Episodic Memory Management Simulation Engine under Concept Drift.

    Coordinates the prequential streaming evaluation (test-then-train) protocol:
    1. Feature extraction via 128D CNN.
    2. Mahalanobis++ OOD detection with Ledoit-Wolf Shrinkage.
    3. Routing In-Distribution -> CNN, Anomaly/Hard OOD -> Episodic Memory (KNNBanditAgent128D).
    4. Active memory management via Double DQN RLAgent with Prioritized Experience Replay (PER).
    5. Continuous curriculum reward evaluation via sliding validation buffer in RewardManager.
    """

    def __init__(
        self,
        capacity: int = MEMORY_CAPACITY,
        k_neighbors: int = KNN_K_NEIGHBORS,
        latent_dim: int = LATENT_DIM,
        buffer_size: int = SLIDING_VALIDATION_BUFFER_SIZE,
        alpha_decay: float = CURRICULUM_ALPHA_DECAY,
        replay_capacity: int = REPLAY_BUFFER_CAPACITY,
        mahalanobis_threshold: float = MAHALANOBIS_THRESHOLD,
        device: Optional[str] = "cpu",
    ) -> None:
        """
        Initializes the complete simulation ecosystem.
        """
        self.capacity = capacity
        self.k_neighbors = k_neighbors
        self.latent_dim = latent_dim
        self.mahalanobis_threshold = mahalanobis_threshold
        self.device = device

        # Components
        self.memory = KNNBanditAgent128D(
            capacity=self.capacity,
            k=self.k_neighbors,
            latent_dim=self.latent_dim,
            n_actions=KNN_N_ACTIONS,
        )
        self.reward_manager = RewardManager(
            buffer_size=buffer_size,
            alpha_decay=alpha_decay,
            latent_dim=self.latent_dim,
        )
        self.rl_agent = RLAgent(
            state_dim=RL_STATE_DIM,
            n_actions=RL_N_ACTIONS,
            buffer_capacity=replay_capacity,
            device=self.device,
        )
        self.mah_detector = MahalanobisPlusPlus(
            n_classes=KNN_N_ACTIONS,
            latent_dim=self.latent_dim,
            threshold=self.mahalanobis_threshold,
        )

        self.pipeline: Optional[OnlineStreamPipeline] = None

    def initialize_detector(self, pipeline: OnlineStreamPipeline, num_samples: int = 5000) -> None:
        """
        Initializes or fits Mahalanobis++ profiles using clean reference data.
        """
        self.pipeline = pipeline
        clean_subset = pipeline.x_train[:num_samples]
        clean_labels = pipeline.y_train[:num_samples]

        clean_feats, _, _ = pipeline.extract_features_batch(clean_subset, batch_size=512)
        self.mah_detector.fit(clean_feats, clean_labels)

        # Calibrate threshold to 95th percentile of clean in-distribution data
        self.mah_detector.calibrate_threshold(clean_feats, percentile=95.0)

    def execute_simulation(
        self,
        n_episodes: int = 1,
        n_steps: int = SIMULATION_STEPS,
        noise_injection_rate: float = SIMULATION_NOISE_RATE,
        noise_level: float = SIMULATION_NOISE_LEVEL,
        batch_size: int = 32,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        log_interval: int = SIMULATION_LOG_INTERVAL,
        train_frequency: int = 4,
        output_csv_path: Optional[str] = SIMULATION_CSV_PATH,
        save_agent_path: Optional[str] = RL_AGENT_CHECKPOINT_PATH,
    ) -> Dict[str, List[float]]:
        """
        Executes the online streaming simulation with structured CSV logging and metrics tracking.

        Args:
            n_episodes: Total simulation episodes (typically 1 for streaming run).
            n_steps: Total sequential stream steps per episode (default: 50,000).
            noise_injection_rate: Fraction/probability of incoming samples corrupted with noise.
            noise_level: Gaussian noise intensity for corrupted samples.
            batch_size: Mini-batch size for Double DQN updates.
            epsilon_start: Initial exploration rate.
            epsilon_end: Final minimum exploration rate.
            log_interval: Step frequency for structured logging.
            output_csv_path: Target path for structured CSV log.
            save_agent_path: Target path for saving trained RLAgent weights.

        Returns:
            Dictionary containing metrics histories: rewards, losses, alphas, epsilons, sizes, steps.
        """
        if self.pipeline is None:
            self.pipeline = OnlineStreamPipeline()
            self.initialize_detector(self.pipeline)

        # Metrics aggregation lists
        all_rewards: List[float] = []
        all_losses: List[float] = []
        all_alphas: List[float] = []
        all_epsilons: List[float] = []
        all_sizes: List[float] = []
        all_steps: List[float] = []
        all_actions: List[int] = []

        # Prepare CSV logging
        csv_file = None
        csv_writer = None
        if output_csv_path:
            os.makedirs(os.path.dirname(output_csv_path) if os.path.dirname(output_csv_path) else ".", exist_ok=True)
            csv_file = open(output_csv_path, mode="w", newline="", encoding="utf-8")
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(["step", "size", "reward_mean", "alpha", "loss", "epsilon"])

        logger.info("=" * 70)
        logger.info(f"STARTING PHASE 3 ONLINE RL SIMULATION")
        logger.info(f"Episodes: {n_episodes} | Steps: {n_steps:,} | Capacity: {self.capacity:,}")
        logger.info(f"Noise Rate: {noise_injection_rate:.2f} | Noise Intensity: {noise_level:.2f}")
        logger.info(f"Mahalanobis++ Threshold: {self.mah_detector.threshold:.2f}")
        logger.info("=" * 70)

        # Prepare input stream generator in manageable memory-efficient chunks
        n_available = len(self.pipeline.x_train)
        chunk_size = 2048
        current_chunk_idx = 0
        current_chunk_pos = 0

        # Pre-allocate streaming chunk buffers
        stream_x_chunk = np.empty((0, 28, 28, 1), dtype=np.float32)
        stream_y_chunk = np.empty(0, dtype=np.int32)
        stream_feats_chunk = np.empty((0, self.latent_dim), dtype=np.float32)
        stream_preds_chunk = np.empty(0, dtype=np.int32)
        stream_mah_dists_chunk = np.empty(0, dtype=np.float32)
        stream_entropies_chunk = np.empty(0, dtype=np.float32)
        stream_pred_errors_chunk = np.empty(0, dtype=np.float32)
        stream_is_anomalies_chunk = np.empty(0, dtype=np.bool_)

        def refill_stream_chunk() -> None:
            nonlocal stream_x_chunk, stream_y_chunk, stream_feats_chunk, stream_preds_chunk
            nonlocal stream_mah_dists_chunk, stream_entropies_chunk, stream_pred_errors_chunk, stream_is_anomalies_chunk
            nonlocal current_chunk_idx, current_chunk_pos

            # Sample indices with wrapping
            indices = (np.arange(current_chunk_idx, current_chunk_idx + chunk_size)) % n_available
            current_chunk_idx = (current_chunk_idx + chunk_size) % n_available

            x_batch = self.pipeline.x_train[indices].copy()
            y_batch = self.pipeline.y_train[indices].copy()

            # Apply noise perturbation to a subset of samples based on noise_injection_rate
            noise_mask = np.random.rand(chunk_size) < noise_injection_rate
            if np.any(noise_mask):
                x_batch[noise_mask] = inject_noise(x_batch[noise_mask], noise_level)

            # Pass through CNN in single batch
            feats, preds, probs = self.pipeline.extract_features_batch(x_batch, batch_size=chunk_size)

            # Vectorized Mahalanobis++ distances and entropy across the entire chunk
            mah_dists, _ = self.mah_detector.compute_distances_batch(feats)
            entropies = -np.sum(probs * np.log(probs + 1e-12), axis=1).astype(np.float32)
            is_oods = (mah_dists > self.mah_detector.threshold)
            pred_errors = (preds != y_batch).astype(np.float32)
            is_anomalies = is_oods | (pred_errors > 0.0)

            stream_x_chunk = x_batch
            stream_y_chunk = y_batch
            stream_feats_chunk = feats
            stream_preds_chunk = preds
            stream_mah_dists_chunk = mah_dists
            stream_entropies_chunk = entropies
            stream_pred_errors_chunk = pred_errors
            stream_is_anomalies_chunk = is_anomalies
            current_chunk_pos = 0

        refill_stream_chunk()

        recent_rewards: List[float] = []
        recent_loss: float = 0.0
        start_time = time.time()

        for episode in range(n_episodes):
            for step in range(n_steps):
                # Ensure streaming buffer has available samples
                if current_chunk_pos >= len(stream_feats_chunk):
                    refill_stream_chunk()

                z_t = stream_feats_chunk[current_chunk_pos]
                y_true = int(stream_y_chunk[current_chunk_pos])
                pred_cnn = int(stream_preds_chunk[current_chunk_pos])
                d_M = float(stream_mah_dists_chunk[current_chunk_pos])
                local_entropy = float(stream_entropies_chunk[current_chunk_pos])
                prediction_error = float(stream_pred_errors_chunk[current_chunk_pos])
                is_anomaly = bool(stream_is_anomalies_chunk[current_chunk_pos])
                current_chunk_pos += 1

                # Linear epsilon annealing: epsilon_start -> epsilon_end
                progress = min(1.0, float(step / max(1, n_steps - 1)))
                current_epsilon = float(max(epsilon_end, epsilon_start - (epsilon_start - epsilon_end) * progress))

                # Update sliding validation buffer on prediction error or hard OOD samples
                if is_anomaly:
                    self.reward_manager.update_validation_buffer(z_t, y_true, pred_cnn)

                # 2. Decision & Active Memory Curation
                if self.memory.size < self.capacity:
                    # Phase A: Memory filling stage (capacity not yet reached)
                    self.memory.add_experience(z_t, y_true, reward=1.0)
                    step_reward = 1.0 if (pred_cnn == y_true) else 0.5
                    all_rewards.append(step_reward)
                    recent_rewards.append(step_reward)

                elif is_anomaly:
                    # Phase B: Memory saturated (size == capacity) and receives an anomaly
                    # Active RL Eviction is triggered!
                    # Vectorized fast Euclidean distance to nearest neighbor in memory
                    diff = self.memory._states[:self.memory.size] - z_t
                    min_knn_dist = float(np.sqrt(np.min(np.sum(diff * diff, axis=1))))
                    ram_occupancy = float(self.memory.size / self.capacity)

                    state_vec = self.rl_agent.get_state_vector(
                        mahalanobis_dist=d_M,
                        local_entropy=local_entropy,
                        min_knn_dist=min_knn_dist,
                        prediction_error=prediction_error,
                        ram_occupancy=ram_occupancy,
                    )

                    # Select discrete action via epsilon-greedy policy:
                    # 0: Ignore, 1: FIFO, 2: LFU, 3: Redundant
                    action = self.rl_agent.select_action(state_vec, epsilon=current_epsilon)
                    all_actions.append(action)

                    # Execute eviction mechanic
                    if action == 0:
                        # Action 0: Ignore / Reject incoming representation
                        evicted_idx = -1
                    elif action == 1:
                        # Action 1: Evict oldest (FIFO)
                        evicted_idx = self.memory.evict_oldest(z_t, y_true, 1.0)
                    elif action == 2:
                        # Action 2: Evict least frequently used (LFU)
                        evicted_idx = self.memory.evict_least_frequently_used(z_t, y_true, 1.0)
                    elif action == 3:
                        # Action 3: Evict most redundant vector in same class
                        evicted_idx = self.memory.evict_most_redundant(z_t, y_true, 1.0)
                    else:
                        evicted_idx = self.memory.evict_oldest(z_t, y_true, 1.0)

                    # Compute composite curriculum reward in [-1.0, 1.0]
                    step_reward = self.reward_manager.compute_reward(
                        evicted_index=evicted_idx,
                        memory=self.memory,
                        new_state=z_t,
                    )
                    all_rewards.append(step_reward)
                    recent_rewards.append(step_reward)

                    # Construct next state representation
                    next_ram_occupancy = float(self.memory.size / self.capacity)
                    next_state_vec = self.rl_agent.get_state_vector(
                        mahalanobis_dist=d_M,
                        local_entropy=local_entropy,
                        min_knn_dist=min_knn_dist,
                        prediction_error=prediction_error,
                        ram_occupancy=next_ram_occupancy,
                    )

                    # Store transition into Prioritized Experience Replay buffer
                    self.rl_agent.store_transition(
                        state=state_vec,
                        action=action,
                        reward=step_reward,
                        next_state=next_state_vec,
                        done=False,
                    )

                    # Perform Double DQN gradient step periodically
                    if len(all_actions) % train_frequency == 0:
                        loss = self.rl_agent.update_weights(batch_size=batch_size)
                        if loss is not None:
                            recent_loss = float(loss)
                            all_losses.append(recent_loss)

                else:
                    # Phase C: In-distribution nominal sample handled cleanly by CNN
                    step_reward = 1.0 if (pred_cnn == y_true) else -1.0
                    all_rewards.append(step_reward)
                    recent_rewards.append(step_reward)

                # Track metrics
                all_epsilons.append(current_epsilon)
                all_alphas.append(self.reward_manager.get_current_alpha())
                all_sizes.append(float(self.memory.size))
                all_steps.append(float(step))

                # Periodic structured logging to CSV and console
                if (step + 1) % log_interval == 0 or step == n_steps - 1:
                    mean_r = float(np.mean(recent_rewards[-log_interval:]) if recent_rewards else 0.0)
                    current_alpha = self.reward_manager.get_current_alpha()

                    if csv_writer:
                        csv_writer.writerow([
                            step + 1,
                            self.memory.size,
                            f"{mean_r:.4f}",
                            f"{current_alpha:.4f}",
                            f"{recent_loss:.4f}",
                            f"{current_epsilon:.4f}",
                        ])
                        csv_file.flush()

                    elapsed = time.time() - start_time
                    fps = (step + 1) / max(0.001, elapsed)
                    logger.info(
                        f"Step {step + 1:>6,}/{n_steps:,} | "
                        f"Size: {self.memory.size:>4}/{self.capacity} | "
                        f"R_mean: {mean_r:>+6.3f} | "
                        f"Alpha: {current_alpha:>5.3f} | "
                        f"Loss: {recent_loss:>6.4f} | "
                        f"Eps: {current_epsilon:>5.3f} | "
                        f"Speed: {fps:.0f} steps/s"
                    )

        if csv_file:
            csv_file.close()

        # Save trained RL agent checkpoint
        if save_agent_path:
            self.rl_agent.save(save_agent_path)

        total_elapsed = time.time() - start_time
        logger.info("=" * 70)
        logger.info(f"SIMULATION COMPLETE: {n_steps:,} steps processed in {total_elapsed:.2f}s.")
        logger.info(f"Final Memory Size: {self.memory.size}/{self.capacity}")
        logger.info(f"Final Epsilon: {all_epsilons[-1]:.4f} (started at {all_epsilons[0]:.4f})")
        logger.info(f"Final Alpha: {all_alphas[-1]:.4f}")
        logger.info(f"Total Transitions Stored in PER: {self.rl_agent.replay_buffer.size}")
        logger.info("=" * 70)

        return {
            "rewards": all_rewards,
            "losses": all_losses,
            "alphas": all_alphas,
            "epsilons": all_epsilons,
            "sizes": all_sizes,
            "steps": all_steps,
            "actions": [float(a) for a in all_actions],
        }


# =============================================================================
# 5. PUBLIC INTERFACE CONTRACT
# =============================================================================

def run_simulation(
    n_episodes: int = 1,
    noise_injection_rate: float = SIMULATION_NOISE_RATE,
    capacity: int = MEMORY_CAPACITY,
    n_steps: Optional[int] = None,
    output_csv_path: Optional[str] = SIMULATION_CSV_PATH,
) -> Dict[str, List[float]]:
    """
    Public interface function required by Phase 3 contract in Plano_de_Acao_EAAI.md.

    Args:
        n_episodes: Number of simulation episodes.
        noise_injection_rate: Fraction of samples subjected to noise perturbation.
        capacity: Maximum episodic memory capacity (default: 5000).
        n_steps: Total steps to execute (default: SIMULATION_STEPS = 50000).
        output_csv_path: Optional CSV output path for structured logging.

    Returns:
        Dictionary mapping metric names ("rewards", "losses", etc.) to lists of float values.
    """
    steps_to_run = SIMULATION_STEPS if n_steps is None else n_steps

    sim = TrainRLOnlineSimulation(
        capacity=capacity,
        k_neighbors=KNN_K_NEIGHBORS,
        latent_dim=LATENT_DIM,
        buffer_size=SLIDING_VALIDATION_BUFFER_SIZE,
        alpha_decay=CURRICULUM_ALPHA_DECAY,
    )

    return sim.execute_simulation(
        n_episodes=n_episodes,
        n_steps=steps_to_run,
        noise_injection_rate=noise_injection_rate,
        output_csv_path=output_csv_path,
    )


# =============================================================================
# 6. COMMAND-LINE ENTRY POINT
# =============================================================================

def main() -> None:
    """Command-line entry point for standalone execution."""
    parser = argparse.ArgumentParser(description="Online RL Simulation under Non-Stationary Environments (Phase 3).")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes (default: 1).")
    parser.add_argument("--steps", type=int, default=SIMULATION_STEPS, help="Number of streaming steps (default: 50000).")
    parser.add_argument("--capacity", type=int, default=MEMORY_CAPACITY, help="Memory capacity (default: 5000).")
    parser.add_argument("--noise-rate", type=float, default=SIMULATION_NOISE_RATE, help="Noise injection rate (default: 0.1).")
    parser.add_argument("--noise-level", type=float, default=SIMULATION_NOISE_LEVEL, help="Noise intensity (default: 0.6).")
    parser.add_argument("--log-interval", type=int, default=SIMULATION_LOG_INTERVAL, help="Logging interval (default: 500).")
    parser.add_argument("--output-csv", type=str, default=SIMULATION_CSV_PATH, help="Path for CSV logging output.")
    parser.add_argument("--save-agent", type=str, default=RL_AGENT_CHECKPOINT_PATH, help="Path to save trained RL agent.")

    args = parser.parse_args()

    sim = TrainRLOnlineSimulation(capacity=args.capacity)
    sim.execute_simulation(
        n_episodes=args.episodes,
        n_steps=args.steps,
        noise_injection_rate=args.noise_rate,
        noise_level=args.noise_level,
        log_interval=args.log_interval,
        output_csv_path=args.output_csv,
        save_agent_path=args.save_agent,
    )


if __name__ == "__main__":
    main()
