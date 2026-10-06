"""
Multi-Metric Evaluator for AutoML Trials (CIFAR-100 Case Study).

Computes the 5 canonical evaluation metrics required by the Master Execution Plan:
1. clean_accuracy: Top-1 accuracy on clean test set [0.0, 1.0] (Weight: 0.30, Higher is better)
2. noisy_accuracy: Top-1 accuracy under Gaussian noise sigma=0.6 [0.0, 1.0] (Weight: 0.25, Higher is better)
3. d_kl_eviction: Kullback-Leibler divergence between memory buffer class distribution and uniform 1/100 (Weight: 0.20, Lower is better)
4. inference_latency_ms: End-to-end inference latency per query in milliseconds (Weight: 0.15, Lower is better)
5. ram_footprint_mb: Theoretical pre-allocated memory footprint in MB (Weight: 0.10, Lower is better)
"""

from __future__ import annotations
import time
import logging
from typing import Tuple, Dict, Any, Optional, List
import numpy as np

try:
    import tensorflow as tf
except ModuleNotFoundError:
    tf = None

logger = logging.getLogger(__name__)


class TrialEvaluator:
    """
    Evaluates deep semiparametric edge AI systems across the 5 canonical objective metrics.
    """

    @staticmethod
    def compute_ram_footprint(capacity: int, latent_dim: int, n_classes: int = 100) -> float:
        """
        Computes the theoretical pre-allocated RAM footprint in MB according to
        the formal specification:

            RAM = [ capacity * (latent_dim * 4 + 4 + 4 + 8)
                  + n_classes * (latent_dim * 4 + latent_dim^2 * 4) ] / (1024^2)

        Args:
            capacity: Total episodic memory capacity (experiences).
            latent_dim: Latent representation dimensionality.
            n_classes: Number of classification categories (default: 100).

        Returns:
            RAM footprint in Megabytes (float).
        """
        # Memory buffer experiences: float32 features + int32 action + float32 reward + int64 timestamp
        exp_bytes = capacity * (latent_dim * 4 + 4 + 4 + 8)
        # Arbiter profiles: mean vector (latent_dim * 4) + covariance/precision (latent_dim^2 * 4) per class
        arbiter_bytes = n_classes * (latent_dim * 4 + (latent_dim ** 2) * 4)
        total_mb = (exp_bytes + arbiter_bytes) / (1024.0 * 1024.0)
        return float(total_mb)

    @staticmethod
    def compute_kl_divergence(
        memory_bank: Any,
        n_classes: int = 100,
        eps: float = 1e-12,
    ) -> float:
        """
        Computes the Kullback-Leibler divergence D_KL(P || U) between the
        empirical class distribution of experiences currently stored in the
        episodic buffer and the ideal uniform distribution U(1/n_classes):

            D_KL(P || U) = sum_{c=0}^{n_classes-1} P(c) * ln( (P(c) + eps) / (1/n_classes) )

        Returns:
            D_KL in nats (float). Lower is better.
        """
        if memory_bank is None or getattr(memory_bank, "size", 0) == 0:
            return 0.0

        size = memory_bank.size
        actions = memory_bank._actions[:size]
        counts = np.bincount(actions, minlength=n_classes)[:n_classes]
        total = np.sum(counts)

        if total == 0:
            return 0.0

        p = counts.astype(np.float64) / float(total)
        u = 1.0 / float(n_classes)

        # D_KL formula
        kl_div = np.sum(p * np.log((p + eps) / u))
        return float(max(0.0, kl_div))

    @staticmethod
    def _extract_cnn_outputs(
        model: Any,
        images: np.ndarray,
        batch_size: int = 250,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extracts latent bottleneck features, class predictions, and probabilities in batches.
        """
        global tf
        if tf is None:
            import tensorflow as tf_runtime  # type: ignore
            tf = tf_runtime

        n_samples = len(images)
        feats_list: List[np.ndarray] = []
        probs_list: List[np.ndarray] = []
        preds_list: List[np.ndarray] = []

        for i in range(0, n_samples, batch_size):
            chunk = images[i : i + batch_size]
            tensor_x = tf.convert_to_tensor(chunk, dtype=tf.float32)
            out = model(tensor_x, training=False)
            z = out["latent_features"].numpy().astype(np.float32)
            p = out["probabilities"].numpy().astype(np.float32)
            y_hat = np.argmax(p, axis=1).astype(np.int32)

            feats_list.append(z)
            probs_list.append(p)
            preds_list.append(y_hat)

        all_feats = np.vstack(feats_list)
        all_probs = np.vstack(probs_list)
        all_preds = np.concatenate(preds_list)
        return all_feats, all_preds, all_probs

    @staticmethod
    def _predict_hybrid_system(
        model: Any,
        arbiter: Any,
        memory_bank: Any,
        images: np.ndarray,
        batch_size: int = 250,
    ) -> np.ndarray:
        """
        Performs inference across the hybrid CNN + Arbiter + Episodic Memory pipeline.
        """
        feats, cnn_preds, probs = TrialEvaluator._extract_cnn_outputs(
            model, images, batch_size=batch_size
        )
        n_samples = len(images)
        final_preds = np.empty(n_samples, dtype=np.int32)

        # Run arbiter OOD prediction
        if arbiter is not None and hasattr(arbiter, "predict"):
            arbiter_res = arbiter.predict(feats, probs)
            is_ood_flags = arbiter_res["is_ood"]
        else:
            is_ood_flags = np.zeros(n_samples, dtype=bool)

        has_memory = (memory_bank is not None) and (getattr(memory_bank, "size", 0) > 0)

        for i in range(n_samples):
            z = feats[i]
            p_cnn = cnn_preds[i]
            is_ood = is_ood_flags[i]

            if is_ood and has_memory:
                y_pred = memory_bank.get_action(z)
            else:
                y_pred = p_cnn

            final_preds[i] = y_pred

        return final_preds

    @staticmethod
    def evaluate(
        model: Any,
        arbiter: Any,
        memory_bank: Any,
        rl_agent: Any,
        test_data: Tuple[np.ndarray, np.ndarray],
        max_eval_samples: Optional[int] = 1000,
        noise_sigma: float = 0.6,
        batch_size: int = 250,
        latency_benchmark_queries: int = 100,
    ) -> Dict[str, float]:
        """
        Executes comprehensive multi-metric evaluation for an AutoML trial.

        Args:
            model: Restored RawModelCIFAR100 backbone instance.
            arbiter: Calibrated DualUncertaintyArbiter instance.
            memory_bank: Seeded/trained KNNBanditAgent128D memory instance.
            rl_agent: Trained RLAgent instance (Double DQN + PER).
            test_data: Tuple of (x_test, y_test) arrays.
            max_eval_samples: Maximum test samples to evaluate for accuracy (default: 1000).
            noise_sigma: Standard deviation of Gaussian noise for noisy accuracy (default: 0.6).
            batch_size: Chunk size for batched forward passes (default: 250).
            latency_benchmark_queries: Number of sequential queries for latency benchmark.

        Returns:
            Dict containing the 5 canonical metrics:
                - clean_accuracy: float in [0.0, 1.0]
                - noisy_accuracy: float in [0.0, 1.0]
                - d_kl_eviction: float >= 0.0
                - inference_latency_ms: float > 0.0
                - ram_footprint_mb: float > 0.0
        """
        x_all, y_all = test_data

        if max_eval_samples is not None and max_eval_samples > 0:
            eval_size = min(max_eval_samples, len(x_all))
            x_eval = x_all[:eval_size].astype(np.float32)
            y_eval = y_all[:eval_size].astype(np.int32)
        else:
            x_eval = x_all.astype(np.float32)
            y_eval = y_all.astype(np.int32)

        # 1. Clean Accuracy [0.0, 1.0]
        clean_preds = TrialEvaluator._predict_hybrid_system(
            model=model,
            arbiter=arbiter,
            memory_bank=memory_bank,
            images=x_eval,
            batch_size=batch_size,
        )
        clean_accuracy = float(np.mean(clean_preds == y_eval))

        # 2. Noisy Accuracy [0.0, 1.0] with Gaussian noise sigma=0.6
        noise = np.random.normal(loc=0.0, scale=noise_sigma, size=x_eval.shape)
        x_noisy = np.clip(x_eval + noise, 0.0, 1.0).astype(np.float32)
        noisy_preds = TrialEvaluator._predict_hybrid_system(
            model=model,
            arbiter=arbiter,
            memory_bank=memory_bank,
            images=x_noisy,
            batch_size=batch_size,
        )
        noisy_accuracy = float(np.mean(noisy_preds == y_eval))

        # 3. D_KL Eviction Distribution Matching
        d_kl_eviction = TrialEvaluator.compute_kl_divergence(
            memory_bank=memory_bank,
            n_classes=100,
        )

        # 4. End-to-End Inference Latency (ms)
        n_lat = min(latency_benchmark_queries, len(x_eval))
        x_lat = x_eval[:n_lat]

        # Warmup single query
        if n_lat > 0:
            _ = TrialEvaluator._predict_hybrid_system(
                model=model,
                arbiter=arbiter,
                memory_bank=memory_bank,
                images=x_lat[:1],
                batch_size=1,
            )

        t_start = time.perf_counter()
        for idx in range(n_lat):
            sample_img = x_lat[idx : idx + 1]
            # CNN forward pass
            feats, p_cnn, probs = TrialEvaluator._extract_cnn_outputs(
                model, sample_img, batch_size=1
            )
            z = feats[0]
            # Arbiter uncertainty check
            if arbiter is not None:
                arb_out = arbiter.predict(feats, probs)
                is_ood = arb_out["is_ood"][0]
                d_M = float(arb_out["mahalanobis_dist"][0])
                ent = float(arb_out["entropy"][0])
            else:
                is_ood = False
                d_M = 0.0
                ent = 0.0

            # Memory query
            if is_ood and memory_bank is not None and memory_bank.size > 0:
                _ = memory_bank.get_action(z)

            # RL Agent decision simulation
            if rl_agent is not None and memory_bank is not None and memory_bank.size > 0:
                diff = memory_bank._states[: memory_bank.size] - z
                min_knn = float(np.sqrt(np.min(np.sum(diff * diff, axis=1))))
                ram_occ = float(memory_bank.size / memory_bank.capacity)
                state_vec = rl_agent.get_state_vector(
                    mahalanobis_dist=d_M,
                    local_entropy=ent,
                    min_knn_dist=min_knn,
                    prediction_error=0.0,
                    ram_occupancy=ram_occ,
                )
                _ = rl_agent.select_action(state_vec, epsilon=0.0)

        t_end = time.perf_counter()
        inference_latency_ms = float(((t_end - t_start) * 1000.0) / max(1, n_lat))

        # 5. Theoretical RAM Footprint (MB)
        capacity = getattr(memory_bank, "capacity", 5000)
        latent_dim = getattr(memory_bank, "latent_dim", 128)
        ram_footprint_mb = TrialEvaluator.compute_ram_footprint(
            capacity=capacity,
            latent_dim=latent_dim,
            n_classes=100,
        )

        metrics: Dict[str, float] = {
            "clean_accuracy": round(clean_accuracy, 4),
            "noisy_accuracy": round(noisy_accuracy, 4),
            "d_kl_eviction": round(d_kl_eviction, 4),
            "inference_latency_ms": round(inference_latency_ms, 3),
            "ram_footprint_mb": round(ram_footprint_mb, 2),
        }
        return metrics
