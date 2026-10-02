"""
Dual Uncertainty OOD Arbiter for CIFAR-10 (Kaur et al., 2021; Nguyen, 2026).
Combines unnormalized Mahalanobis distance with predictive Shannon entropy.
Preserves class-conditional centroids and Ledoit-Wolf shrinkage precision matrices.
"""
import os
from typing import Tuple, Dict, Any, Optional
import numpy as np
from sklearn.covariance import LedoitWolf


class DualUncertaintyArbiter:
    """
    Dual Uncertainty Out-of-Distribution Arbiter.
    
    Combines:
    1. Unnormalized Mahalanobis distance in the 128D latent bottleneck space
       using class-conditional Ledoit-Wolf empirical shrinkage precision matrices.
    2. Predictive Shannon entropy over softmax probabilities.
    
    Rejection condition:
        is_ood = (d_M > threshold_mahalanobis) or (entropy > threshold_entropy)
    """
    def __init__(self, n_classes: int = 10, latent_dim: int = 128):
        self.n_classes = n_classes
        self.latent_dim = latent_dim
        self.profiles: Dict[int, Dict[str, np.ndarray]] = {}
        self.threshold_mahalanobis = 15.0
        self.threshold_entropy = 1.20
        self.is_fitted = False

    @property
    def threshold(self) -> float:
        """Alias for backward compatibility with single-threshold interfaces."""
        return self.threshold_mahalanobis

    def fit(self, latent_features: np.ndarray, probabilities: np.ndarray, labels: np.ndarray, percentile: float = 95.0) -> None:
        """
        Fits class-conditional centroids and Ledoit-Wolf precision on unnormalized features.
        Calibrates dual uncertainty thresholds using in-distribution percentiles.
        
        Args:
            latent_features: Array of shape (N, latent_dim).
            probabilities: Array of shape (N, n_classes).
            labels: Array of shape (N,) with integer class labels.
            percentile: In-distribution calibration percentile (default 95.0).
        """
        latent_features = np.asarray(latent_features, dtype=np.float32)
        probabilities = np.asarray(probabilities, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.int32).reshape(-1)

        self.profiles.clear()
        for c in range(self.n_classes):
            idx = np.where(labels == c)[0]
            if len(idx) == 0:
                raise ValueError(f"No samples found for class {c} during Arbiter fitting.")
            class_feats = latent_features[idx]
            lw = LedoitWolf().fit(class_feats)
            self.profiles[c] = {
                "mu": lw.location_.astype(np.float32),
                "precision": lw.precision_.astype(np.float32)
            }
        self.is_fitted = True

        # Calibrate thresholds at specified percentile of clean in-distribution data
        dists = self.compute_mahalanobis_batch(latent_features)
        entropies = self.compute_entropy_batch(probabilities)
        self.threshold_mahalanobis = float(np.percentile(dists, percentile))
        self.threshold_entropy = float(np.percentile(entropies, percentile))

    def compute_entropy_batch(self, probabilities: np.ndarray) -> np.ndarray:
        """
        Computes predictive Shannon entropy for a batch of probability vectors.
        
        Args:
            probabilities: Array of shape (N, n_classes).
            
        Returns:
            Array of shape (N,) with non-negative Shannon entropies in nats.
        """
        p = np.clip(np.asarray(probabilities, dtype=np.float32), 1e-12, 1.0)
        return -np.sum(p * np.log(p), axis=-1)

    def compute_entropy(self, probability: np.ndarray) -> float:
        """Computes predictive Shannon entropy for a single sample."""
        p = np.asarray(probability, dtype=np.float32).reshape(1, -1)
        return float(self.compute_entropy_batch(p)[0])

    def compute_mahalanobis_batch(self, latent_features: np.ndarray) -> np.ndarray:
        """
        Vectorized computation of minimum Mahalanobis distance across all class profiles.
        
        Args:
            latent_features: Array of shape (N, latent_dim).
            
        Returns:
            Array of shape (N,) containing minimum Mahalanobis distance per sample.
        """
        if not self.is_fitted:
            raise RuntimeError("DualUncertaintyArbiter must be fitted or loaded before computing distances.")

        feats = np.asarray(latent_features, dtype=np.float32)
        if feats.ndim == 1:
            feats = feats.reshape(1, -1)
        n_samples = len(feats)
        all_dists = np.zeros((n_samples, self.n_classes), dtype=np.float32)
        for c in range(self.n_classes):
            diff = feats - self.profiles[c]["mu"]
            sq_dists = np.sum((diff @ self.profiles[c]["precision"]) * diff, axis=1)
            all_dists[:, c] = np.sqrt(np.maximum(0.0, sq_dists))
        return np.min(all_dists, axis=1)

    def compute_distances_batch(self, latent_features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Vectorized computation of minimum distances and closest class indices.
        
        Returns:
            Tuple of (min_distances, closest_classes).
        """
        if not self.is_fitted:
            raise RuntimeError("DualUncertaintyArbiter must be fitted or loaded before computing distances.")

        feats = np.asarray(latent_features, dtype=np.float32)
        if feats.ndim == 1:
            feats = feats.reshape(1, -1)
        n_samples = len(feats)
        all_dists = np.zeros((n_samples, self.n_classes), dtype=np.float32)
        for c in range(self.n_classes):
            diff = feats - self.profiles[c]["mu"]
            sq_dists = np.sum((diff @ self.profiles[c]["precision"]) * diff, axis=1)
            all_dists[:, c] = np.sqrt(np.maximum(0.0, sq_dists))
        min_dists = np.min(all_dists, axis=1)
        closest_classes = np.argmin(all_dists, axis=1)
        return min_dists, closest_classes

    def compute_distance(self, latent_feature: np.ndarray, target_class: Optional[int] = None) -> float:
        """
        Computes Mahalanobis distance for a single latent feature vector.
        
        Args:
            latent_feature: Vector of shape (latent_dim,) or (1, latent_dim).
            target_class: Optional specific class centroid to evaluate against.
        """
        if not self.is_fitted:
            raise RuntimeError("DualUncertaintyArbiter must be fitted or loaded before computing distances.")

        z = np.asarray(latent_feature, dtype=np.float32).reshape(1, -1)
        if target_class is not None:
            if target_class not in self.profiles:
                return float("inf")
            p = self.profiles[target_class]
            diff = z - p["mu"]
            sq_dist = float(np.sum((diff @ p["precision"]) * diff))
            return float(np.sqrt(max(0.0, sq_dist)))
        return float(self.compute_mahalanobis_batch(z)[0])

    def is_out_of_distribution(self, latent_feature: np.ndarray, probability: np.ndarray) -> Tuple[bool, float, float]:
        """
        Evaluates a single sample against calibrated dual uncertainty thresholds.
        
        Args:
            latent_feature: Vector of shape (latent_dim,) or (1, latent_dim).
            probability: Vector of shape (n_classes,) or (1, n_classes).
            
        Returns:
            Tuple of (is_ood: bool, d_mahalanobis: float, shannon_entropy: float)
        """
        z = np.asarray(latent_feature, dtype=np.float32).reshape(1, -1)
        pr = np.asarray(probability, dtype=np.float32).reshape(1, -1)
        d_m = float(self.compute_mahalanobis_batch(z)[0])
        ent = float(self.compute_entropy_batch(pr)[0])
        is_ood = bool((d_m > self.threshold_mahalanobis) or (ent > self.threshold_entropy))
        return is_ood, d_m, ent

    def is_ood(self, latent_feature: np.ndarray, probability: Optional[np.ndarray] = None) -> bool:
        """
        Convenience method returning boolean OOD verdict.
        If probability is not provided, evaluates Mahalanobis distance alone.
        """
        if probability is None:
            return self.compute_distance(latent_feature) > self.threshold_mahalanobis
        is_ood_flag, _, _ = self.is_out_of_distribution(latent_feature, probability)
        return is_ood_flag

    def save(self, path: str) -> None:
        """Saves calibrated profiles and thresholds to compressed .npz archive."""
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted DualUncertaintyArbiter.")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        save_dict = {
            "n_classes": np.array(self.n_classes, dtype=np.int32),
            "latent_dim": np.array(self.latent_dim, dtype=np.int32),
            "threshold_mahalanobis": np.array(self.threshold_mahalanobis, dtype=np.float32),
            "threshold_entropy": np.array(self.threshold_entropy, dtype=np.float32),
        }
        for c in range(self.n_classes):
            save_dict[f"mu_{c}"] = self.profiles[c]["mu"]
            save_dict[f"precision_{c}"] = self.profiles[c]["precision"]
        np.savez_compressed(path, **save_dict)

    def load(self, path: str) -> None:
        """Loads profiles and calibrated thresholds from .npz archive."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Arbiter profile file not found at: {path}")
        data = np.load(path)
        self.n_classes = int(data["n_classes"])
        self.latent_dim = int(data["latent_dim"])
        self.threshold_mahalanobis = float(data["threshold_mahalanobis"])
        self.threshold_entropy = float(data["threshold_entropy"])
        self.profiles.clear()
        for c in range(self.n_classes):
            self.profiles[c] = {
                "mu": data[f"mu_{c}"].astype(np.float32),
                "precision": data[f"precision_{c}"].astype(np.float32)
            }
        self.is_fitted = True
