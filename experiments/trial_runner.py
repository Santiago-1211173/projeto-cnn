"""
Automated AutoML Trial Runner (CIFAR-100 Case Study).

Executes an isolated end-to-end trial for a specific hyperparameter configuration:
1. Configures CNN backbone (ResNet-14) with latent_dim.
2. Trains backbone (AdamW, label smoothing, cutout) for fast_screening or full convergence.
3. Calibrates Dual Uncertainty Arbiter (Ledoit-Wolf shrinkage, Shannon entropy).
4. Seeds episodic memory bank (KNNBanditAgent128D) with class-balanced prototypes.
5. Runs streaming prequential RL simulation (Double DQN + PER) under concept drift.
6. Computes the 5 canonical metrics via TrialEvaluator.
7. Persists trial artifacts (config.json, metrics.json, model and agent checkpoints).
"""

from __future__ import annotations
import os
import sys
import json
import time
import logging
from typing import Dict, Any, Tuple, Optional, List
import numpy as np

# Ensure project root in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.trial_evaluator import TrialEvaluator

logger = logging.getLogger(__name__)

# Module-level references loaded dynamically to avoid circular or environment import errors
tf = None
torch = None
load_cifar100_raw = None
RawModelCIFAR100 = None
Adam = None
DualUncertaintyArbiter = None
KNNBanditAgent128D = None
RLAgent = None
RewardManager = None
CIFAR100_DATA_DIR = None
RANDOM_SEED = 42


def _ensure_dependencies() -> None:
    """Verifies and loads TensorFlow, PyTorch, and project model dependencies."""
    global tf, torch, load_cifar100_raw, RawModelCIFAR100, Adam
    global DualUncertaintyArbiter, KNNBanditAgent128D, RLAgent, RewardManager
    global CIFAR100_DATA_DIR, RANDOM_SEED

    if RawModelCIFAR100 is None:
        from src.config import CIFAR100_DATA_DIR as c_dir, RANDOM_SEED as r_seed
        from src.data.cifar100_loader import load_cifar100_raw as l_cifar
        from src.cifar100.model import RawModelCIFAR100 as m_cifar
        from src.cifar100.optimizers import Adam as opt_adam
        from src.cifar100.ood_arbiter import DualUncertaintyArbiter as arb_cls
        from src.models.knn_bandit_agent import KNNBanditAgent128D as knn_cls
        from src.models.rl_agent import RLAgent as rl_cls
        from src.models.reward_manager import RewardManager as rew_cls

        CIFAR100_DATA_DIR = c_dir
        RANDOM_SEED = r_seed
        load_cifar100_raw = l_cifar
        RawModelCIFAR100 = m_cifar
        Adam = opt_adam
        DualUncertaintyArbiter = arb_cls
        KNNBanditAgent128D = knn_cls
        RLAgent = rl_cls
        RewardManager = rew_cls

    if tf is None:
        try:
            import tensorflow as tf_runtime  # type: ignore
            tf = tf_runtime
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "TensorFlow is required to run trials. Please execute in the 'tf_l40s' conda environment."
            )

    if torch is None:
        try:
            import torch as torch_runtime  # type: ignore
            torch = torch_runtime
        except ModuleNotFoundError:
            raise ModuleNotFoundError("PyTorch is required to run RL agents.")


class TrialRunner:
    """
    Executes an isolated hyperparameter evaluation trial.
    """

    def __init__(
        self,
        trial_id: str,
        config: Dict[str, Any],
        output_base_dir: str,
    ) -> None:
        """
        Initializes the TrialRunner.

        Args:
            trial_id: Unique identifier for the trial (e.g. 'trial_001').
            config: Hyperparameter dictionary.
            output_base_dir: Base directory where trial artifacts will be stored.
        """
        self.trial_id: str = trial_id
        self.config: Dict[str, Any] = dict(config)
        self.output_base_dir: str = os.path.abspath(output_base_dir)
        self.trial_dir: str = os.path.join(self.output_base_dir, self.trial_id)
        self.checkpoints_dir: str = os.path.join(self.trial_dir, "checkpoints")

        os.makedirs(self.trial_dir, exist_ok=True)
        os.makedirs(self.checkpoints_dir, exist_ok=True)

    def _select_balanced_prototypes(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        num_prototypes: int,
        n_classes: int = 100,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Selects class-balanced prototypes across all n_classes."""
        samples_per_class = max(1, num_prototypes // n_classes)
        selected_indices: List[int] = []

        for c in range(n_classes):
            class_idxs = np.where(labels == c)[0]
            if len(class_idxs) < samples_per_class:
                chosen = class_idxs
            else:
                chosen = class_idxs[:samples_per_class]
            selected_indices.extend(chosen)

        if len(selected_indices) < num_prototypes:
            remaining = np.setdiff1d(np.arange(len(labels)), selected_indices)
            needed = num_prototypes - len(selected_indices)
            selected_indices.extend(remaining[:needed])

        selected_indices = np.array(selected_indices[:num_prototypes], dtype=np.int32)
        return images[selected_indices], labels[selected_indices]

    def _train_cnn(
        self,
        model: Any,
        x_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int,
        lr: float,
        latent_dim: int,
        batch_size: int = 128,
        label_smoothing: float = 0.1,
    ) -> None:
        """Trains ResNet-14 backbone for specified number of epochs."""
        global tf, Adam
        _ensure_dependencies()

        optimizer = Adam(learning_rate=lr, weight_decay=1e-4)
        ls_eps = float(label_smoothing)

        @tf.function
        def train_step(x_b: tf.Tensor, y_b: tf.Tensor) -> tf.Tensor:
            with tf.GradientTape() as tape:
                outputs = model(x_b, training=True)
                probs = outputs["probabilities"]

                y_one_hot = tf.one_hot(y_b, depth=100)
                y_smooth = y_one_hot * (1.0 - ls_eps) + (ls_eps / 100.0)

                probs_clipped = tf.clip_by_value(probs, 1e-12, 1.0)
                loss = -tf.reduce_sum(y_smooth * tf.math.log(probs_clipped)) / tf.cast(
                    tf.shape(x_b)[0], tf.float32
                )

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(model.trainable_variables, grads)
            return loss

        # Fast training dataset pipeline
        dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        dataset = dataset.shuffle(5000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

        for epoch in range(epochs):
            total_loss = 0.0
            num_batches = 0
            for bx, by in dataset:
                loss_val = train_step(bx, by)
                total_loss += float(loss_val)
                num_batches += 1

            avg_loss = total_loss / max(1, num_batches)
            logger.info(
                f"[{self.trial_id}] CNN Epoch {epoch + 1:02d}/{epochs:02d} - Loss: {avg_loss:.4f}"
            )

    def _simulate_streaming_rl(
        self,
        model: Any,
        arbiter: Any,
        memory: Any,
        rl_agent: Any,
        reward_manager: Any,
        x_stream: np.ndarray,
        y_stream: np.ndarray,
        steps: int,
        latent_dim: int,
        batch_size: int = 32,
    ) -> None:
        """Executes streaming concept drift simulation with RL eviction."""
        global tf
        n_samples = len(x_stream)
        stream_ptr = 0

        for step in range(steps):
            idx = stream_ptr % n_samples
            stream_ptr += 1

            img = x_stream[idx : idx + 1]
            y_true = int(y_stream[idx])

            # Injected noise 15% of time
            if np.random.rand() < 0.15:
                noise = np.random.normal(loc=0.0, scale=0.6, size=img.shape)
                img = np.clip(img + noise, 0.0, 1.0).astype(np.float32)

            tensor_img = tf.convert_to_tensor(img, dtype=tf.float32)
            out = model(tensor_img, training=False)
            z = out["latent_features"].numpy().astype(np.float32)[0]
            probs = out["probabilities"].numpy().astype(np.float32)
            p_cnn = int(np.argmax(probs[0]))

            arb_res = arbiter.predict(z.reshape(1, -1), probs)
            is_ood = bool(arb_res["is_ood"][0])
            d_M = float(arb_res["mahalanobis_dist"][0])
            ent = float(arb_res["entropy"][0])

            # Episodic decision
            if is_ood and memory.size > 0:
                y_pred = memory.get_action(z)
                idxs, _ = memory.get_nearest_neighbors(z, k=memory.k)
                memory.increment_usage(idxs)
            else:
                y_pred = p_cnn

            # If OOD or classification error, RL eviction policy handles experience
            if is_ood or (p_cnn != y_true):
                if memory.size > 0:
                    diff = memory._states[: memory.size] - z
                    min_knn_dist = float(np.sqrt(np.min(np.sum(diff * diff, axis=1))))
                else:
                    min_knn_dist = 0.0

                ram_occ = float(memory.size / max(1, memory.capacity))
                state_vec = rl_agent.get_state_vector(
                    mahalanobis_dist=d_M,
                    local_entropy=ent,
                    min_knn_dist=min_knn_dist,
                    prediction_error=float(p_cnn != y_true),
                    ram_occupancy=ram_occ,
                )

                action = rl_agent.select_action(state_vec, epsilon=0.10)

                # Execute Action
                if action == 0:
                    # Filter outlier
                    pass
                elif action == 1:
                    memory.evict_oldest(z, p_cnn, 1.0)
                elif action == 2:
                    memory.evict_least_frequently_used(z, p_cnn, 1.0)
                elif action == 3:
                    memory.evict_most_redundant(z, p_cnn, 1.0)

                step_reward = 1.0 if (is_ood and action == 0) or (not is_ood and action != 0) else -1.0
                next_ram_occ = float(memory.size / max(1, memory.capacity))
                next_state_vec = rl_agent.get_state_vector(
                    mahalanobis_dist=d_M,
                    local_entropy=ent,
                    min_knn_dist=min_knn_dist,
                    prediction_error=float(p_cnn != y_true),
                    ram_occupancy=next_ram_occ,
                )
                rl_agent.store_transition(
                    state=state_vec,
                    action=action,
                    reward=step_reward,
                    next_state=next_state_vec,
                    done=False,
                )

                # Periodically update RL agent
                if (step + 1) % 4 == 0 and len(rl_agent.replay_buffer) >= batch_size:
                    rl_agent.update(batch_size=batch_size)

    def run(self, fast_screening: bool = True) -> Dict[str, Any]:
        """
        Executes the trial and records the multi-metric performance.

        Args:
            fast_screening: If True, executes 5 CNN epochs and 2,000 RL steps.
                            If False, executes 50 CNN epochs and 50,000 RL steps.

        Returns:
            Dict containing trial_id, config, metrics, and artifacts_path.
        """
        _ensure_dependencies()

        # Hyperparameter extraction
        latent_dim = int(self.config.get("latent_dim", 128))
        mahalanobis_threshold = float(self.config.get("mahalanobis_threshold", 15.0))
        entropy_threshold = float(self.config.get("entropy_threshold", 2.0))
        curriculum_alpha_decay = float(self.config.get("curriculum_alpha_decay", 0.995))
        memory_capacity = int(self.config.get("memory_capacity", 5000))
        rl_lr = float(self.config.get("rl_lr", 1e-3))
        rl_gamma = float(self.config.get("rl_gamma", 0.99))
        knn_k = int(self.config.get("knn_k", 10))
        min_alpha = float(self.config.get("min_alpha", 0.01))

        epochs = int(self.config.get("epochs", 5 if fast_screening else 50))
        sim_steps = int(self.config.get("sim_steps", 2000 if fast_screening else 50000))
        seed = int(self.config.get("seed", RANDOM_SEED))

        # Seed determinism
        np.random.seed(seed)
        tf.random.set_seed(seed)
        if torch is not None:
            torch.manual_seed(seed)

        logger.info("=" * 70)
        logger.info(f"STARTING AUTOML TRIAL: {self.trial_id}")
        logger.info(
            f"Latent Dim: {latent_dim} | Cap: {memory_capacity} | k: {knn_k} | "
            f"tau_M: {mahalanobis_threshold} | tau_H: {entropy_threshold} | Fast: {fast_screening}"
        )
        logger.info("=" * 70)

        # 1. Load Data
        x_train, y_train, x_test, y_test = load_cifar100_raw(CIFAR100_DATA_DIR)
        x_train = x_train.astype(np.float32)
        y_train = y_train.astype(np.int32)
        x_test = x_test.astype(np.float32)
        y_test = y_test.astype(np.int32)

        # Allow training subset for ultra-fast screening if specified
        max_train = self.config.get("train_samples", None)
        if max_train is not None and max_train < len(x_train):
            x_tr, y_tr = self._select_balanced_prototypes(x_train, y_train, max_train, 100)
        else:
            x_tr, y_tr = x_train, y_train

        # 2. ResNet-14 Backbone Training
        model = RawModelCIFAR100(latent_dim=latent_dim)
        if epochs > 0:
            self._train_cnn(
                model=model,
                x_train=x_tr,
                y_train=y_tr,
                epochs=epochs,
                lr=float(self.config.get("lr", 0.001)),
                latent_dim=latent_dim,
                batch_size=int(self.config.get("batch_size", 128)),
            )

        # Save CNN checkpoint
        ckpt = tf.train.Checkpoint(model=model)
        model_ckpt_path = os.path.join(self.checkpoints_dir, "modelo_dissecado")
        ckpt.save(model_ckpt_path)

        # 3. Dual Uncertainty Arbiter Calibration
        num_protos = min(memory_capacity, len(x_tr))
        proto_x, proto_y = self._select_balanced_prototypes(
            x_tr, y_tr, num_prototypes=num_protos, n_classes=100
        )
        proto_feats, proto_preds, proto_probs = TrialEvaluator._extract_cnn_outputs(
            model, proto_x, batch_size=250
        )

        arbiter = DualUncertaintyArbiter(n_classes=100, latent_dim=latent_dim)
        arbiter.fit(proto_feats, proto_probs, proto_y, percentile=95.0)

        # Apply specific thresholds from hyperparameter config
        arbiter.threshold_mahalanobis = mahalanobis_threshold
        arbiter.threshold_entropy = entropy_threshold

        arbiter_path = os.path.join(self.trial_dir, "arbiter_profiles.npz")
        arbiter.save(arbiter_path)

        # 4. Episodic Memory Bank Seeding
        memory = KNNBanditAgent128D(
            capacity=memory_capacity,
            k=knn_k,
            latent_dim=latent_dim,
            n_actions=100,
        )
        memory.add_experience_batch(
            states=proto_feats,
            actions=proto_y,
            rewards=np.ones(len(proto_y), dtype=np.float32),
        )
        memory_path = os.path.join(self.trial_dir, "knn_memory_bank.npz")
        memory.save(memory_path)

        # 5. Streaming RL Simulation
        rl_agent = RLAgent(
            state_dim=5,
            n_actions=4,
            lr=rl_lr,
            gamma=rl_gamma,
            target_update_interval=100,
            buffer_capacity=10000,
            device="cpu",
        )
        reward_manager = RewardManager(
            buffer_size=100,
            alpha_decay=curriculum_alpha_decay,
            min_alpha=min_alpha,
            latent_dim=latent_dim,
        )

        if sim_steps > 0:
            self._simulate_streaming_rl(
                model=model,
                arbiter=arbiter,
                memory=memory,
                rl_agent=rl_agent,
                reward_manager=reward_manager,
                x_stream=x_train,
                y_stream=y_train,
                steps=sim_steps,
                latent_dim=latent_dim,
            )

        rl_agent_path = os.path.join(self.checkpoints_dir, "rl_agent.pt")
        rl_agent.save(rl_agent_path)

        # 6. Evaluation across 5 canonical metrics
        eval_samples = self.config.get("eval_samples", 1000)
        metrics = TrialEvaluator.evaluate(
            model=model,
            arbiter=arbiter,
            memory_bank=memory,
            rl_agent=rl_agent,
            test_data=(x_test, y_test),
            max_eval_samples=eval_samples,
            noise_sigma=0.6,
        )

        # 7. Persist Artifacts
        config_path = os.path.join(self.trial_dir, "config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(self.config, f, indent=2)

        metrics_path = os.path.join(self.trial_dir, "metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"[{self.trial_id}] Completed. Metrics: {metrics}")

        return {
            "trial_id": self.trial_id,
            "config": self.config,
            "metrics": metrics,
            "artifacts_path": self.trial_dir,
        }
