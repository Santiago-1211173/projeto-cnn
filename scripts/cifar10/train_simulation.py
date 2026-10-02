"""
Online Streaming RL Simulation under Non-Stationary Concept Drift (CIFAR-10).
Phase 4 of EAAI Action Plan: Active Episodic Memory Management via Double DQN + PER.

Orchestrates streaming prequential evaluation under concept drift and noise stress:
1. Passes streaming CIFAR-10 data through converged ResNet-9 backbone (RawModelCIFAR10).
2. Dual Uncertainty Arbiter (Mahalanobis distance + Shannon entropy) routes representations.
3. Active memory curation on KNNBanditAgent128D (5,000 capacity, 128D, k=10).
4. Double DQN with Prioritized Experience Replay (RLAgent) learns specialized eviction:
   - Action 0: Filter / Ignore destructive noise outliers (Alonso & Krichmar, 2024; Alabed, 2019).
   - Action 1: FIFO eviction for temporal drift.
   - Action 2: LFU eviction for infrequent patterns.
   - Action 3: Redundant eviction for class distribution matching (Isele & Cosgun, AAAI 2018).

Scientific References:
- Isele & Cosgun (AAAI 2018): Selective Experience Replay for Continual Learning.
- Alabed (2019): RLCache - Reinforcement Learning for Adaptive Cache Eviction.
- Alonso & Krichmar (Nature Communications 2024): Self-organizing neural memory.
- Kaur et al. (ICML 2021) & Nguyen (2026): Dual uncertainty fusion for robust OOD detection.
- van Hasselt et al. (2016): Deep Reinforcement Learning with Double Q-learning.
- Schaul et al. (ICLR 2016): Prioritized Experience Replay.
"""

from __future__ import annotations
import os
import sys
import csv
import time
import logging
import argparse
from typing import Tuple, Dict, Any, List, Optional
import numpy as np
import tensorflow as tf

# Ensure project root in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import (
    CIFAR10_DATA_DIR,
    RANDOM_SEED,
    MEMORY_CAPACITY,
    LATENT_DIM,
    SIMULATION_STEPS,
    SIMULATION_LOG_INTERVAL,
)
from src.data.loader import load_dataset_raw
from src.cifar10.model import RawModelCIFAR10
from src.cifar10.ood_arbiter import DualUncertaintyArbiter
from src.models.knn_bandit_agent import KNNBanditAgent128D
from src.models.rl_agent import RLAgent
from src.models.reward_manager import RewardManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def inject_noise_batch(images: np.ndarray, noise_level: float) -> np.ndarray:
    """Injects zero-mean Gaussian noise perturbation clipped strictly to [0.0, 1.0]."""
    if noise_level <= 0.0:
        return images.copy()
    noise = np.random.normal(loc=0.0, scale=noise_level, size=images.shape)
    return np.clip(images + noise, 0.0, 1.0).astype(np.float32)


def extract_features_chunk(
    model: RawModelCIFAR10,
    images: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts 128D latent bottleneck representations, predicted classes, and probabilities.
    """
    batch_tensor = tf.convert_to_tensor(images, dtype=tf.float32)
    out = model(batch_tensor, training=False)
    latent = out["latent_features"].numpy().astype(np.float32)
    probs = out["probabilities"].numpy().astype(np.float32)
    preds = np.argmax(probs, axis=1).astype(np.int32)
    return latent, preds, probs


def run_cifar10_simulation(
    steps: int = SIMULATION_STEPS,
    noise_rate: float = 0.15,
    redundancy_rate: float = 0.15,
    noise_levels: Tuple[float, ...] = (0.2, 0.4, 0.6, 0.8),
    batch_size: int = 32,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    log_interval: int = SIMULATION_LOG_INTERVAL,
    train_frequency: int = 4,
    checkpoint_dir: str = "outputs/cifar10/checkpoints",
    profiles_path: str = "outputs/cifar10/mahalanobis_pp_profiles.npz",
    memory_bank_path: str = "outputs/cifar10/knn_memory_bank_128d.npz",
    output_csv_path: str = "outputs/cifar10/train_rl_simulation_log.csv",
    save_agent_path: str = "outputs/cifar10/checkpoints/rl_agent_phase3.pt",
) -> Dict[str, Any]:
    """
    Executes the streaming online RL simulation under concept drift for CIFAR-10.
    """
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    os.makedirs(os.path.dirname(save_agent_path), exist_ok=True)

    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    # 1. Hardware Configuration
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                pass

    # 2. Load Converged CNN Backbone (Phase 1)
    logger.info("=" * 70)
    logger.info("PHASE 4: ONLINE RL SIMULATION UNDER DRIFT (CIFAR-10)")
    logger.info("=" * 70)
    checkpoint_full_dir = os.path.abspath(checkpoint_dir)
    latest_ckpt = tf.train.latest_checkpoint(checkpoint_full_dir)
    if not latest_ckpt:
        raise FileNotFoundError(f"CNN checkpoint not found in {checkpoint_full_dir}!")

    logger.info(f"Loading converged CIFAR-10 backbone: {latest_ckpt}")
    cnn = RawModelCIFAR10()
    ckpt = tf.train.Checkpoint(model=cnn)
    ckpt.restore(latest_ckpt).expect_partial()
    logger.info("Backbone weights restored successfully (128D invariant contract).")

    # 3. Load Dual Uncertainty Arbiter (Phase 2)
    logger.info(f"Loading Dual Uncertainty Arbiter profiles: {profiles_path}")
    arbiter = DualUncertaintyArbiter(n_classes=10, latent_dim=128)
    arbiter.load(profiles_path)
    logger.info(
        f"Calibrated Thresholds -> Mahalanobis (tau_M): {arbiter.threshold_mahalanobis:.2f} | "
        f"Entropy (tau_H): {arbiter.threshold_entropy:.2f}"
    )

    # 4. Load Episodic Memory Bank (Phase 3)
    logger.info(f"Loading seeded episodic memory bank: {memory_bank_path}")
    memory = KNNBanditAgent128D(capacity=MEMORY_CAPACITY, k=10, latent_dim=LATENT_DIM)
    memory.load(memory_bank_path)
    logger.info(f"Episodic memory bank initialized: {memory.size}/{memory.capacity} clean prototypes.")

    # 5. Initialize Invariant RLAgent and RewardManager
    logger.info("Initializing Double DQN RLAgent with Prioritized Experience Replay...")
    rl_agent = RLAgent(
        state_dim=5,
        n_actions=4,
        lr=1e-3,
        gamma=0.99,
        target_update_interval=100,
        buffer_capacity=10000,
        device="cpu"
    )

    reward_manager = RewardManager(
        buffer_size=100,
        alpha_decay=0.995,
        min_alpha=0.01,
        latent_dim=LATENT_DIM,
    )

    # 6. Load Streaming Dataset
    logger.info("Loading CIFAR-10 streaming dataset...")
    x_train, y_train = load_dataset_raw("cifar10", CIFAR10_DATA_DIR, kind="train")
    x_train = (x_train.astype(np.float32) / 255.0)
    y_train = y_train.astype(np.int32)
    n_samples_available = len(x_train)

    logger.info(
        f"Simulation Parameters: Steps={steps:,} | Noise Rate={noise_rate:.2f} | "
        f"Redundancy Rate={redundancy_rate:.2f} | Batch Size={batch_size} | "
        f"PER Capacity={rl_agent.replay_buffer.capacity:,}"
    )

    # Prepare CSV Writer
    csv_file = open(output_csv_path, mode="w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["step", "size", "reward_mean", "alpha", "loss", "epsilon"])

    # Streaming chunk buffer management
    chunk_size = 2048
    chunk_x = np.empty((0, 32, 32, 3), dtype=np.float32)
    chunk_y = np.empty(0, dtype=np.int32)
    chunk_feats = np.empty((0, LATENT_DIM), dtype=np.float32)
    chunk_preds = np.empty(0, dtype=np.int32)
    chunk_mah = np.empty(0, dtype=np.float32)
    chunk_ent = np.empty(0, dtype=np.float32)
    chunk_is_ood = np.empty(0, dtype=bool)
    chunk_is_err = np.empty(0, dtype=bool)
    chunk_ptr = 0
    stream_idx = 0

    def refill_stream():
        nonlocal chunk_x, chunk_y, chunk_feats, chunk_preds, chunk_mah, chunk_ent
        nonlocal chunk_is_ood, chunk_is_err, chunk_ptr, stream_idx

        indices = (np.arange(stream_idx, stream_idx + chunk_size)) % n_samples_available
        stream_idx = (stream_idx + chunk_size) % n_samples_available

        bx = x_train[indices].copy()
        by = y_train[indices].copy()

        # 1. Inject temporal redundancy: repeat existing clean prototypes with micro-perturbation
        rand_vals = np.random.rand(chunk_size)
        red_mask = rand_vals < redundancy_rate
        noise_mask = (rand_vals >= redundancy_rate) & (rand_vals < (redundancy_rate + noise_rate))

        # 2. Inject noise drift into designated fraction
        if np.any(noise_mask):
            noisy_indices = np.where(noise_mask)[0]
            for ni in noisy_indices:
                selected_noise = float(np.random.choice(noise_levels))
                bx[ni] = inject_noise_batch(bx[ni : ni + 1], selected_noise)[0]

        # Vectorized CNN feature extraction
        feats, preds, probs = extract_features_chunk(cnn, bx)

        # For redundant entries, directly substitute states from memory bank with near-zero distance
        if np.any(red_mask):
            red_indices = np.where(red_mask)[0]
            mem_sample_idx = np.random.choice(memory.size, size=len(red_indices))
            # Create micro-perturbation: d_min in [0.01, 0.05]
            jitter = np.random.normal(loc=0.0, scale=0.01, size=(len(red_indices), LATENT_DIM)).astype(np.float32)
            feats[red_indices] = memory._states[mem_sample_idx] + jitter
            by[red_indices] = memory._actions[mem_sample_idx]
            preds[red_indices] = by[red_indices]
            # Low entropy and nominal mahalanobis for clean redundant samples
            probs[red_indices] = 0.01
            for i, c in enumerate(by[red_indices]):
                probs[red_indices[i], c] = 0.91

        mah_dists = arbiter.compute_mahalanobis_batch(feats)
        entropies = arbiter.compute_entropy_batch(probs)

        is_oods = (mah_dists > arbiter.threshold_mahalanobis) | (entropies > arbiter.threshold_entropy)
        is_errs = (preds != by)

        chunk_x = bx
        chunk_y = by
        chunk_feats = feats
        chunk_preds = preds
        chunk_mah = mah_dists
        chunk_ent = entropies
        chunk_is_ood = is_oods
        chunk_is_err = is_errs
        chunk_ptr = 0

    refill_stream()

    # Metrics Tracking
    all_rewards: List[float] = []
    recent_rewards: List[float] = []
    action_counts: Dict[int, int] = {0: 0, 1: 0, 2: 0, 3: 0}
    recent_loss: float = 0.0
    decision_steps = 0
    t_start = time.time()

    logger.info("Executing prequential streaming simulation...")

    for step in range(steps):
        if chunk_ptr >= len(chunk_feats):
            refill_stream()

        z_t = chunk_feats[chunk_ptr]
        y_true = int(chunk_y[chunk_ptr])
        pred_cnn = int(chunk_preds[chunk_ptr])
        d_M = float(chunk_mah[chunk_ptr])
        local_ent = float(chunk_ent[chunk_ptr])
        is_ood_sample = bool(chunk_is_ood[chunk_ptr])
        is_err_sample = bool(chunk_is_err[chunk_ptr])
        chunk_ptr += 1

        # Linear epsilon decay
        progress = min(1.0, float(step / max(1, steps - 1)))
        current_epsilon = float(max(epsilon_end, epsilon_start - (epsilon_start - epsilon_end) * progress))

        # Measure distance to nearest neighbor in memory
        diff = memory._states[:memory.size] - z_t
        d_min = float(np.sqrt(np.min(np.sum(diff * diff, axis=1))))
        is_redundant = (d_min < 0.2 and not is_ood_sample)
        is_anomaly = is_ood_sample or is_err_sample or is_redundant

        if is_anomaly:
            decision_steps += 1
            reward_manager.update_validation_buffer(z_t, y_true, pred_cnn)

            ram_occ = float(memory.size / memory.capacity)

            # Construct normalized 5D state
            state_vec = rl_agent.get_state_vector(
                mahalanobis_dist=d_M,
                local_entropy=local_ent,
                min_knn_dist=d_min,
                prediction_error=1.0 if is_err_sample else 0.0,
                ram_occupancy=ram_occ,
            )

            # Action Selection: epsilon-greedy
            action = rl_agent.select_action(state_vec, epsilon=current_epsilon)
            action_counts[action] = action_counts.get(action, 0) + 1

            # Execute Eviction Action
            if action == 0:
                evicted_idx = -1  # Action 0: Reject / Ignore
            elif action == 1:
                evicted_idx = memory.evict_oldest(z_t, y_true, 1.0)
            elif action == 2:
                evicted_idx = memory.evict_least_frequently_used(z_t, y_true, 1.0)
            elif action == 3:
                evicted_idx = memory.evict_most_redundant(z_t, y_true, 1.0)
            else:
                evicted_idx = memory.evict_oldest(z_t, y_true, 1.0)

            # Composite Reward Shaping anchored in EAAI Literature
            # 1. Anti-pollution shielding under severe noise / OOD (Alonso & Krichmar 2024; Alabed 2019)
            if is_ood_sample:
                if action == 0:
                    step_reward = 1.0   # Correctly shielded cache from noise corruption
                else:
                    step_reward = -1.0  # Polluted cache with noisy representation
            # 2. Distribution matching under redundancy (Isele & Cosgun, AAAI 2018)
            elif is_redundant:
                if action == 3:
                    step_reward = 1.0   # Evicted redundant prototype within same class
                elif action == 0:
                    step_reward = -0.5  # Missed update on valid prototype
                else:
                    step_reward = 0.2   # Sub-optimal blind eviction
            # 3. Novel in-distribution hard representations
            else:
                if action in [1, 2, 3]:
                    step_reward = 0.8   # Stored novel informative representation
                else:
                    step_reward = -0.8  # Rejected novel prototype

            all_rewards.append(step_reward)
            recent_rewards.append(step_reward)

            # Next state representation
            next_ram_occ = float(memory.size / memory.capacity)
            next_state_vec = rl_agent.get_state_vector(
                mahalanobis_dist=d_M,
                local_entropy=local_ent,
                min_knn_dist=d_min,
                prediction_error=1.0 if is_err_sample else 0.0,
                ram_occupancy=next_ram_occ,
            )

            # Store transition in Prioritized Experience Replay buffer
            rl_agent.store_transition(
                state=state_vec,
                action=action,
                reward=step_reward,
                next_state=next_state_vec,
                done=False
            )

            # Double DQN Gradient Step
            if decision_steps % train_frequency == 0 and rl_agent.replay_buffer.size >= batch_size:
                loss = rl_agent.update_weights(batch_size=batch_size)
                if loss is not None:
                    recent_loss = float(loss)

        else:
            # Clean nominal sample handled by CNN
            step_reward = 1.0
            all_rewards.append(step_reward)
            recent_rewards.append(step_reward)

        # Periodic Structured Logging
        if (step + 1) % log_interval == 0 or step == steps - 1:
            mean_r = float(np.mean(recent_rewards[-log_interval:]) if recent_rewards else 0.0)
            current_alpha = reward_manager.get_current_alpha()

            csv_writer.writerow([
                step + 1,
                memory.size,
                f"{mean_r:.4f}",
                f"{current_alpha:.4f}",
                f"{recent_loss:.4f}",
                f"{current_epsilon:.4f}",
            ])
            csv_file.flush()

            elapsed = time.time() - t_start
            fps = (step + 1) / max(0.001, elapsed)
            logger.info(
                f"Step {step + 1:>6,}/{steps:,} | "
                f"Mem: {memory.size:>4}/{memory.capacity} | "
                f"R_mean: {mean_r:>+6.3f} | "
                f"Loss: {recent_loss:>6.4f} | "
                f"Eps: {current_epsilon:>5.3f} | "
                f"A[0..3]: [{action_counts[0]},{action_counts[1]},{action_counts[2]},{action_counts[3]}] | "
                f"Speed: {fps:.0f} steps/s"
            )

    csv_file.close()
    logger.info(f"Structured CSV log saved to: {output_csv_path}")

    # Save trained RLAgent weights
    logger.info(f"Saving trained RLAgent checkpoint to: {save_agent_path}")
    rl_agent.save(save_agent_path)

    total_time = time.time() - t_start
    logger.info("=" * 70)
    logger.info(f"PHASE 4 SIMULATION COMPLETE: {steps:,} steps processed in {total_time:.2f}s.")
    logger.info(f"Final Action Distribution: {action_counts}")
    logger.info(f"Total Transitions in PER: {rl_agent.replay_buffer.size:,}")
    logger.info("=" * 70)

    # 7. Verification Gate Check
    logger.info("\nExecuting Phase 4 Verification Gate...")
    eval_agent = RLAgent(state_dim=5, n_actions=4, device="cpu")
    eval_agent.load(save_agent_path)

    s_noisy = np.array([30.0, 2.2, 1.5, 1.0, 1.0], dtype=np.float32)
    act_noisy = eval_agent.select_action(s_noisy, epsilon=0.0)

    s_red = np.array([5.0, 0.2, 0.02, 0.0, 1.0], dtype=np.float32)
    act_red = eval_agent.select_action(s_red, epsilon=0.0)

    logger.info(f"Policy Verification -> Action on extreme noise: {act_noisy} | Action on redundancy: {act_red}")

    return {
        "steps": steps,
        "action_counts": action_counts,
        "act_noisy": act_noisy,
        "act_red": act_red,
        "checkpoint": save_agent_path,
        "log": output_csv_path,
    }


def main():
    parser = argparse.ArgumentParser(description="CIFAR-10 Online RL Simulation under Concept Drift.")
    parser.add_argument("--steps", type=int, default=SIMULATION_STEPS, help="Total simulation steps (default: 50000).")
    parser.add_argument("--noise-rate", type=float, default=0.15, help="Noise injection rate (default: 0.15).")
    parser.add_argument("--redundancy-rate", type=float, default=0.15, help="Redundancy injection rate (default: 0.15).")
    parser.add_argument("--batch-size", type=int, default=32, help="DQN training batch size (default: 32).")
    parser.add_argument("--checkpoint-dir", type=str, default="outputs/cifar10/checkpoints", help="CNN checkpoint directory.")
    parser.add_argument("--profiles", type=str, default="outputs/cifar10/mahalanobis_pp_profiles.npz", help="Arbiter profiles.")
    parser.add_argument("--memory-bank", type=str, default="outputs/cifar10/knn_memory_bank_128d.npz", help="Seeded memory bank.")
    parser.add_argument("--output-csv", type=str, default="outputs/cifar10/train_rl_simulation_log.csv", help="CSV log output.")
    parser.add_argument("--save-agent", type=str, default="outputs/cifar10/checkpoints/rl_agent_phase3.pt", help="Saved agent path.")
    parser.add_argument("--log-interval", type=int, default=SIMULATION_LOG_INTERVAL, help="Logging interval.")
    args = parser.parse_args()

    run_cifar10_simulation(
        steps=args.steps,
        noise_rate=args.noise_rate,
        redundancy_rate=args.redundancy_rate,
        batch_size=args.batch_size,
        checkpoint_dir=args.checkpoint_dir,
        profiles_path=args.profiles,
        memory_bank_path=args.memory_bank,
        output_csv_path=args.output_csv,
        save_agent_path=args.save_agent,
        log_interval=args.log_interval,
    )


if __name__ == "__main__":
    main()
