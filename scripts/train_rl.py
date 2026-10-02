"""
Training script for the k-NN Bandit RL Agent (Episodic Memory).
Unifies 10D and 128D memory bank seeding using the centralized configuration.
Supports both MNIST and CIFAR-10 datasets via --dataset argument.
"""

import sys
import os
import argparse
import logging
import numpy as np
import tensorflow as tf

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def add_noise_batch(images: np.ndarray, intensity: float) -> np.ndarray:
    """Injects Gaussian noise into a batch of images."""
    noise = np.random.normal(loc=0.0, scale=intensity, size=images.shape)
    return np.clip(images + noise, 0., 1.)

def extract_states_cnn(images: np.ndarray, cnn, latent_dim: int, batch_size: int = 500) -> np.ndarray:
    """Passes images through the CNN and extracts either probabilities (10D) or latent features (128D)."""
    all_states = []
    for i in range(0, len(images), batch_size):
        batch = images[i : i + batch_size]
        batch_tensor = tf.convert_to_tensor(batch, dtype=tf.float32)
        outputs = cnn(batch_tensor)
        if latent_dim == 128:
            states = outputs["latent_features"].numpy()
        else:
            states = outputs["probabilities"].numpy()
        all_states.append(states)
    return np.vstack(all_states)

def main():
    parser = argparse.ArgumentParser(description="Train/Seed the k-NN Bandit Agent.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        choices=["mnist", "cifar10"],
        help="Dataset to seed memory with (default: mnist)."
    )
    parser.add_argument(
        "--latent_dim",
        type=int,
        default=128,
        choices=[10, 128],
        help="Dimensionality of state space: 10 (probabilities) or 128 (latent features)."
    )
    parser.add_argument(
        "--k",
        type=int,
        default=30,
        help="Number of nearest neighbors to query."
    )
    args = parser.parse_args()

    os.environ["DATASET"] = args.dataset
    if "src.config" in sys.modules:
        import importlib
        importlib.reload(sys.modules["src.config"])

    from src.config import (
        DATA_DIR,
        CHECKPOINT_DIR,
        MEMORY_BANK_10D_PATH,
        MEMORY_BANK_128D_PATH,
        NOISE_SWEEP,
        DATASET
    )
    from src.data.loader import load_dataset_raw
    from src.models.knn_bandit_agent import KNNBanditAgent

    if args.dataset == "cifar10":
        from src.models.custom_cnn_cifar10 import RawModelCIFAR10 as ModelClass
    else:
        from src.models.custom_cnn import RawModel as ModelClass

    # 1. Configure Hardware
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    # 2. Load CNN
    logger.info(f"Loading CNN (Feature Extractor) [{DATASET.upper()}]...")
    cnn = ModelClass()
    ckpt = tf.train.Checkpoint(model=cnn)
    latest_ckpt = tf.train.latest_checkpoint(CHECKPOINT_DIR)
    if not latest_ckpt:
        logger.error(f"Checkpoint not found in {CHECKPOINT_DIR}. Train the CNN first!")
        sys.exit(1)
    ckpt.restore(latest_ckpt).expect_partial()
    logger.info("CNN weights loaded.")

    # 3. Instantiate Agent
    if args.latent_dim == 128:
        logger.info(f"Instantiating 128D Specialist k-NN Agent (k={args.k}, use_pca=True, components=48)...")
        agent = KNNBanditAgent(k=args.k, latent_dim=128, use_pca=True, pca_components=48)
        save_path = MEMORY_BANK_128D_PATH
    else:
        logger.info(f"Instantiating 10D Specialist k-NN Agent (k={args.k}, use_pca=False)...")
        agent = KNNBanditAgent(k=args.k, latent_dim=10, use_pca=False)
        save_path = MEMORY_BANK_10D_PATH

    # 4. Load Dataset
    logger.info(f"Loading {DATASET.upper()} training dataset...")
    x_train, y_train = load_dataset_raw(args.dataset, DATA_DIR, kind='train')
    x_train = x_train.astype(np.float32) / 255.0

    # 5. Oracle Seeding across the noise sweep
    logger.info("\n==================================================")
    logger.info(f"ORACLE SEEDING (Noise levels: {NOISE_SWEEP})")
    logger.info("==================================================")

    for noise_level in NOISE_SWEEP:
        logger.info(f"\nProcessing noise level {noise_level}...")
        
        # Add noise
        if noise_level > 0.0:
            x_noisy = add_noise_batch(x_train, noise_level)
        else:
            x_noisy = x_train.copy()
        
        # Extract states
        states = extract_states_cnn(x_noisy, cnn, latent_dim=args.latent_dim, batch_size=500)
        
        # Extract probabilities for error tracking (always 10D)
        if args.latent_dim == 10:
            probs = states
        else:
            probs = extract_states_cnn(x_noisy, cnn, latent_dim=10, batch_size=500)
        
        preds_cnn = np.argmax(probs, axis=1)
        
        # Positive experiences: correct label -> +1.0
        agent.add_experience_batch(states, y_train, np.ones(len(y_train)))
        
        # Negative experiences: CNN wrong prediction -> -1.0
        errors = preds_cnn != y_train
        num_errors = np.sum(errors)
        if num_errors > 0:
            agent.add_experience_batch(
                states[errors], preds_cnn[errors], np.full(num_errors, -1.0)
            )
        
        acc_cnn = np.mean(preds_cnn == y_train) * 100
        logger.info(f"  Added {len(y_train):,} positive and {num_errors:,} negative samples | CNN Accuracy: {acc_cnn:.1f}%")

    # 6. Build the k-NN Index
    logger.info("\nBuilding k-NN search index...")
    agent.build_index()

    # 7. Quick Evaluation
    logger.info("\n==================================================")
    logger.info("QUICK TEST EVALUATION")
    logger.info("==================================================")

    sample_idx = np.random.choice(len(x_train), size=3000, replace=False)
    for test_noise in [0.0, 0.3, 0.6]:
        if test_noise > 0:
            sample_imgs = add_noise_batch(x_train[sample_idx], test_noise)
        else:
            sample_imgs = x_train[sample_idx]

        # Extract states for evaluation
        test_states = extract_states_cnn(sample_imgs, cnn, latent_dim=args.latent_dim, batch_size=500)
        
        # Extract probabilities for CNN benchmark
        if args.latent_dim == 10:
            test_probs = test_states
        else:
            test_probs = extract_states_cnn(sample_imgs, cnn, latent_dim=10, batch_size=500)

        acc_knn = np.mean(agent.get_action_batch(test_states, epsilon=0.0) == y_train[sample_idx]) * 100
        acc_cnn = np.mean(np.argmax(test_probs, axis=1) == y_train[sample_idx]) * 100
        
        condition = f"Noise {test_noise:.1f}" if test_noise > 0 else "Clean"
        logger.info(f"  [{condition}] CNN: {acc_cnn:.1f}% | k-NN Agent: {acc_knn:.1f}%")

    # Save memory bank
    stats = agent.get_memory_stats()
    logger.info(f"\nMemory size: {stats['size']:,}")
    logger.info(f"Positive rewards ratio: {stats['reward_positive_pct']:.1f}%")
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    agent.save(save_path)
    logger.info(f"Success! Agent memory saved to: {save_path}")

if __name__ == "__main__":
    main()
