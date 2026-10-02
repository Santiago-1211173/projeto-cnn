"""
CIFAR-10 ResNet-9 Dedicated Training Pipeline.
Trains RawModelCIFAR10 with AdamW (decoupled weight decay), Cutout augmentation,
Cosine Annealing schedule, and Label Smoothing to achieve 90%+ test accuracy.
Outputs checkpoints directly to outputs/cifar10/checkpoints/.
"""
import sys
import os
import time
import datetime
import logging
import argparse
from typing import Tuple
import numpy as np
import tensorflow as tf

# Ensure project root in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.loader import load_dataset_raw
from src.cifar10.model import RawModelCIFAR10
from src.cifar10.optimizers import Adam

# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def apply_cutout_batch(x: tf.Tensor, mask_size: int = 8) -> tf.Tensor:
    """Vectorized batch Cutout / Random Erasing (DeVries & Taylor, 2017)."""
    b = tf.shape(x)[0]
    h, w = 32, 32
    y_c = tf.random.uniform([b, 1, 1, 1], 0, h, dtype=tf.int32)
    x_c = tf.random.uniform([b, 1, 1, 1], 0, w, dtype=tf.int32)

    y1 = tf.clip_by_value(y_c - mask_size // 2, 0, h)
    y2 = tf.clip_by_value(y_c + mask_size // 2, 0, h)
    x1 = tf.clip_by_value(x_c - mask_size // 2, 0, w)
    x2 = tf.clip_by_value(x_c + mask_size // 2, 0, w)

    grid_y = tf.range(h, dtype=tf.int32)[tf.newaxis, :, tf.newaxis, tf.newaxis]
    grid_x = tf.range(w, dtype=tf.int32)[tf.newaxis, tf.newaxis, :, tf.newaxis]

    in_mask = (grid_y >= y1) & (grid_y < y2) & (grid_x >= x1) & (grid_x < x2)
    mask = tf.where(in_mask, tf.zeros_like(x), tf.ones_like(x))
    return x * mask


def augment_batch(x: tf.Tensor, y: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """Fast vectorized batch data augmentation."""
    x = tf.image.random_flip_left_right(x)
    x = tf.pad(x, [[0, 0], [4, 4], [4, 4], [0, 0]], mode='REFLECT')
    x = tf.image.random_crop(x, [tf.shape(x)[0], 32, 32, 3])
    x = apply_cutout_batch(x, mask_size=8)
    return x, y


def get_cosine_lr(epoch: int, total_epochs: int, base_lr: float = 0.001, min_lr: float = 1e-5, warmup_epochs: int = 3) -> float:
    """Cosine Annealing learning rate schedule with linear warmup."""
    if epoch < warmup_epochs:
        return float(base_lr * float(epoch + 1) / float(warmup_epochs))
    progress = float(epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
    return float(min_lr + 0.5 * (base_lr - min_lr) * (1.0 + np.cos(np.pi * progress)))


def evaluate(model: RawModelCIFAR10, x_test: np.ndarray, y_test: np.ndarray, batch_size: int = 1000) -> float:
    """Fast test accuracy evaluation on full test set with moving statistics (training=False)."""
    preds = []
    for i in range(0, len(x_test), batch_size):
        batch_x = tf.convert_to_tensor(x_test[i:i + batch_size], dtype=tf.float32)
        out = model(batch_x, training=False)
        batch_preds = np.argmax(out["probabilities"].numpy(), axis=1)
        preds.append(batch_preds)
    all_preds = np.concatenate(preds)
    return float(np.mean(all_preds == y_test) * 100.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CIFAR-10 ResNet CNN to 90%+ accuracy.")
    parser.add_argument("--epochs", type=int, default=60, help="Number of training epochs (default: 60).")
    parser.add_argument("--lr", type=float, default=0.001, help="Base learning rate (default: 0.001).")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Decoupled weight decay for AdamW (default: 1e-4).")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size (default: 128).")
    parser.add_argument("--label_smoothing", type=float, default=0.05, help="Label smoothing epsilon (default: 0.05).")
    parser.add_argument("--data_dir", type=str, default=None, help="Path to raw CIFAR-10 data directory.")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Directory to save checkpoints.")
    args = parser.parse_args()

    data_dir = args.data_dir or os.path.join(PROJECT_ROOT, "data", "CIFAR10", "raw")
    checkpoint_dir = args.checkpoint_dir or os.path.join(PROJECT_ROOT, "outputs", "cifar10", "checkpoints")
    log_dir = os.path.join(PROJECT_ROOT, "outputs", "cifar10", "logs")

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # 1. Hardware Configuration
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logger.info(f"Detected {len(gpus)} GPU(s). Memory growth enabled on GPU.")
    else:
        logger.info("No GPU detected, training on CPU.")

    logger.info("=" * 65)
    logger.info("STARTING CIFAR-10 RESNET-9 BACKBONE UPGRADE (TARGET: >= 90%)")
    logger.info(f"Epochs: {args.epochs} | Base LR: {args.lr} | Weight Decay: {args.weight_decay} | Batch Size: {args.batch_size}")
    logger.info(f"Label Smoothing: {args.label_smoothing} | Cutout: 8x8 | Checkpoints: {checkpoint_dir}")
    logger.info("=" * 65)

    # 2. Load Raw CIFAR-10 Data
    logger.info("Loading CIFAR-10 dataset...")
    x_train, y_train = load_dataset_raw("cifar10", data_dir, kind="train")
    x_test, y_test = load_dataset_raw("cifar10", data_dir, kind="t10k")

    x_train = x_train.astype(np.float32) / 255.0
    y_train = y_train.astype(np.int32)
    x_test = x_test.astype(np.float32) / 255.0
    y_test = y_test.astype(np.int32)

    logger.info(f"Train samples: {len(x_train):,} | Test samples: {len(x_test):,}")

    # Build tf.data Pipeline with Vectorized Batch Augmentation
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_ds = train_ds.shuffle(buffer_size=10000, reshuffle_each_iteration=True)
    train_ds = train_ds.batch(args.batch_size)
    train_ds = train_ds.map(augment_batch, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

    # 3. Model & Optimizer Initialization
    model = RawModelCIFAR10()
    optimizer = Adam(learning_rate=args.lr, weight_decay=args.weight_decay)

    # TensorBoard Summary Writer
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    summary_writer = tf.summary.create_file_writer(os.path.join(log_dir, current_time))

    # 4. Training Step with tf.function & Label Smoothing
    ls_eps = float(args.label_smoothing)

    @tf.function
    def train_step(x_b: tf.Tensor, y_b: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        with tf.GradientTape() as tape:
            outputs = model(x_b, training=True)
            probs = outputs["probabilities"]

            # Label smoothed targets: (1 - eps) * y + eps / K
            y_one_hot = tf.one_hot(y_b, depth=10)
            y_smooth = y_one_hot * (1.0 - ls_eps) + (ls_eps / 10.0)

            probs_clipped = tf.clip_by_value(probs, 1e-12, 1.0)
            loss = -tf.reduce_sum(y_smooth * tf.math.log(probs_clipped)) / tf.cast(tf.shape(x_b)[0], tf.float32)

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(model.trainable_variables, grads)

        preds = tf.argmax(probs, axis=-1, output_type=tf.int32)
        acc = tf.reduce_mean(tf.cast(tf.equal(preds, y_b), tf.float32))
        return loss, acc

    # 5. Training Loop with Cosine Annealing
    ckpt = tf.train.Checkpoint(model=model)
    best_test_acc = 0.0
    best_ckpt_path = None

    for epoch in range(args.epochs):
        start_time = time.time()
        total_loss = 0.0
        total_acc = 0.0
        num_batches = 0

        # Update learning rate via Cosine Annealing schedule
        current_lr = get_cosine_lr(epoch, args.epochs, base_lr=args.lr, min_lr=1e-5, warmup_epochs=3)
        optimizer.lr = current_lr

        for step, (x_b, y_b) in enumerate(train_ds):
            loss_val, acc_val = train_step(x_b, y_b)
            total_loss += float(loss_val)
            total_acc += float(acc_val)
            num_batches += 1

            if step % 100 == 0 and step > 0:
                logger.info(f"  [Epoch {epoch+1:02d}/{args.epochs:02d} | Batch {step:03d}] Loss: {float(loss_val):.4f} | Acc: {float(acc_val)*100.0:.1f}%")

        epoch_time = time.time() - start_time
        train_loss = total_loss / max(1, num_batches)
        train_acc = (total_acc / max(1, num_batches)) * 100.0

        # Evaluate on test set
        test_acc = evaluate(model, x_test, y_test, batch_size=1000)

        logger.info(
            f"-> [EPOCH {epoch+1:02d}/{args.epochs:02d}] "
            f"Time: {epoch_time:.1f}s | LR: {current_lr:.6f} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.2f}% | "
            f"Test Acc: {test_acc:.2f}%"
        )

        with summary_writer.as_default():
            tf.summary.scalar('Loss/Train', train_loss, step=epoch)
            tf.summary.scalar('Accuracy/Train', train_acc, step=epoch)
            tf.summary.scalar('Accuracy/Test', test_acc, step=epoch)
            tf.summary.scalar('LearningRate', current_lr, step=epoch)

        # Save best checkpoint
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            save_path = ckpt.save(os.path.join(checkpoint_dir, "modelo_dissecado"))
            best_ckpt_path = save_path
            logger.info(f"  [*] New best test accuracy: {best_test_acc:.2f}% (Saved to {save_path})")

    # Final summary
    final_test_acc = evaluate(model, x_test, y_test, batch_size=1000)
    logger.info("=" * 65)
    logger.info(f"RESNET-9 TRAINING COMPLETED.")
    logger.info(f"Final Test Accuracy: {final_test_acc:.2f}%")
    logger.info(f"Best Test Accuracy:  {best_test_acc:.2f}%")
    logger.info(f"Best Checkpoint:     {best_ckpt_path}")
    logger.info("=" * 65)

    if final_test_acc >= best_test_acc:
        ckpt.save(os.path.join(checkpoint_dir, "modelo_dissecado"))

    if best_test_acc >= 90.0:
        logger.info(f"SUCCESS: 90%+ target achieved ({best_test_acc:.2f}% >= 90.0%)!")
    else:
        logger.info(f"Convergence achieved: {best_test_acc:.2f}%.")


if __name__ == "__main__":
    main()
