"""
CIFAR-100 ResNet-14 Dedicated Training Pipeline.
Trains RawModelCIFAR100 with AdamW / SGDMomentum, CutMix / Cutout augmentation,
Cosine Annealing schedule, Label Smoothing, and Channel-wise Z-Score standardization.
Outputs checkpoints directly to outputs/cifar100/checkpoints_opt1/ (or user-specified).
"""

from __future__ import annotations
import sys
import os
import time
import datetime
import logging
import argparse
import json
import csv
from typing import Tuple
import numpy as np

# Ensure project root in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import CIFAR100_DATA_DIR, RANDOM_SEED

# Ensure Python with TensorFlow is used if launched from base Anaconda
try:
    import tensorflow as tf
except ModuleNotFoundError:
    tf_python = r"C:\Users\sanfr\.conda\envs\tf_l40s\python.exe"
    if os.path.exists(tf_python) and sys.executable.lower() != tf_python.lower():
        import subprocess
        res = subprocess.run([tf_python] + sys.argv)
        sys.exit(res.returncode)
    raise

from src.data.cifar100_loader import load_cifar100_raw
from src.cifar100.model import RawModelCIFAR100, RawModelCIFAR100V2
from src.cifar100.optimizers import Adam, SGDMomentum

# Logging Setup
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Canonical CIFAR-100 Channel Statistics
CIFAR100_MEAN = tf.constant([0.50707516, 0.48654887, 0.44091784], dtype=tf.float32)
CIFAR100_STD = tf.constant([0.26733429, 0.25643846, 0.27615047], dtype=tf.float32)


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


def apply_cutmix_batch(
    x: tf.Tensor,
    y_smooth: tf.Tensor,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """CutMix augmentation (Yun et al., ICCV 2019) on batch tensors."""
    b = tf.shape(x)[0]
    indices = tf.random.shuffle(tf.range(b))
    x2 = tf.gather(x, indices)
    y2 = tf.gather(y_smooth, indices)

    lam = tf.random.uniform([], 0.0, 1.0)
    h, w = 32, 32
    cut_ratio = tf.sqrt(1.0 - lam)
    rw = tf.cast(32.0 * cut_ratio, tf.int32)
    rh = tf.cast(32.0 * cut_ratio, tf.int32)

    rx = tf.random.uniform([], 0, w, dtype=tf.int32)
    ry = tf.random.uniform([], 0, h, dtype=tf.int32)

    x1 = tf.clip_by_value(rx - rw // 2, 0, w)
    x2_box = tf.clip_by_value(rx + rw // 2, 0, w)
    y1 = tf.clip_by_value(ry - rh // 2, 0, h)
    y2_box = tf.clip_by_value(ry + rh // 2, 0, h)

    grid_y = tf.range(h, dtype=tf.int32)[:, tf.newaxis, tf.newaxis]
    grid_x = tf.range(w, dtype=tf.int32)[tf.newaxis, :, tf.newaxis]
    in_box = (grid_y >= y1) & (grid_y < y2_box) & (grid_x >= x1) & (grid_x < x2_box)
    mask = tf.cast(in_box, tf.float32)

    actual_box_area = tf.cast((y2_box - y1) * (x2_box - x1), tf.float32)
    actual_lam = 1.0 - (actual_box_area / (32.0 * 32.0))

    x_mixed = x * (1.0 - mask) + x2 * mask
    y_mixed = actual_lam * y_smooth + (1.0 - actual_lam) * y2
    return x_mixed, y_mixed


def get_cosine_lr(
    epoch: int,
    total_epochs: int,
    base_lr: float = 0.001,
    min_lr: float = 1e-5,
    warmup_epochs: int = 5,
) -> float:
    """Cosine Annealing learning rate schedule with linear warmup."""
    if epoch < warmup_epochs:
        return float(base_lr * float(epoch + 1) / float(max(1, warmup_epochs)))
    progress = float(epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
    return float(min_lr + 0.5 * (base_lr - min_lr) * (1.0 + np.cos(np.pi * progress)))


def evaluate(
    model: RawModelCIFAR100,
    x_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int = 500,
    standardize: bool = True,
) -> float:
    """Fast test accuracy evaluation on full test set with moving statistics (training=False)."""
    preds = []
    for i in range(0, len(x_test), batch_size):
        batch_x = tf.convert_to_tensor(x_test[i : i + batch_size], dtype=tf.float32)
        if standardize:
            batch_x = (batch_x - CIFAR100_MEAN) / CIFAR100_STD
        out = model(batch_x, training=False)
        batch_preds = np.argmax(out["probabilities"].numpy(), axis=1)
        preds.append(batch_preds)
    all_preds = np.concatenate(preds)
    return float(np.mean(all_preds == y_test) * 100.0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train CIFAR-100 ResNet-14 CNN backbone."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Base learning rate (default: 0.001 for AdamW, 0.1 for SGD).",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        choices=["adamw", "sgd"],
        default="adamw",
        help="Optimizer choice: 'adamw' or 'sgd' (default: adamw).",
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=128,
        help="Latent representation dimension (default: 128).",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Decoupled weight decay (default: 1e-4 for AdamW, 5e-4 for SGD).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size (default: 128).",
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.1,
        help="Label smoothing epsilon (default: 0.1).",
    )
    parser.add_argument(
        "--cutmix-prob",
        type=float,
        default=0.5,
        help="Probability of CutMix augmentation per batch (default: 0.5).",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=5,
        help="Warmup epochs for cosine annealing (default: 5).",
    )
    parser.add_argument(
        "--no-standardize",
        action="store_true",
        help="Disable channel Z-Score standardization (defaults to enabled).",
    )
    parser.add_argument(
        "--model-version",
        type=str,
        choices=["v1", "v2"],
        default="v1",
        help="Model architecture version: 'v1' (ResNet-14) or 'v2' (ResNet-18 V2) (default: v1).",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Path to raw CIFAR-100 data directory.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Directory to save checkpoints.",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Directory to save logs.",
    )
    args = parser.parse_args()

    standardize = not args.no_standardize
    model_version = args.model_version.lower()

    # Adjust default learning rate and weight decay if SGD is requested and not overridden
    if args.optimizer == "sgd" and args.lr == 0.001:
        args.lr = 0.1
    if args.optimizer == "sgd" and args.weight_decay == 1e-4:
        args.weight_decay = 5e-4

    # Determinism
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    default_ckpt_sub = "checkpoints_opt2" if model_version == "v2" else "checkpoints_opt1"
    default_log_sub = "logs_opt2" if model_version == "v2" else "logs_opt1"

    data_dir = args.data_dir or CIFAR100_DATA_DIR
    checkpoint_dir = args.checkpoint_dir or os.path.join(
        PROJECT_ROOT, "outputs", "cifar100", default_ckpt_sub
    )
    log_dir = args.log_dir or os.path.join(
        PROJECT_ROOT, "outputs", "cifar100", default_log_sub
    )

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # 1. Hardware Configuration
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                pass
        logger.info(f"Detected {len(gpus)} GPU(s). Memory growth enabled.")
    else:
        logger.info("No GPU detected, training on CPU.")

    arch_title = (
        "RESNET-18 V2 (4 STAGES, STRIDED DOWNSAMPLING, BATCHNORM BOTTLENECK)"
        if model_version == "v2"
        else "RESNET-14 V1 (3 STAGES, MAXPOOL DOWNSAMPLING)"
    )

    logger.info("=" * 72)
    logger.info(f"CIFAR-100 {arch_title} TRAINING PIPELINE")
    logger.info(
        f"Epochs: {args.epochs} | Optimizer: {args.optimizer.upper()} | Base LR: {args.lr} | "
        f"Latent Dim: {args.latent_dim} | Batch Size: {args.batch_size}"
    )
    logger.info(
        f"Weight Decay: {args.weight_decay} | Label Smoothing: {args.label_smoothing} | "
        f"CutMix Prob: {args.cutmix_prob} | Standardize: {standardize}"
    )
    logger.info(f"Checkpoints: {checkpoint_dir}")
    logger.info("=" * 72)

    # 2. Load Raw CIFAR-100 Data
    logger.info(f"Loading CIFAR-100 dataset from: {data_dir}...")
    x_train, y_train, x_test, y_test = load_cifar100_raw(data_dir=data_dir)

    x_train = x_train.astype(np.float32)
    y_train = y_train.astype(np.int32)
    x_test = x_test.astype(np.float32)
    y_test = y_test.astype(np.int32)

    logger.info(f"Train samples: {len(x_train):,} | Test samples: {len(x_test):,}")

    # Build tf.data Pipeline (CPU hosts batching, GPU executes augmentations)
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_ds = train_ds.shuffle(buffer_size=10000, seed=RANDOM_SEED, reshuffle_each_iteration=True)
    train_ds = train_ds.batch(args.batch_size)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

    # 3. Model & Optimizer Initialization
    if model_version == "v2":
        model = RawModelCIFAR100V2(latent_dim=args.latent_dim)
    else:
        model = RawModelCIFAR100(latent_dim=args.latent_dim)
    if args.optimizer == "sgd":
        optimizer = SGDMomentum(
            learning_rate=args.lr,
            momentum=0.9,
            nesterov=True,
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = Adam(learning_rate=args.lr, weight_decay=args.weight_decay)

    # TensorBoard Summary Writer
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    summary_writer = tf.summary.create_file_writer(os.path.join(log_dir, current_time))

    # 4. Training Step with tf.function & GPU-Accelerated Augmentation
    ls_eps = float(args.label_smoothing)
    cutmix_p = float(args.cutmix_prob)

    @tf.function
    def train_step(x_b: tf.Tensor, y_b: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        b = tf.shape(x_b)[0]

        # Geometric augmentation: Random Horizontal Flip + Pad & Crop
        x_aug = tf.image.random_flip_left_right(x_b)
        x_aug = tf.pad(x_aug, [[0, 0], [4, 4], [4, 4], [0, 0]], mode="REFLECT")
        x_aug = tf.image.random_crop(x_aug, [b, 32, 32, 3])

        # Smoothed targets
        y_one_hot = tf.one_hot(y_b, depth=100)
        y_smooth = y_one_hot * (1.0 - ls_eps) + (ls_eps / 100.0)

        # CutMix or Cutout
        if cutmix_p > 0.0:
            rand_val = tf.random.uniform([])
            if rand_val < cutmix_p:
                x_aug, y_smooth = apply_cutmix_batch(x_aug, y_smooth)
            else:
                x_aug = apply_cutout_batch(x_aug, mask_size=8)
        else:
            x_aug = apply_cutout_batch(x_aug, mask_size=8)

        # Channel Z-score standardization
        if standardize:
            x_aug = (x_aug - CIFAR100_MEAN) / CIFAR100_STD

        with tf.GradientTape() as tape:
            outputs = model(x_aug, training=True)
            probs = outputs["probabilities"]
            probs_clipped = tf.clip_by_value(probs, 1e-12, 1.0)
            loss = -tf.reduce_sum(y_smooth * tf.math.log(probs_clipped)) / tf.cast(
                b, tf.float32
            )

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(model.trainable_variables, grads)

        preds = tf.argmax(probs, axis=-1, output_type=tf.int32)
        true_labels = tf.argmax(y_smooth, axis=-1, output_type=tf.int32)
        acc = tf.reduce_mean(tf.cast(tf.equal(preds, true_labels), tf.float32))
        return loss, acc

    # 5. Training Loop with Cosine Annealing and History Logging
    ckpt = tf.train.Checkpoint(model=model)
    best_test_acc = 0.0
    best_ckpt_path = None
    best_epoch = 0

    history_csv = os.path.join(checkpoint_dir, "train_history.csv")
    csv_file = open(history_csv, mode="w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["epoch", "time_s", "lr", "train_loss", "train_acc", "test_acc"])

    min_lr = 1e-5 if args.optimizer == "adamw" else 1e-4

    for epoch in range(args.epochs):
        start_time = time.time()
        total_loss = 0.0
        total_acc = 0.0
        num_batches = 0

        # Update learning rate via Cosine Annealing schedule
        current_lr = get_cosine_lr(
            epoch,
            args.epochs,
            base_lr=args.lr,
            min_lr=min_lr,
            warmup_epochs=args.warmup_epochs,
        )
        optimizer.lr = current_lr

        for step, (x_b, y_b) in enumerate(train_ds):
            loss_val, acc_val = train_step(x_b, y_b)
            total_loss += float(loss_val)
            total_acc += float(acc_val)
            num_batches += 1

            if step % 100 == 0 and step > 0:
                logger.info(
                    f"  [Epoch {epoch+1:03d}/{args.epochs:03d} | Batch {step:03d}] "
                    f"Loss: {float(loss_val):.4f} | Acc: {float(acc_val)*100.0:.1f}%"
                )

        epoch_time = time.time() - start_time
        train_loss = total_loss / max(1, num_batches)
        train_acc = (total_acc / max(1, num_batches)) * 100.0

        # Evaluate on test set
        test_acc = evaluate(
            model, x_test, y_test, batch_size=500, standardize=standardize
        )

        logger.info(
            f"-> [EPOCH {epoch+1:03d}/{args.epochs:03d}] "
            f"Time: {epoch_time:.1f}s | LR: {current_lr:.6f} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.2f}% | "
            f"Test Acc: {test_acc:.2f}%"
        )

        csv_writer.writerow([
            epoch + 1,
            round(epoch_time, 2),
            round(current_lr, 7),
            round(train_loss, 4),
            round(train_acc, 2),
            round(test_acc, 2),
        ])
        csv_file.flush()

        with summary_writer.as_default():
            tf.summary.scalar("Loss/Train", train_loss, step=epoch)
            tf.summary.scalar("Accuracy/Train", train_acc, step=epoch)
            tf.summary.scalar("Accuracy/Test", test_acc, step=epoch)
            tf.summary.scalar("LearningRate", current_lr, step=epoch)

        # Save best checkpoint
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch + 1
            save_path = ckpt.save(os.path.join(checkpoint_dir, "modelo_dissecado"))
            best_ckpt_path = save_path
            logger.info(
                f"  [*] New best test accuracy: {best_test_acc:.2f}% (Saved to {save_path})"
            )

    csv_file.close()

    # Final evaluation and checkpoint
    final_test_acc = evaluate(
        model, x_test, y_test, batch_size=500, standardize=standardize
    )
    logger.info("=" * 72)
    logger.info(f"{arch_title} TRAINING COMPLETED.")
    logger.info(f"Final Test Accuracy: {final_test_acc:.2f}%")
    logger.info(f"Best Test Accuracy:  {best_test_acc:.2f}% (Epoch {best_epoch})")
    logger.info(f"Best Checkpoint:     {best_ckpt_path}")
    logger.info(f"Training History:    {history_csv}")
    logger.info("=" * 72)

    # Save metadata JSON for downstream pipelines
    meta = {
        "dataset": "cifar100",
        "model_version": model_version,
        "arch_title": arch_title,
        "num_classes": 100,
        "latent_dim": args.latent_dim,
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "best_test_acc": round(best_test_acc, 2),
        "final_test_acc": round(final_test_acc, 2),
        "standardize": standardize,
        "mean": [0.50707516, 0.48654887, 0.44091784],
        "std": [0.26733429, 0.25643846, 0.27615047],
        "optimizer": args.optimizer,
        "base_lr": args.lr,
        "weight_decay": args.weight_decay,
        "cutmix_prob": args.cutmix_prob,
        "label_smoothing": args.label_smoothing,
        "best_checkpoint": best_ckpt_path,
    }
    meta_path = os.path.join(checkpoint_dir, "model_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"Metadata saved to:   {meta_path}")


if __name__ == "__main__":
    main()
