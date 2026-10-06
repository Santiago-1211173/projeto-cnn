"""
Main training script for the CNN (Single-GPU).
Trains the custom CNN model using GPU if available.
Supports both MNIST and CIFAR-10 datasets via --dataset argument.
"""

import sys
import os
import time
import datetime
import logging
import argparse
from typing import Tuple
import tensorflow as tf

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from src.scratch.losses import categorical_crossentropy
from src.scratch.optimizers import SGD

# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def calculate_accuracy(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Calculates accuracy by comparing the predicted class with the true class."""
    predicted_classes = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
    correct_predictions = tf.equal(predicted_classes, y_true)
    return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

def main() -> None:
    parser = argparse.ArgumentParser(description="Train the custom CNN model.")
    parser.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "cifar10", "cifar100"],
                        help="Dataset to train on (default: mnist).")
    args = parser.parse_args()

    # Set DATASET env var BEFORE importing config
    os.environ["DATASET"] = args.dataset

    from src.config import (
        CNN_BATCH_SIZE, CNN_EPOCHS, CNN_LEARNING_RATE,
        DATA_DIR, CHECKPOINT_DIR, LOG_DIR, DATASET
    )

    # Select model and dataset loader based on dataset
    if args.dataset == "cifar100":
        from src.cifar100.model import RawModelCIFAR100 as ModelClass
        from src.data.cifar100_loader import create_cifar100_dataset
        train_dataset = create_cifar100_dataset(DATA_DIR, batch_size=CNN_BATCH_SIZE)
    elif args.dataset == "cifar10":
        from src.models.custom_cnn_cifar10 import RawModelCIFAR10 as ModelClass
        from src.data.cifar10_loader import create_cifar10_dataset
        train_dataset = create_cifar10_dataset(DATA_DIR, batch_size=CNN_BATCH_SIZE)
    else:
        from src.models.custom_cnn import RawModel as ModelClass
        from src.data.loader import create_dataset
        train_dataset = create_dataset(DATA_DIR, batch_size=CNN_BATCH_SIZE)

    # 1. Configure Hardware
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    logger.info("\n=========================================")
    logger.info(f"Starting CNN training [{DATASET.upper()}] on {'GPU 0' if gpus else 'CPU'}")
    logger.info("=========================================\n")

    # 2. Allocation on GPU if available
    device_name = '/GPU:0' if gpus else '/CPU:0'
    with tf.device(device_name):
        # Instantiate Model and Optimizer
        model = ModelClass()
        optimizer = SGD(learning_rate=CNN_LEARNING_RATE)

        # Setup TensorBoard Summary Writer
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        os.makedirs(LOG_DIR, exist_ok=True)
        summary_log_dir = os.path.join(LOG_DIR, current_time)
        summary_writer = tf.summary.create_file_writer(summary_log_dir)

        # 3. Training Step compiled via tf.function
        @tf.function
        def train_step(x_batch: tf.Tensor, y_batch: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
            with tf.GradientTape() as tape:
                outputs = model(x_batch)
                predictions = outputs["probabilities"]
                loss = categorical_crossentropy(y_batch, predictions, CNN_BATCH_SIZE)

            # Backpropagation
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(model.trainable_variables, gradients)

            # Metrics computation
            accuracy = calculate_accuracy(y_batch, predictions)
            return loss, accuracy

        # 4. Training Loop
        logger.info("Starting training loop...")
        for epoch in range(CNN_EPOCHS):
            start_time = time.time()
            total_loss = 0.0
            total_acc = 0.0
            steps = 0

            for step, (x_batch, y_batch) in enumerate(train_dataset):
                loss, acc = train_step(x_batch, y_batch)

                total_loss += float(loss)
                total_acc += float(acc)
                steps += 1

                if step % 50 == 0 and step > 0:
                    logger.info(f"  [Epoch {epoch+1} | Batch {step}] Loss: {float(loss):.4f} | Acc: {float(acc):.4f}")

            # End of epoch metrics
            avg_loss = total_loss / steps
            avg_acc = total_acc / steps
            epoch_time = time.time() - start_time

            logger.info(f"-> END OF EPOCH {epoch+1}: Time: {epoch_time:.2f}s | Avg Loss: {avg_loss:.4f} | Avg Acc: {avg_acc:.4f}\n")

            # Write to TensorBoard
            with summary_writer.as_default():
                tf.summary.scalar('Loss/Train', avg_loss, step=epoch)
                tf.summary.scalar('Accuracy/Train', avg_acc, step=epoch)

        # Save checkpoint at the end of training
        logger.info("Saving model checkpoint...")
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        ckpt = tf.train.Checkpoint(model=model)
        ckpt.save(os.path.join(CHECKPOINT_DIR, "modelo_dissecado"))
        logger.info(f"Training completed! Checkpoint saved to {CHECKPOINT_DIR}")

if __name__ == "__main__":
    main()
