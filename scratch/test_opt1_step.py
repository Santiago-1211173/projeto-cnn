import os
import sys
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import src.config
import tensorflow as tf
from src.cifar100.model import RawModelCIFAR100
from src.cifar100.optimizers import Adam

MEAN = tf.constant([0.50707516, 0.48654887, 0.44091784], dtype=tf.float32)
STD = tf.constant([0.26733429, 0.25643846, 0.27615047], dtype=tf.float32)

def apply_cutmix(x: tf.Tensor, y_smooth: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    batch_size = tf.shape(x)[0]
    indices = tf.random.shuffle(tf.range(batch_size))
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

# Test step
model = RawModelCIFAR100(latent_dim=128)
opt = Adam(learning_rate=0.001, weight_decay=1e-4)

x_dummy = tf.random.uniform([16, 32, 32, 3], 0.0, 1.0)
y_dummy = tf.random.uniform([16], 0, 100, dtype=tf.int32)

y_one_hot = tf.one_hot(y_dummy, depth=100)
y_smooth = y_one_hot * 0.9 + 0.001

x_mixed, y_mixed = apply_cutmix(x_dummy, y_smooth)
x_norm = (x_mixed - MEAN) / STD

with tf.GradientTape() as tape:
    out = model(x_norm, training=True)
    probs = out["probabilities"]
    loss = -tf.reduce_sum(y_mixed * tf.math.log(tf.clip_by_value(probs, 1e-12, 1.0))) / 16.0

grads = tape.gradient(loss, model.trainable_variables)
opt.apply_gradients(model.trainable_variables, grads)

print(f"Test step OK! Loss: {float(loss):.4f}")
