import os
import sys
import time
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import src.config
import tensorflow as tf
from src.data.cifar100_loader import load_cifar100_raw
from src.cifar100.model import RawModelCIFAR100
from src.cifar100.optimizers import Adam, SGDMomentum

CIFAR100_MEAN = tf.constant([0.50707516, 0.48654887, 0.44091784], dtype=tf.float32)
CIFAR100_STD = tf.constant([0.26733429, 0.25643846, 0.27615047], dtype=tf.float32)

def apply_cutout_batch(x: tf.Tensor, mask_size: int = 8) -> tf.Tensor:
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

def apply_cutmix_batch(x: tf.Tensor, y_smooth: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
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

def evaluate(model, x_test, y_test, batch_size=500, standardize=True):
    preds = []
    for i in range(0, len(x_test), batch_size):
        batch_x = tf.convert_to_tensor(x_test[i : i + batch_size], dtype=tf.float32)
        if standardize:
            batch_x = (batch_x - CIFAR100_MEAN) / CIFAR100_STD
        out = model(batch_x, training=False)
        preds.append(np.argmax(out["probabilities"].numpy(), axis=1))
    return float(np.mean(np.concatenate(preds) == y_test) * 100.0)

# Quick 1-epoch test
x_train, y_train, x_test, y_test = load_cifar100_raw()
x_train = x_train.astype(np.float32)
y_train = y_train.astype(np.int32)
x_test = x_test.astype(np.float32)
y_test = y_test.astype(np.int32)

batch_size = 128
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

model = RawModelCIFAR100(latent_dim=128)
opt = Adam(learning_rate=0.001, weight_decay=1e-4)
ls_eps = 0.1
cutmix_prob = 0.5
standardize = True

@tf.function
def train_step(x_b: tf.Tensor, y_b: tf.Tensor):
    b = tf.shape(x_b)[0]
    x_aug = tf.image.random_flip_left_right(x_b)
    x_aug = tf.pad(x_aug, [[0, 0], [4, 4], [4, 4], [0, 0]], mode="REFLECT")
    x_aug = tf.image.random_crop(x_aug, [b, 32, 32, 3])

    y_one_hot = tf.one_hot(y_b, depth=100)
    y_smooth = y_one_hot * (1.0 - ls_eps) + (ls_eps / 100.0)

    if tf.random.uniform([]) < cutmix_prob:
        x_aug, y_smooth = apply_cutmix_batch(x_aug, y_smooth)
    else:
        x_aug = apply_cutout_batch(x_aug, mask_size=8)

    if standardize:
        x_aug = (x_aug - CIFAR100_MEAN) / CIFAR100_STD

    with tf.GradientTape() as tape:
        out = model(x_aug, training=True)
        probs = tf.clip_by_value(out["probabilities"], 1e-12, 1.0)
        loss = -tf.reduce_sum(y_smooth * tf.math.log(probs)) / tf.cast(b, tf.float32)

    grads = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(model.trainable_variables, grads)
    preds = tf.argmax(probs, axis=-1, output_type=tf.int32)
    acc = tf.reduce_mean(tf.cast(tf.equal(preds, y_b), tf.float32))
    return loss, acc

print("Running 1 epoch dry-run on full CIFAR-100 (50k images)...")
t0 = time.time()
total_loss = 0.0
for step, (xb, yb) in enumerate(train_ds):
    loss, _ = train_step(xb, yb)
    total_loss += float(loss)
    if step == 0:
        print("  First step complete (tf.function compiled)")
t_epoch = time.time() - t0
test_acc = evaluate(model, x_test, y_test)
print(f"Epoch 1 finished in {t_epoch:.2f}s! Mean Loss: {total_loss / (step + 1):.4f} | Test Acc: {test_acc:.2f}%")
