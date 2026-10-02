import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import tensorflow as tf

def apply_cutout(x: tf.Tensor, mask_size: int = 8) -> tf.Tensor:
    h, w = 32, 32
    y = tf.random.uniform([], 0, h, dtype=tf.int32)
    x_c = tf.random.uniform([], 0, w, dtype=tf.int32)

    y1 = tf.clip_by_value(y - mask_size // 2, 0, h)
    y2 = tf.clip_by_value(y + mask_size // 2, 0, h)
    x1 = tf.clip_by_value(x_c - mask_size // 2, 0, w)
    x2 = tf.clip_by_value(x_c + mask_size // 2, 0, w)

    grid_y = tf.range(h)[:, tf.newaxis, tf.newaxis]
    grid_x = tf.range(w)[tf.newaxis, :, tf.newaxis]
    in_mask = (grid_y >= y1) & (grid_y < y2) & (grid_x >= x1) & (grid_x < x2)
    mask = tf.where(in_mask, tf.zeros([h, w, 1], dtype=tf.float32), tf.ones([h, w, 1], dtype=tf.float32))
    return x * mask

img = tf.ones([32, 32, 3], dtype=tf.float32)
out = apply_cutout(img, mask_size=8)
print("Cutout sum (should be less than 32*32*3 = 3072):", float(tf.reduce_sum(out)))
assert float(tf.reduce_sum(out)) < 3072.0
print("Cutout test passed!")
