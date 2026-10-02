import tensorflow as tf
import time

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

def augment_batch(x: tf.Tensor, y: tf.Tensor):
    x = tf.image.random_flip_left_right(x)
    x = tf.pad(x, [[0, 0], [4, 4], [4, 4], [0, 0]], mode='REFLECT')
    x = tf.image.random_crop(x, [tf.shape(x)[0], 32, 32, 3])
    x = apply_cutout_batch(x, mask_size=8)
    return x, y

batch_x = tf.random.normal([128, 32, 32, 3])
batch_y = tf.zeros([128], dtype=tf.int32)

t0 = time.time()
for _ in range(50):
    _ = augment_batch(batch_x, batch_y)
t1 = time.time()
print(f"50 batches processed in {t1-t0:.4f}s ({(t1-t0)/50*1000:.2f}ms/batch)")
