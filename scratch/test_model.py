import os
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import tensorflow as tf
import numpy as np
from src.cifar10.model import RawModelCIFAR10
from src.cifar10.optimizers import Adam

m = RawModelCIFAR10()
opt = Adam()

x = tf.random.normal([4, 32, 32, 3])
y = tf.constant([0, 1, 2, 3], dtype=tf.int32)

with tf.GradientTape() as tape:
    out = m(x, training=True)
    probs = out["probabilities"]
    loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y, probs))

grads = tape.gradient(loss, m.trainable_variables)
opt.apply_gradients(m.trainable_variables, grads)

out_infer = m(x, training=False)
print("Forward/backward test passed!")
print("Latent shape:", out_infer["latent_features"].shape)
print("Probs shape:", out_infer["probabilities"].shape)
assert out_infer["latent_features"].shape == (4, 128)
assert out_infer["probabilities"].shape == (4, 10)
print("Contract verified!")
