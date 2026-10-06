import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import src.config
import tensorflow as tf
from src.cifar100.model import RawModelCIFAR100
from src.cifar100.optimizers import SGDMomentum

model = RawModelCIFAR100(latent_dim=128)
opt = SGDMomentum(learning_rate=0.1, momentum=0.9, nesterov=True, weight_decay=5e-4)

x = tf.random.normal([16, 32, 32, 3])
y = tf.random.uniform([16], 0, 100, dtype=tf.int32)

with tf.GradientTape() as tape:
    out = model(x, training=True)
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=out["logits"]))

grads = tape.gradient(loss, model.trainable_variables)
opt.apply_gradients(model.trainable_variables, grads)
print(f"SGDMomentum OK! Loss: {float(loss):.4f}")
