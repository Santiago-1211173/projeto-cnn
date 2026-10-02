import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import tensorflow as tf
from src.data.loader import load_dataset_raw
from src.cifar10.model import RawModelCIFAR10

print("Running Phase 1 Verification Gate...")

x_test, y_test = load_dataset_raw('cifar10', os.path.join(PROJECT_ROOT, 'data/CIFAR10/raw'), kind='t10k')
x_test = x_test.astype(np.float32) / 255.0

m = RawModelCIFAR10()
ckpt = tf.train.Checkpoint(model=m)
latest_ckpt = tf.train.latest_checkpoint(os.path.join(PROJECT_ROOT, 'outputs/cifar10/checkpoints'))
print("Latest checkpoint:", latest_ckpt)
ckpt.restore(latest_ckpt).expect_partial()

preds = []
all_latents = []
for i in range(0, len(x_test), 500):
    out = m(x_test[i:i+500], training=False)
    preds.append(np.argmax(out['probabilities'].numpy(), axis=1))
    all_latents.append(out['latent_features'].numpy())

concat_preds = np.concatenate(preds)
concat_latents = np.concatenate(all_latents, axis=0)

acc = np.mean(concat_preds == y_test) * 100.0
print(f"Trained CIFAR-10 CNN Accuracy: {acc:.2f}% (Target: >= 75.0%)")
print(f"Latent features shape: {concat_latents.shape} (Target: (10000, 128))")

assert acc >= 75.0, f"Model did not achieve target convergence: {acc:.2f}% < 75.0%"
assert concat_latents.shape == (10000, 128), f"Latent contract violated: {concat_latents.shape}"

print("=" * 60)
print("PHASE 1 VERIFICATION GATE PASSED SUCCESSFULLY!")
print("=" * 60)
