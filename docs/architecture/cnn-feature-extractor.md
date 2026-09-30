# CNN Feature Extractor

> Part of the [Active Episodic Memory Management via Reinforcement Learning for Robust CNN Inference on Out-of-Distribution Data](../../README.md) documentation.
> Parent: [System Architecture](README.md) | Up: [System Architecture](README.md)

---

## 1. Overview and Dual Purpose

The parametric backbone of the vision pipeline is implemented by the `RawModel` class defined in [`src/models/custom_cnn.py`](../../src/models/custom_cnn.py). Rather than relying on standard black-box high-level Keras classes, the network is built entirely from low-level TensorFlow primitives (`tf.Module`) defined in [`src/scratch/layers.py`](../../src/scratch/layers.py), [`src/scratch/activations.py`](../../src/scratch/activations.py), and [`src/scratch/optimizers.py`](../../src/scratch/optimizers.py).

The CNN fulfills two distinct roles in the edge architecture:
1. **Primary Parametric Classifier:** Directly categorizes clean, in-distribution input images into one of 10 digit classes ($0 \dots 9$) with minimal latency ($< 0.5$ ms).
2. **Visual Feature Extractor (Cortical Representation):** Projects 2D spatial pixel arrays ($28 \times 28 \times 1$) into a compact, semantically dense **128-dimensional latent vector** ($z \in \mathbb{R}^{128}$). This latent representation acts as the universal sensory interface for the downstream Mahalanobis++ OOD detector and the $k$-NN episodic memory buffer.

---

## 2. Layer-by-Layer Architecture Specification

The feature extractor consists of two convolutional stages (each followed by a spatial max-pooling reduction), a vector flattening transformation, a 128D latent bottleneck layer, and a final 10-unit linear classification head.

| # | Layer / Operation | Type | Input Shape | Output Shape | Mathematical Formulation | Trainable Parameters |
|:---|:---|:---|:---|:---|:---|:---|
| **0** | **Input Image** | Input | $28 \times 28 \times 1$ | $28 \times 28 \times 1$ | Grayscale pixel intensities normalized to $[0.0, 1.0]$ | 0 |
| **1** | **`conv1`** | Conv2D | $28 \times 28 \times 1$ | $26 \times 26 \times 32$ | 32 filters of $3 \times 3$, stride 1, padding 'VALID': $Y = X * W + b$ | $3 \times 3 \times 1 \times 32 + 32 =$ **320** |
| **-** | *ReLU* | Activation | $26 \times 26 \times 32$ | $26 \times 26 \times 32$ | $\max(0, x)$ | 0 |
| **2** | **`pool1`** | MaxPool2D | $26 \times 26 \times 32$ | $13 \times 13 \times 32$ | Window $2 \times 2$, stride 2, padding 'VALID' | 0 |
| **3** | **`conv2`** | Conv2D | $13 \times 13 \times 32$ | $11 \times 11 \times 64$ | 64 filters of $3 \times 3$, stride 1, padding 'VALID': $Y = X * W + b$ | $3 \times 3 \times 32 \times 64 + 64 =$ **18,496** |
| **-** | *ReLU* | Activation | $11 \times 11 \times 64$ | $11 \times 11 \times 64$ | $\max(0, x)$ | 0 |
| **4** | **`pool2`** | MaxPool2D | $11 \times 11 \times 64$ | $5 \times 5 \times 64$ | Window $2 \times 2$, stride 2, padding 'VALID' | 0 |
| **5** | **`flatten`** | Flatten | $5 \times 5 \times 64$ | $1600$ | Vector reshaping: $\mathbb{R}^{5 \times 5 \times 64} \to \mathbb{R}^{1600}$ | 0 |
| **6** | **`latent_dense`** | Dense | $1600$ | $128$ | Linear projection: $z_{\text{raw}} = x_{\text{flat}} W + b$ | $1600 \times 128 + 128 =$ **204,928** |
| **-** | *ReLU* | Activation | $128$ | $128$ | Latent vector: $z = \max(0, z_{\text{raw}})$ | 0 |
| **7** | **`classifier_dense`** | Dense | $128$ | $10$ | Logits projection: $l = z W + b$ | $128 \times 10 + 10 =$ **1,290** |
| **-** | *Softmax* | Activation | $10$ | $10$ | $p_i = \frac{\exp(l_i - \max(l))}{\sum_j \exp(l_j - \max(l))}$ | 0 |

### Parameter Summary
- **Total Learnable Parameters:** **225,034 parameters** (weights and biases).
- **Weight Tensor Storage:** $\approx 900.1$ KB in single-precision floating point (`float32`), allowing the entire model to reside in the L2/L3 cache of modern edge GPUs and microcontrollers.

---

## 3. Weight Initialization Strategies

To ensure numerical stability and prevent gradient vanishing or explosion during training from scratch:

1. **Convolutional Layers (`Conv2DLayer`):** Initialized using **He Normal** (Kaiming Normal) initialization:
   $$W \sim \mathcal{N}\left(0, \sqrt{\frac{2}{k_h \cdot k_w \cdot c_{\text{in}}}}\right)$$
   This initialization matches the variance of activations following rectified linear unit (ReLU) transformations, preventing signal attenuation across spatial layers.
2. **Dense Layers (`DenseLayer`):** Initialized using **Glorot Uniform** (Xavier Uniform) initialization:
   $$W \sim \mathcal{U}\left(-\sqrt{\frac{6}{d_{\text{in}} + d_{\text{out}}}}, +\sqrt{\frac{6}{d_{\text{in}} + d_{\text{out}}}}\right)$$
   This bounds activation variance entering the classifier head and balances forward and backward signal propagation.
3. **Biases:** All bias tensors are initialized strictly to zero ($b = \mathbf{0}$).

---

## 4. Training Configuration and Optimization Details

The CNN is trained offline using [`scripts/train_cnn.py`](../../scripts/train_cnn.py) on the standard MNIST training split (60,000 samples) with the following specifications:

- **Optimizer:** Custom Stochastic Gradient Descent (`SGD` in [`src/scratch/optimizers.py`](../../src/scratch/optimizers.py)) executing in-place tensor subtractions via TensorFlow's atomic `assign_sub` operation.
- **Learning Rate:** $\eta = 0.01$ (constant, without scheduler).
- **Batch Size:** $B = 256$ samples per step.
- **Epochs:** $E = 10$ full dataset sweeps.
- **Loss Function:** Multi-GPU Categorical Cross-Entropy (`src/scratch/losses.py`):
  $$\mathcal{L} = -\frac{1}{B_{\text{global}}} \sum_{i=1}^{B} \sum_{c=1}^{C} y_{i, c} \ln(\hat{p}_{i, c} + \epsilon)$$
  where $\epsilon = 10^{-15}$ avoids numerical undefined values, and predictions are normalized by the global batch size.
- **Validation Accuracy Achieved:** $> 98.7\%$ on clean test images.

---

## 5. Architectural Justification of the 128D Latent Bottleneck

The design decision to fix the penultimate dense representation at **128 dimensions** is critical to the stability and throughput of the hybrid architecture. It is governed by four principles:

### 5.1. The Information Bottleneck Principle
The flattened feature map from `pool2` contains $5 \times 5 \times 64 = 1,600$ dimensions. A direct projection from 1,600 dimensions to 10 class logits creates an abrupt dimensional collapse, forcing early layers to overfit to pixel-level high-frequency noise. Reducing 1,600 spatial dimensions to 128 (a $12.5\times$ compression factor) creates an optimal information bottleneck: it discards idiosyncratic pixel noise while retaining essential morphological invariants (loops, strokes, and intersections).

### 5.2. Mitigating the Curse of Dimensionality in Episodic Memory
When incoming images exhibit noise or drift, control is handed to the $k$-NN episodic memory buffer. In ultra-high-dimensional spaces ($D > 500$), Euclidean distances suffer from metric collapse: the ratio between the distance to the nearest neighbor and the distance to the farthest neighbor approaches 1:
$$\lim_{D \to \infty} \frac{d_{\max} - d_{\min}}{d_{\min}} \to 0$$
At 128 dimensions, the Euclidean metric remains geometrically discriminative, allowing spatial clustering into distinct class manifolds while enabling sub-millisecond neighbor queries.

### 5.3. Regularization and Overfitting Avoidance
- **Under-dimensioning ($D < 32$):** Compressing representations to 8 or 16 dimensions causes manifold overlap between visually similar classes (e.g., digits 3, 5, and 8), degrading classification accuracy.
- **Over-dimensioning ($D > 512$):** Allocating 512 or 1024 dimensions allows the network to memorize noisy sensor artifacts, increasing memory footprints by $4\times$ to $8\times$.
- **Optimal Balance ($D = 128$):** Provides the ideal representational capacity for handwritten digit clustering while maintaining minimal buffer size ($5,000 \times 128 \times 4 \text{ bytes} \approx 2.56 \text{ MB}$).

### 5.4. Hardware Alignment and CUDA Warps
On modern NVIDIA tensor-core architectures (such as the NVIDIA L40S used in this project), memory controllers and warp schedulers access memory in 32-thread alignments. Matrix dimensions that are exact multiples of 32, 64, or 128 maximize memory coalescing, minimize cache line thrashing, and achieve peak computational throughput.

---

## 6. Code Usage Example: Extracting Latent Vectors

The snippet below illustrates how to load the model checkpoint, perform a forward inference pass, and extract the 128D latent vector:

```python
# src/models/custom_cnn.py
import tensorflow as tf
import numpy as np
from src.models.custom_cnn import RawModel
from src.config import CHECKPOINT_DIR

# 1. Instantiate from-scratch model
model = RawModel(name="custom_cnn_inference")

# 2. Restore pre-trained weights from checkpoint
checkpoint = tf.train.Checkpoint(model=model)
latest_ckpt = tf.train.latest_checkpoint(CHECKPOINT_DIR)
if latest_ckpt:
    checkpoint.restore(latest_ckpt).expect_partial()
    print(f"Restored weights from: {latest_ckpt}")

# 3. Prepare dummy batch (N=2, 28x28x1, float32)
dummy_images = np.random.rand(2, 28, 28, 1).astype(np.float32)
tensor_input = tf.convert_to_tensor(dummy_images)

# 4. Forward execution
outputs = model(tensor_input)

# 5. Access latent vectors and class probabilities
latent_vectors = outputs["latent_features"].numpy()  # Shape: (2, 128)
class_probs = outputs["probabilities"].numpy()       # Shape: (2, 10)

print(f"Latent feature shape: {latent_vectors.shape}")  # (2, 128)
print(f"Probability distribution shape: {class_probs.shape}")  # (2, 10)
print(f"Predicted class: {np.argmax(class_probs, axis=1)}")
```

---

**Navigation:**
- Previous: [System Architecture](README.md)
- Up: [System Architecture](README.md)
- Next: [OOD Detection](ood-detection.md)

---
Licensed under the GNU General Public License v3.0. See [LICENSE](../../LICENSE) for details.
