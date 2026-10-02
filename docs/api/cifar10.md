# CIFAR-10 Dedicated Module API Reference

> Part of the [Active Episodic Memory Management via Reinforcement Learning for Robust CNN Inference on Out-of-Distribution Data](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/README.md) documentation.  
> Parent: [API Reference Index](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/docs/api/README.md) | Up: [Documentation Index](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/docs/README.md)

---

## 1. Module Overview

The `src/cifar10` package provides dedicated neural network layers, an upgraded ResNet-9 vision backbone, a pure TensorFlow Adam/AdamW optimizer with decoupled weight decay, and a Dual Uncertainty Out-of-Distribution (OOD) Arbiter tailored for high-dimensional natural color images ($32 \times 32 \times 3$).

### Core Design Contract
All models in this module preserve the **invariant 128D latent bottleneck contract**:
- Input: Visual tensor $x \in \mathbb{R}^{B \times 32 \times 32 \times 3}$.
- Output: Dictionary containing:
  - `"latent_features"`: $z \in \mathbb{R}^{B \times 128}$ (ReLU bottleneck activations).
  - `"probabilities"`: $p \in \Delta^9$ (10-class Softmax posterior distribution).

---

## 2. `RawModelCIFAR10`

**Module:** [`src.cifar10.model`](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/src/cifar10/model.py)  
**Inherits:** `tf.Module`

### 2.1. Class Description
An upgraded ResNet-9 convolutional neural network designed to achieve $\ge 90\%$ classification accuracy on CIFAR-10 while preserving the exact 128-dimensional latent space contract required by downstream episodic memory and reinforcement learning agents.

### 2.2. Constructor Signature
```python
RawModelCIFAR10(name: str = "custom_cnn_cifar10")
```

#### Structural Stages and Layers:
- `self.prep_conv`: `Conv2DLayer(in_channels=3, out_channels=64, kernel_size=3, padding='SAME')`
- `self.prep_bn`: `BatchNorm2DLayer(num_features=64)`
- `self.res1`: `ResidualBlock(in_channels=64, out_channels=64, stride=1)`
- `self.pool1`: `MaxPool2DLayer(pool_size=2, stride=2, padding='SAME')` (Output: $16 \times 16 \times 64$)
- `self.res2`: `ResidualBlock(in_channels=64, out_channels=128, stride=1)` (Output: $16 \times 16 \times 128$)
- `self.pool2`: `MaxPool2DLayer(pool_size=2, stride=2, padding='SAME')` (Output: $8 \times 8 \times 128$)
- `self.res3`: `ResidualBlock(in_channels=128, out_channels=256, stride=1)` (Output: $8 \times 8 \times 256$)
- `self.pool3`: `MaxPool2DLayer(pool_size=2, stride=2, padding='SAME')` (Output: $4 \times 4 \times 256$)
- `self.res4`: `ResidualBlock(in_channels=256, out_channels=256, stride=1)` (Output: $4 \times 4 \times 256$)
- `self.gap`: `GlobalAvgPool2DLayer()` (Output: $256\text{D}$)
- `self.latent_dense`: `DenseLayer(in_features=256, out_features=128)` (Output: $128\text{D}$)
- `self.classifier_dense`: `DenseLayer(in_features=128, out_features=10)` (Output: $10\text{D}$)

### 2.3. Method: `__call__`
```python
def __call__(self, x: tf.Tensor, training: bool = True) -> Dict[str, tf.Tensor]:
```
Executes the forward inference or training pass.

- **Parameters:**
  - `x`: Input tensor of shape `(batch_size, 32, 32, 3)` with pixel values normalized to standard normal or $[0, 1]$.
  - `training`: Boolean flag indicating whether Batch Normalization updates running statistics (`True`) or evaluates frozen population statistics (`False`).
- **Returns:**
  - `Dict[str, tf.Tensor]` with keys:
    - `"latent_features"`: Tensor of shape `(batch_size, 128)` after penultimate ReLU non-linearity.
    - `"probabilities"`: Tensor of shape `(batch_size, 10)` representing normalized Softmax class likelihoods.

---

## 3. `DualUncertaintyArbiter`

**Module:** [`src.cifar10.ood_arbiter`](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/src/cifar10/ood_arbiter.py)

### 3.1. Class Description
Out-of-Distribution Arbiter combining unnormalized Mahalanobis distance in the 128D latent space with predictive Shannon entropy over Softmax probability vectors (Kaur et al., ICML 2021; Nguyen, 2026).

$$\text{Decision: Reject if } d_M(z) > \tau_M \quad \lor \quad H(p) > \tau_H$$

### 3.2. Constructor Signature
```python
DualUncertaintyArbiter(n_classes: int = 10, latent_dim: int = 128)
```

- **Parameters:**
  - `n_classes`: Integer count of discrete nominal classes (default: `10`).
  - `latent_dim`: Latent feature vector dimension (default: `128`).

### 3.3. Public Methods

#### `fit`
```python
def fit(self, latent_features: np.ndarray, probabilities: np.ndarray, labels: np.ndarray, percentile: float = 95.0) -> None:
```
Calculates class-conditional centroids $\mu_c$ and Ledoit-Wolf precision matrices $\Sigma_c^{-1}$ on unnormalized features, calibrating thresholds $\tau_M$ and $\tau_H$ at the specified percentile of clean data.

#### `compute_mahalanobis_batch`
```python
def compute_mahalanobis_batch(self, latent_features: np.ndarray) -> np.ndarray:
```
Vectorized computation evaluating the minimum Mahalanobis distance across all $C=10$ class profiles for a batch of latent vectors:
$$d_M(z) = \min_{c \in \{0, \dots, 9\}} \sqrt{(z - \mu_c)^T \Sigma_c^{-1} (z - \mu_c)}$$
- **Input:** Array of shape `(N, 128)`.
- **Returns:** Array of shape `(N,)` with minimum distances.

#### `compute_entropy_batch`
```python
def compute_entropy_batch(self, probabilities: np.ndarray) -> np.ndarray:
```
Computes predictive Shannon entropy in nats:
$$H(p) = -\sum_{c=0}^{9} p_c \ln(p_c + \epsilon)$$
- **Input:** Array of shape `(N, 10)`.
- **Returns:** Array of shape `(N,)` with non-negative Shannon entropies.

#### `predict_batch`
```python
def predict_batch(self, latent_features: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
```
Evaluates dual uncertainty routing condition:
$$\text{is\_ood} = (d_M > \tau_M) \lor (H > \tau_H)$$
- **Input:** `latent_features` `(N, 128)`, `probabilities` `(N, 10)`.
- **Returns:** Boolean array of shape `(N,)` where `True` denotes sample routed to episodic memory.

#### `save` / `load`
```python
def save(self, filepath: str) -> None:
def load(self, filepath: str) -> None:
```
Serializes and deserializes class centroids, Ledoit-Wolf precision matrices, and calibrated threshold constants to/from `.npz` format.

---

## 4. Layer Primitives and Residual Blocks

**Module:** [`src.cifar10.layers`](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/src/cifar10/layers.py)

### 4.1. `BatchNorm2DLayer`
Low-level 2D Batch Normalization primitive implemented via `tf.Module`:
- **Constructor:** `BatchNorm2DLayer(num_features: int, epsilon: float = 1e-5, momentum: float = 0.9, name: str = None)`
- **Parameters:**
  - `gamma`: Trainable scale parameter $\gamma \in \mathbb{R}^{C}$ (initialized to 1.0).
  - `beta`: Trainable shift parameter $\beta \in \mathbb{R}^{C}$ (initialized to 0.0).
  - `moving_mean`, `moving_var`: Non-trainable moving statistics with momentum $\mu = 0.9$.
- **Call:** `__call__(x: tf.Tensor, training: bool = True) -> tf.Tensor`

### 4.2. `Conv2DLayer`
TensorFlow 2D Convolution with He Normal weight initialization:
- **Constructor:** `Conv2DLayer(in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: str = 'SAME', name=None)`
- **Variables:** $W \sim \mathcal{N}\left(0, \sqrt{\frac{2}{k^2 c_{\text{in}}}}\right)$, $b = \mathbf{0}$.
- **Call:** `__call__(x: tf.Tensor) -> tf.Tensor`

### 4.3. `ResidualBlock`
Standard Residual Building Block with identity shortcut or $1 \times 1$ projection:
- **Architecture:** $x \to \text{Conv}(3 \times 3) \to \text{BN} \to \text{ReLU} \to \text{Conv}(3 \times 3) \to \text{BN} + \text{Shortcut}(x) \to \text{ReLU}$
- **Constructor:** `ResidualBlock(in_channels: int, out_channels: int, stride: int = 1, name: str = None)`
- **Call:** `__call__(x: tf.Tensor, training: bool = True) -> tf.Tensor`

### 4.4. `DenseLayer`
Linear dense projection with Glorot Uniform initialization:
- **Constructor:** `DenseLayer(in_features: int, out_features: int, name=None)`
- **Call:** `__call__(x: tf.Tensor) -> tf.Tensor`

### 4.5. `GlobalAvgPool2DLayer`
Spatial dimension reduction computing mean across height and width:
- **Call:** `__call__(x: tf.Tensor) -> tf.Tensor` (Reduces $(B, H, W, C) \to (B, C)$).

---

## 5. `Adam` Optimizer with Decoupled Weight Decay

**Module:** [`src.cifar10.optimizers`](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/src/cifar10/optimizers.py)

### 5.1. Class Description
Pure TensorFlow Adam/AdamW implementation from scratch supporting decoupled weight decay (Loshchilov & Hutter, ICLR 2019) and compilation under `@tf.function`.

### 5.2. Constructor Signature
```python
Adam(
    learning_rate: float = 0.001,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-7,
    weight_decay: float = 1e-4
)
```

### 5.3. Method: `apply_gradients`
```python
def apply_gradients(self, variables: List[tf.Variable], gradients: List[tf.Tensor]) -> None:
```
Updates first moment ($m_t$), second moment ($v_t$), applies bias correction, applies decoupled weight decay exclusively to 2D/4D weight kernels, and performs in-place parameter updates via atomic `assign_sub`.

---

## 6. Code Usage Example: End-to-End Pipeline

```python
import numpy as np
import tensorflow as tf
from src.cifar10.model import RawModelCIFAR10
from src.cifar10.ood_arbiter import DualUncertaintyArbiter

# 1. Instantiate ResNet-9 Backbone
model = RawModelCIFAR10()

# 2. Synthetic Batch Forward Pass
batch_x = tf.random.normal((4, 32, 32, 3))
outputs = model(batch_x, training=False)
z = outputs["latent_features"].numpy()    # (4, 128)
probs = outputs["probabilities"].numpy()   # (4, 10)

# 3. Instantiate and Query Dual Uncertainty Arbiter
arbiter = DualUncertaintyArbiter(n_classes=10, latent_dim=128)
arbiter.load("outputs/cifar10/mahalanobis_pp_profiles.npz")

is_ood = arbiter.predict_batch(z, probs)
d_M = arbiter.compute_mahalanobis_batch(z)
entropy = arbiter.compute_entropy_batch(probs)

for i in range(len(is_ood)):
    print(f"Sample {i}: d_M={d_M[i]:.2f} (tau={arbiter.threshold_mahalanobis:.2f}), "
          f"H={entropy[i]:.2f} (tau={arbiter.threshold_entropy:.2f}) -> "
          f"Route: {'Episodic Memory' if is_ood[i] else 'Parametric CNN'}")
```

---

**Navigation:**
- Previous: [API Reference Index](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/docs/api/README.md)
- Up: [API Reference Index](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/docs/api/README.md)
- Next: [Configuration Module API](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/docs/api/config.md)

---

Licensed under the GNU General Public License v3.0. See [LICENSE](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/LICENSE) for details.
