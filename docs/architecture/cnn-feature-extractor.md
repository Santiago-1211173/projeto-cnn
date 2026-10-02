# CNN Feature Extractors

> Part of the [Active Episodic Memory Management via Reinforcement Learning for Robust CNN Inference on Out-of-Distribution Data](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/README.md) documentation.  
> Parent: [System Architecture](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/docs/architecture/README.md) | Up: [System Architecture](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/docs/architecture/README.md)

---

## 1. Overview and Invariant Latent Contract

The parametric cortical backbone of the system extracts semantic visual representations from raw pixel arrays and emits nominal classification predictions. To evaluate semiparametric active memory management across distinct complexity regimes, the repository implements two specialized vision backbones:
1. **MNIST Feature Extractor (`RawModel` in `src/models/custom_cnn.py`):** A custom 4-layer convolutional neural network built exclusively with low-level TensorFlow primitives (`tf.Module`) without high-level Keras abstractions.
2. **CIFAR-10 Feature Extractor (`RawModelCIFAR10` in `src/cifar10/model.py`):** An upgraded ResNet-9 architecture incorporating residual shortcut connections, Batch Normalization, and Global Average Pooling.

### The Invariant 128D Latent Contract
Despite differing input dimensions ($28 \times 28 \times 1$ vs. $32 \times 32 \times 3$) and model capacities (225k vs. 6.57M parameters), **both architectures strictly conform to an identical architectural contract**:
$$\Phi: \mathcal{X} \to \left(\mathbb{R}^{128}, \Delta^9\right)$$
- **Latent Bottleneck:** Penultimate feature vector $z \in \mathbb{R}^{128}$ after ReLU non-linearity.
- **Classification Head:** Linear projection yielding 10-class Softmax posterior probabilities $p \in \Delta^9$.

This invariant interface guarantees that downstream components—including the $k$-NN episodic memory buffer ([`KNNBanditAgent128D`](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/src/models/knn_bandit_agent.py)), the 5D state representation of the RL agent ([`RLAgent`](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/src/models/rl_agent.py)), and the Curriculum Learning reward manager ([`RewardManager`](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/src/models/reward_manager.py))—operate identically across datasets without modification.

---

## 2. MNIST Feature Extractor: Custom 4-Layer CNN

### 2.1. Layer-by-Layer Architecture Specification

The MNIST feature extractor processes $28 \times 28 \times 1$ grayscale inputs through two convolutional stages, spatial max-pooling reductions, a vector flattening transformation, a 128D bottleneck dense layer, and a 10-unit classification head.

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

- **Total Learnable Parameters:** **225,034 parameters**.
- **Weight Storage Footprint:** $\approx 900.1\text{ KB}$ (`float32`).
- **Nominal Validation Accuracy:** $> 98.7\%$ on clean MNIST test set.

---

## 3. CIFAR-10 Feature Extractor: ResNet-9 Backbone

### 3.1. Purpose and Architecture Motivation
Natural RGB images ($32 \times 32 \times 3$) exhibit high intra-class textural variation, multi-modal color distributions, and complex background scenes. A basic 4-layer CNN sub-converges on CIFAR-10 (~54% accuracy), collapsing nearest-neighbor clustering in latent space (Jain & Lindsey, ICLR 2018).

To form tight, well-clustered class manifolds necessary for episodic memory retrieval and OOD detection, `RawModelCIFAR10` ([`src/cifar10/model.py`](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/src/cifar10/model.py)) employs an upgraded ResNet-9 architecture with residual building blocks, Batch Normalization, and Global Average Pooling.

### 3.2. Layer-by-Layer Architecture Specification

| # | Stage / Layer | Component / Type | Input Shape | Output Shape | Details | Trainable Parameters |
|:---|:---|:---|:---|:---|:---|:---|
| **0** | **Input Image** | Input | $32 \times 32 \times 3$ | $32 \times 32 \times 3$ | Normalized RGB pixel tensor | 0 |
| **1** | **Prep Conv** | `Conv2DLayer` + `BatchNorm2D` | $32 \times 32 \times 3$ | $32 \times 32 \times 64$ | Conv $3 \times 3$, stride 1, padding 'SAME' | 1,792 + 128 |
| **2** | **Stage 1 ResBlock** | `ResidualBlock` | $32 \times 32 \times 64$ | $32 \times 32 \times 64$ | 2x Conv(3x3, 64) + BN + ReLU + Identity | 73,856 + 256 |
| **3** | **Pool 1** | `MaxPool2DLayer` | $32 \times 32 \times 64$ | $16 \times 16 \times 64$ | Pool $2 \times 2$, stride 2, padding 'SAME' | 0 |
| **4** | **Stage 2 ResBlock** | `ResidualBlock` (Projection) | $16 \times 16 \times 64$ | $16 \times 16 \times 128$ | Conv(3x3, 128) + 1x1 Shortcut | 221,440 + 512 |
| **5** | **Pool 2** | `MaxPool2DLayer` | $16 \times 16 \times 128$ | $8 \times 8 \times 128$ | Pool $2 \times 2$, stride 2, padding 'SAME' | 0 |
| **6** | **Stage 3 ResBlock** | `ResidualBlock` (Projection) | $8 \times 8 \times 128$ | $8 \times 8 \times 256$ | Conv(3x3, 256) + 1x1 Shortcut | 885,248 + 1,024 |
| **7** | **Pool 3** | `MaxPool2DLayer` | $8 \times 8 \times 256$ | $4 \times 4 \times 256$ | Pool $2 \times 2$, stride 2, padding 'SAME' | 0 |
| **8** | **Stage 4 ResBlock** | `ResidualBlock` | $4 \times 4 \times 256$ | $4 \times 4 \times 256$ | 2x Conv(3x3, 256) + BN + ReLU + Identity | 1,180,160 + 1,024 |
| **9** | **Global Avg Pool** | `GlobalAvgPool2DLayer` | $4 \times 4 \times 256$ | $256$ | Spatial mean reduction across $H \times W$ | 0 |
| **10** | **Latent Bottleneck** | `DenseLayer` + ReLU | $256$ | $128$ | Invariant latent bottleneck projection | 32,896 |
| **11** | **Classifier Head** | `DenseLayer` + Softmax | $128$ | $10$ | 10-class categorical logits projection | 1,290 |

- **Total Learnable Parameters:** **6,568,394 parameters**.
- **Weight Storage Footprint:** $\approx 26.27\text{ MB}$ (`float32`).
- **Nominal Test Accuracy Achieved:** **91.18%** on clean CIFAR-10 test set.

---

## 4. Weight Initialization and Optimization Strategies

### 4.1. Weight Initialization
- **Convolutional Kernels (`Conv2DLayer`):** Initialized using **He Normal** (Kaiming Normal) initialization:
  $$W \sim \mathcal{N}\left(0, \sqrt{\frac{2}{k_h \cdot k_w \cdot c_{\text{in}}}}\right)$$
  Preserves activation variance across repeated ReLU non-linearities and deep residual blocks.
- **Dense Bottleneck & Classifier (`DenseLayer`):** Initialized with **Glorot Uniform** (Xavier Uniform):
  $$W \sim \mathcal{U}\left(-\sqrt{\frac{6}{d_{\text{in}} + d_{\text{out}}}}, +\sqrt{\frac{6}{d_{\text{in}} + d_{\text{out}}}}\right)$$
- **Batch Normalization (`BatchNorm2DLayer`):** $\gamma$ initialized to $1.0$, $\beta$ to $0.0$; running mean to $0.0$, running variance to $1.0$ (momentum $\mu = 0.9$, $\epsilon = 10^{-5}$).

### 4.2. Training Optimization Protocols

| Parameter | MNIST Training (`scripts/train_cnn.py`) | CIFAR-10 Training (`scripts/cifar10/train_cnn.py`) |
|:---|:---|:---|
| **Optimizer** | Custom `SGD` with `assign_sub` | Pure TensorFlow `Adam` with decoupled weight decay |
| **Learning Rate** | $\eta = 0.05$ (constant) | $\eta = 0.001$ with Cosine Annealing decay |
| **Weight Decay** | None | $\lambda = 10^{-4}$ (applied only to 2D/4D weight kernels) |
| **Batch Size** | 128 | 128 |
| **Epochs** | 10 | 25 |
| **Data Augmentation** | None (standard raw digits) | Random Horizontal Flip + Random Crop ($32 \times 32$, pad 4) |
| **Convergence Milestone** | $> 98.5\%$ validation accuracy | $> 90.0\%$ test accuracy (Achieved: **91.18%**) |

---

## 5. Architectural Justification of the 128D Latent Bottleneck

The decision to standardize both models on a **128-dimensional bottleneck** is mathematically and operationally governed by four principles:

1. **Information Bottleneck Principle:**  
   In both networks, spatial features ($1,600\text{D}$ for MNIST, $256\text{D}$ post-GAP for CIFAR-10) are projected into $128\text{D}$. This optimal compression factor forces the representation to discard high-frequency spatial noise and background artifacts while preserving morphological and semantic invariants.

2. **Mitigating Metric Collapse in $k$-NN Retrieval:**  
   In ultra-high dimensions ($D > 500$), the ratio of distances between the nearest and farthest neighbors converges to 1 ($\lim_{D \to \infty} \frac{d_{\max} - d_{\min}}{d_{\min}} \to 0$). At $D=128$, the Euclidean metric remains discriminative, enabling robust nearest-neighbor retrieval within sub-millisecond query budgets.

3. **Strict $O(1)$ RAM Bounds for Edge Microcontrollers:**  
   A pre-allocated buffer of $N=5,000$ exemplars in $128\text{D}$ occupies:
   $$5,000 \times 128 \times 4\text{ bytes} \approx 2.56\text{ MB}$$
   This deterministic footprint permits the entire episodic memory buffer to reside permanently in SRAM/L3 cache on resource-constrained Edge AI accelerators (e.g., NVIDIA Jetson, ARM Cortex-A).

4. **Hardware Alignment with CUDA Warp Architecture:**  
   $128$ is an exact multiple of the 32-thread CUDA warp size. Memory controllers achieve 100% memory coalescing during vectorized distance evaluations, maximizing memory bus utilization.

---

## 6. Code Usage Example: Dual Backbone Inference

```python
import tensorflow as tf
from src.models.custom_cnn import RawModel as MNISTBackbone
from src.cifar10.model import RawModelCIFAR10 as CIFAR10Backbone

# 1. MNIST Forward Pass
mnist_model = MNISTBackbone()
x_mnist = tf.random.uniform((1, 28, 28, 1), dtype=tf.float32)
out_mnist = mnist_model(x_mnist)
print("MNIST Latent Shape:", out_mnist["latent_features"].shape)  # (1, 128)
print("MNIST Probs Shape :", out_mnist["probabilities"].shape)    # (1, 10)

# 2. CIFAR-10 Forward Pass
cifar_model = CIFAR10Backbone()
x_cifar = tf.random.uniform((1, 32, 32, 3), dtype=tf.float32)
out_cifar = cifar_model(x_cifar, training=False)
print("CIFAR Latent Shape:", out_cifar["latent_features"].shape)  # (1, 128)
print("CIFAR Probs Shape :", out_cifar["probabilities"].shape)    # (1, 10)
```

---

**Navigation:**
- Previous: [System Architecture](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/docs/architecture/README.md)
- Up: [System Architecture](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/docs/architecture/README.md)
- Next: [Out-of-Distribution Detection](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/docs/architecture/ood-detection.md)

---

Licensed under the GNU General Public License v3.0. See [LICENSE](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/LICENSE) for details.
