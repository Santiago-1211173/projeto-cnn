# CNN Feature Extractors

> Part of the [Active Episodic Memory Management via Reinforcement Learning for Robust CNN Inference on Out-of-Distribution Data](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/README.md) documentation.  
> Parent: [System Architecture](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/docs/architecture/README.md) | Up: [System Architecture](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/docs/architecture/README.md)

---

## 1. Overview and Invariant Latent Contract

The parametric cortical backbone of the system extracts semantic visual representations from raw pixel arrays and emits nominal classification predictions. To evaluate semiparametric active memory management across distinct complexity regimes, the repository implements three specialized vision backbones:
1. **MNIST Feature Extractor ([`RawModel`](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/src/models/custom_cnn.py)):** A custom 4-layer convolutional neural network built exclusively with low-level TensorFlow primitives (`tf.Module`) without high-level Keras abstractions (225,034 parameters).
2. **CIFAR-10 Feature Extractor ([`RawModelCIFAR10`](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/src/cifar10/model.py)):** An upgraded ResNet-9 architecture incorporating residual shortcut connections, Batch Normalization, and Global Average Pooling (6,568,394 parameters), achieving 91.18% nominal clean test accuracy.
3. **CIFAR-100 Feature Extractor ([`RawModelCIFAR100V2`](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/src/cifar100/model.py)):** An upgraded 18-layer deep residual network (ResNet-18 V2) structured across 4 hierarchical stages (8 residual blocks) featuring learned strided convolutions and a Batch Normalization bottleneck (11,250,532 parameters), achieving 74.27% nominal clean test accuracy.

### The Invariant 128D Latent Contract
Despite differing input dimensions ($28 \times 28 \times 1$ vs. $32 \times 32 \times 3$) and model capacities (225k, 6.57M, and 11.25M parameters), **all three architectures strictly conform to an identical architectural contract**:
$$\Phi: \mathcal{X} \to \left(\mathbb{R}^{128}, \Delta^{C-1}\right), \quad C \in \{10, 100\}$$
- **Latent Bottleneck:** Penultimate feature vector $z \in \mathbb{R}^{128}$ after ReLU non-linearity (and dedicated Batch Normalization in ResNet-18 V2).
- **Classification Head:** Linear projection yielding class Softmax posterior probabilities $p \in \Delta^9$ (MNIST, CIFAR-10) or $p \in \Delta^{99}$ (CIFAR-100).

This invariant interface guarantees that downstream components—including the $k$-NN episodic memory buffer ([`KNNBanditAgent128D`](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/src/models/knn_bandit_agent.py)), the 5D state representation of the RL agent ([`RLAgent`](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/src/models/rl_agent.py)), and the Curriculum Learning reward manager ([`RewardManager`](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/src/models/reward_manager.py))—operate identically across datasets without structural modification.

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

## 4. CIFAR-100 Feature Extractor: ResNet-18 V2 Backbone

### 4.1. Purpose and Architecture Motivation
Fine-grained 100-class natural image classification ($32 \times 32 \times 3$) presents exceptional representational challenges: high class-cardinality entropy ($H_{\max} = \ln(100) \approx 4.6052\text{ nats}$) and subtle inter-class visual boundaries (such as differentiating between subordinate species of aquatic mammals, insects, or trees). Shallow backbones or networks employing non-learned spatial pooling (such as blind Max-Pooling) induce catastrophic spatial information collapse and feature entanglement on 100-class datasets.

To construct linearly separable, tightly clustered class manifolds required for episodic memory retrieval and dual uncertainty gating, `RawModelCIFAR100V2` ([`src/cifar100/model.py`](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/src/cifar100/model.py)) implements an enhanced 18-layer residual architecture (ResNet-18 V2):
- **4 Progressive Hierarchical Stages:** 8 residual blocks expanding channel capacity from 64 to 512 ($64 \to 128 \to 256 \to 512$).
- **Learned Strided Downsampling:** Replaces blind max-pooling with strided convolutions ($s=2$) in the first block of Stages 2, 3, and 4, preserving nuanced spatial topologies across resolution transitions.
- **Global Average Pooling:** Collapses $4 \times 4 \times 512$ feature volumes into a 512D spatial summary vector without flattening parameter explosion.
- **Normalized Latent Bottleneck:** Projects 512D representations into 128D, followed by dedicated Batch Normalization (`latent_bn`) and ReLU activation, stabilizing latent manifold variance before the 100-class linear classification head.

### 4.2. Layer-by-Layer Architecture Specification

| # | Stage / Layer Group | Component / Sub-Layers | Input Shape | Output Shape | Details and Operations | Trainable Parameters |
|:---|:---|:---|:---|:---|:---|:---|
| **0** | **Input Image** | Input | $32 \times 32 \times 3$ | $32 \times 32 \times 3$ | Standardized RGB pixel tensor | 0 |
| **1** | **Prep Stage** | `prep_conv` + `prep_bn` + ReLU | $32 \times 32 \times 3$ | $32 \times 32 \times 64$ | Conv $3 \times 3$, stride 1, padding 'SAME' | 1,792 + 128 = **1,920** |
| **2** | **Stage 1 (64 Ch)** | 2x `ResidualBlock` ($s=1$) | $32 \times 32 \times 64$ | $32 \times 32 \times 64$ | 4x Conv($3 \times 3, 64$) + BN + ReLU + Identity shortcuts | 2x (73,856 + 256) = **148,224** |
| **3** | **Stage 2 (128 Ch)** | ResBlock 1 ($s=2$) + ResBlock 2 ($s=1$) | $32 \times 32 \times 64$ | $16 \times 16 \times 128$ | Conv($3 \times 3, 128, s=2$) + $1 \times 1$ Shortcut + Conv($3 \times 3, 128$) | 230,528 + 295,680 = **526,208** |
| **4** | **Stage 3 (256 Ch)** | ResBlock 1 ($s=2$) + ResBlock 2 ($s=1$) | $16 \times 16 \times 128$ | $8 \times 8 \times 256$ | Conv($3 \times 3, 256, s=2$) + $1 \times 1$ Shortcut + Conv($3 \times 3, 256$) | 919,808 + 1,181,184 = **2,100,992** |
| **5** | **Stage 4 (512 Ch)** | ResBlock 1 ($s=2$) + ResBlock 2 ($s=1$) | $8 \times 8 \times 256$ | $4 \times 4 \times 512$ | Conv($3 \times 3, 512, s=2$) + $1 \times 1$ Shortcut + Conv($3 \times 3, 512$) | 3,674,624 + 4,721,664 = **8,396,288** |
| **6** | **Global Avg Pool** | `GlobalAvgPool2DLayer` | $4 \times 4 \times 512$ | $512$ | Spatial mean reduction across $H \times W$ | 0 |
| **7** | **Latent Bottleneck** | `DenseLayer` + `BatchNorm2D` + ReLU | $512$ | $128$ | Dense ($512 \to 128$) + BN(128) + ReLU activation | 65,664 + 256 = **65,920** |
| **8** | **Classifier Head** | `DenseLayer` + Softmax | $128$ | $100$ | 100-class categorical logits projection | 12,800 + 100 = **12,900** |

- **Total Learnable Parameters:** **11,250,532 parameters** (11,252,452 trainable variables, 11,262,308 total variables including moving BN stats).
- **Weight Storage Footprint:** $\approx 45.0\text{ MB}$ (`float32`).
- **Nominal Test Accuracy Achieved:** **74.27% Top-1 accuracy** on clean CIFAR-100 test set (150 epochs).

---

## 5. Weight Initialization and Optimization Strategies

### 5.1. Weight Initialization
- **Convolutional Kernels (`Conv2DLayer`):** Initialized using **He Normal** (Kaiming Normal) initialization:
  $$W \sim \mathcal{N}\left(0, \sqrt{\frac{2}{k_h \cdot k_w \cdot c_{\text{in}}}}\right)$$
  Preserves activation variance across repeated ReLU non-linearities and deep residual blocks.
- **Dense Bottleneck & Classifier (`DenseLayer`):** Initialized with **Glorot Uniform** (Xavier Uniform):
  $$W \sim \mathcal{U}\left(-\sqrt{\frac{6}{d_{\text{in}} + d_{\text{out}}}}, +\sqrt{\frac{6}{d_{\text{in}} + d_{\text{out}}}}\right)$$
- **Batch Normalization (`BatchNorm2DLayer`):** $\gamma$ initialized to $1.0$, $\beta$ to $0.0$; running mean to $0.0$, running variance to $1.0$ (momentum $\mu = 0.9$, $\epsilon = 10^{-5}$).

### 5.2. Training Optimization Protocols

| Parameter | MNIST Training (`scripts/train_cnn.py`) | CIFAR-10 Training (`scripts/cifar10/train_cnn.py`) | CIFAR-100 Training (`scripts/cifar100/train_cnn.py`) |
|:---|:---|:---|:---|
| **Backbone Model** | `RawModel` (4-Layer CNN) | `RawModelCIFAR10` (ResNet-9) | `RawModelCIFAR100V2` (ResNet-18 V2) |
| **Optimizer** | Custom `SGD` with `assign_sub` | Pure TensorFlow `Adam` with decoupled weight decay | Custom `AdamW` (`SGDMomentum` option supported) |
| **Learning Rate** | $\eta = 0.05$ (constant) | $\eta = 0.001$ with Cosine Annealing decay | $\eta = 0.001$ with Cosine Annealing decay |
| **Weight Decay** | None | $\lambda = 10^{-4}$ (applied only to 2D/4D weight kernels) | $\lambda = 10^{-4}$ (applied only to 2D/4D weight kernels) |
| **Batch Size** | 128 | 128 | 128 |
| **Epochs** | 10 | 25 | 150 |
| **Data Augmentation** | None (standard raw digits) | Random Horizontal Flip + Random Crop ($32 \times 32$, pad 4) | Random Crop + Flip + CutMix ($p=0.5$) + Label Smoothing ($0.1$) |
| **Convergence Milestone** | $> 98.5\%$ validation accuracy | $> 90.0\%$ test accuracy (Achieved: **91.18%**) | $> 74.0\%$ test accuracy (Achieved: **74.27%**) |

---

## 6. Architectural Justification of the 128D Latent Bottleneck

The decision to standardize all three backbones on an invariant **128-dimensional bottleneck** is mathematically and operationally governed by four principles:

1. **Information Bottleneck Principle:**  
   In all three networks, high-dimensional intermediate representations ($1,600\text{D}$ for MNIST, $256\text{D}$ post-GAP for CIFAR-10, $512\text{D}$ post-GAP for CIFAR-100) are compressed into $128\text{D}$. This compression factor forces the representations to discard high-frequency noise and background clutter while preserving semantic discriminability.

2. **Mitigating Metric Collapse in $k$-NN Retrieval:**  
   In ultra-high dimensions ($D > 500$), the ratio of distances between the nearest and farthest neighbors converges to 1 ($\lim_{D \to \infty} \frac{d_{\max} - d_{\min}}{d_{\min}} \to 0$). At $D=128$, the Euclidean metric remains discriminative, enabling robust nearest-neighbor retrieval within sub-millisecond query budgets across both 10-class and 100-class settings.

3. **Strict $O(1)$ RAM Bounds for Edge Microcontrollers:**  
   A pre-allocated buffer of $N=5,000$ exemplars in $128\text{D}$ occupies:
   $$5,000 \times 128 \times 4\text{ bytes} \approx 2.56\text{ MB}$$
   This deterministic footprint permits the entire episodic memory buffer to reside permanently in SRAM/L3 cache on resource-constrained Edge AI accelerators (e.g., NVIDIA Jetson, ARM Cortex-A), allocating 500 prototypes per class on 10-class tasks and 50 prototypes per class on CIFAR-100.

4. **Hardware Alignment with CUDA Warp Architecture:**  
   $128$ is an exact multiple of the 32-thread CUDA warp size. Memory controllers achieve 100% memory coalescing during vectorized distance evaluations, maximizing memory bus utilization.

---

## 7. Code Usage Example: Tri-Backbone Inference

```python
import tensorflow as tf
from src.models.custom_cnn import RawModel as MNISTBackbone
from src.cifar10.model import RawModelCIFAR10 as CIFAR10Backbone
from src.cifar100.model import load_cifar100_backbone, RawModelCIFAR100V2

# 1. MNIST Forward Pass
mnist_model = MNISTBackbone()
x_mnist = tf.random.uniform((1, 28, 28, 1), dtype=tf.float32)
out_mnist = mnist_model(x_mnist)
print("MNIST Latent Shape   :", out_mnist["latent_features"].shape)  # (1, 128)
print("MNIST Probs Shape    :", out_mnist["probabilities"].shape)    # (1, 10)

# 2. CIFAR-10 Forward Pass
cifar10_model = CIFAR10Backbone()
x_cifar10 = tf.random.uniform((1, 32, 32, 3), dtype=tf.float32)
out_cifar10 = cifar10_model(x_cifar10, training=False)
print("CIFAR-10 Latent Shape:", out_cifar10["latent_features"].shape)  # (1, 128)
print("CIFAR-10 Probs Shape :", out_cifar10["probabilities"].shape)    # (1, 10)

# 3. CIFAR-100 Forward Pass (Restored ResNet-18 V2 Checkpoint)
cifar100_model, standardize = load_cifar100_backbone("outputs/cifar100/checkpoints")
x_cifar100 = tf.random.uniform((1, 32, 32, 3), dtype=tf.float32)
out_cifar100 = cifar100_model(x_cifar100, training=False)
print("CIFAR-100 Latent Shape:", out_cifar100["latent_features"].shape)  # (1, 128)
print("CIFAR-100 Probs Shape :", out_cifar100["probabilities"].shape)    # (1, 100)
```

---

**Navigation:**
- Previous: [System Architecture](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/docs/architecture/README.md)
- Up: [System Architecture](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/docs/architecture/README.md)
- Next: [Out-of-Distribution Detection](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/docs/architecture/ood-detection.md)

---

Licensed under the GNU General Public License v3.0. See [LICENSE](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/LICENSE) for details.
