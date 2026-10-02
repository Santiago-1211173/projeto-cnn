# Configuration Reference

> Part of the [Active Episodic Memory Management via Reinforcement Learning for Robust CNN Inference on Out-of-Distribution Data](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/README.md) documentation.  
> Parent: [Getting Started Index](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/docs/getting-started/README.md) | Up: [Documentation Index](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/docs/README.md)

---

## 1. Overview and Design Philosophy

The system architecture utilizes a centralized configuration and environment registry located at [`src/config.py`](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/src/config.py), supplemented by dataset-specific configurations in `src/cifar10/` and runtime flags across dedicated scripts in `scripts/cifar10/`.

### Architectural Principles of Configuration
1. **Zero Hardcoded Magic Numbers:** Operational scripts import constants directly from `src.config` or pass typed CLI arguments rather than embedding hidden defaults.
2. **Deterministic Reproducibility:** Global random seeds (`RANDOM_SEED = 42`) are propagated across NumPy, TensorFlow, and PyTorch runtime engines.
3. **Dynamic Multi-Dataset Path Resolution:** File paths resolve dynamically relative to `PROJECT_ROOT`, branching automatically based on dataset targeting (`DATASET=mnist` or `DATASET=cifar10`).
4. **Invariant Shared Contract:** Architectural dimensions shared between models (`LATENT_DIM = 128`, `MEMORY_CAPACITY = 5000`, `RL_STATE_DIM = 5`, `RL_N_ACTIONS = 4`) remain invariant across datasets.

---

## 2. Multi-Dataset Path and Environment Resolution

The repository supports dual datasets via directory isolation and environment variables:

```bash
# Target dataset via environment variable
export DATASET="cifar10"   # Options: "mnist" or "cifar10"
```

### Path Resolution Matrix

| Dataset Target | Raw Data Path | Checkpoints Directory | Metrics and Dashboard Path |
|:---|:---|:---|:---|
| **MNIST (`mnist`)** | `data/MNIST/raw/` | `outputs/mnist/checkpoints/` | `outputs/mnist/` (`eaai_metrics.json`, `eaai_evaluation_dashboard.png`) |
| **CIFAR-10 (`cifar10`)** | `data/CIFAR10/raw/cifar-10-batches-py/` | `outputs/cifar10/checkpoints/` | `outputs/cifar10/` (`eaai_metrics.json`, `eaai_evaluation_dashboard.png`) |

---

## 3. Configuration Parameters Master Table

The table below catalogs primary configuration parameters across MNIST and CIFAR-10 modules:

| Parameter | Default (MNIST) | Default (CIFAR-10) | Python Type | Subsystem / Scope | Functional Description |
|:---|:---:|:---:|:---:|:---|:---|
| `PROJECT_ROOT` | Dynamic `abspath` | Dynamic `abspath` | `str` | System Paths | Absolute path to repository root. |
| `RANDOM_SEED` | `42` | `42` | `int` | General | Master pseudo-random number generator seed. |
| `LATENT_DIM` | `128` | `128` | `int` | Invariant Contract | Bottleneck representation dimensionality $z \in \mathbb{R}^{128}$. |
| `MEMORY_CAPACITY` | `5000` | `5000` | `int` | Episodic Memory | Maximum exemplar bound $N_{\max}$ of pre-allocated buffer. |
| `KNN_K` | `30` | `10` | `int` | Episodic Memory | Nearest neighbors queried during distance-weighted voting. |
| `OOD_ARBITER_TYPE` | `Mahalanobis++` | `Dual Uncertainty` | `str` | OOD Detection | Out-of-Distribution arbitration mechanism. |
| `MAHALANOBIS_THRESHOLD` | `12.5` | `16.0380` | `float` | OOD Detection | Calibrated Mahalanobis distance threshold $\tau_M$. |
| `ENTROPY_THRESHOLD` | N/A | `0.7382` | `float` | OOD Detection | Calibrated predictive Shannon entropy threshold $\tau_H$ (nats). |
| `NOISE_SWEEP` | `[0.0, 0.2, 0.4, 0.6, 0.8]` | `[0.0, 0.2, 0.4, 0.6, 0.8]` | `list[float]` | Evaluation | Gaussian perturbation levels $\sigma$ applied during drift testing. |
| `RL_STATE_DIM` | `5` | `5` | `int` | RL Agent | Dimension of state vector fed to Double DQN ($s_t \in \mathbb{R}^5$). |
| `RL_N_ACTIONS` | `4` | `4` | `int` | RL Agent | Cardinality of discrete eviction action space $\{0, 1, 2, 3\}$. |
| `CURRICULUM_ALPHA_DECAY`| `0.995` | `0.995` | `float` | Reward System | Multiplicative step decay for Curriculum interpolation $\alpha$. |
| `SLIDING_VALIDATION_BUFFER_SIZE`| `100` | `100` | `int` | Reward System | Capacity of circular validation buffer tracking accuracy $R_{\text{acc}}$. |
| `REPLAY_BUFFER_CAPACITY`| `10000` | `10000` | `int` | RL Agent | Capacity of binary SumTree backing Prioritized Experience Replay. |
| `CNN_OPTIMIZER` | `SGD` (`assign_sub`) | `Adam` (Decoupled decay) | `class` | CNN Backbone | Optimization algorithm for parametric feature extractor. |
| `CNN_LEARNING_RATE` | `0.05` | `0.001` (Cosine decay) | `float` | CNN Backbone | Initial optimizer learning rate $\eta$. |
| `CNN_WEIGHT_DECAY` | `0.0` | `1e-4` | `float` | CNN Backbone | Decoupled $L_2$ weight regularization factor. |
| `CNN_BATCH_SIZE` | `128` | `128` | `int` | CNN Backbone | Mini-batch size for parametric training. |
| `CNN_EPOCHS` | `10` | `25` | `int` | CNN Backbone | Full training epochs through nominal dataset. |
| `SIMULATION_STEPS` | `50000` | `50000` | `int` | RL Simulation | Total streaming prequential simulation steps. |
| `SIMULATION_NOISE_RATE` | `0.1` | `0.1` | `float` | RL Simulation | Bernoulli probability $p_{\text{noise}}$ of perturbation per step. |
| `SIMULATION_NOISE_LEVEL`| `0.6` | `0.6` | `float` | RL Simulation | Gaussian perturbation standard deviation $\sigma$ during simulation. |

---

## 4. Subsystem Breakdown and Detailed Explanations

### 4.1. Invariant 128D Architectural Contract
Both `RawModel` (MNIST) and `RawModelCIFAR10` (CIFAR-10) project visual activations to a 128D latent vector:
```python
# Invariant latent bottleneck across all models:
LATENT_DIM = 128
```
This guarantees that:
- `KNNBanditAgent128D` requires zero modifications when switching between datasets.
- Pre-allocated NumPy memory arrays remain identical in size ($2.54\text{ MB}$).
- The Double DQN policy network input layer is always 5D.

### 4.2. OOD Arbitration Configurations

#### MNIST Regime: Mahalanobis++ on Unit Hypersphere
```python
# src/config.py
MAHALANOBIS_THRESHOLD = 12.5
MAHALANOBIS_PP_PROFILES_PATH = os.path.join(OUTPUT_DIR, "mnist", "mahalanobis_pp_profiles.npz")
```
- Operates on $L_2$-normalized features ($z_{\text{norm}} = z / \|z\|_2$).
- Utilizes Ledoit-Wolf analytic covariance shrinkage.
- Threshold calibrated to 95th percentile ($\tau = 12.5$).

#### CIFAR-10 Regime: Dual Uncertainty Arbiter
```python
# scripts/cifar10/profile_latent.py
THRESHOLD_MAHALANOBIS = 16.0380
THRESHOLD_ENTROPY = 0.7382
PROFILES_PATH = os.path.join(OUTPUT_DIR, "cifar10", "mahalanobis_pp_profiles.npz")
```
- Operates on unnormalized latent representations combined with Softmax posteriors.
- Dual decision rule: Reject sample if $d_M(z) > \tau_M$ or $H(p) > \tau_H$.

### 4.3. Parametric Backbone Training Configurations

```python
# MNIST Configuration:
CNN_BATCH_SIZE = 128
CNN_EPOCHS = 10
CNN_LEARNING_RATE = 0.05
# Optimizer: Custom SGD with assign_sub

# CIFAR-10 Configuration:
CIFAR_BATCH_SIZE = 128
CIFAR_EPOCHS = 25
CIFAR_LEARNING_RATE = 0.001
CIFAR_WEIGHT_DECAY = 1e-4
# Optimizer: Pure TensorFlow Adam with decoupled weight decay
```

---

## 5. Edge AI Memory Footprint Sizing

Because the episodic memory buffer is engineered for resource-constrained Edge AI devices (such as NVIDIA Jetson Orin Nano, Raspberry Pi 5, or microcontrollers), RAM allocation is strictly deterministic:

$$M_{\text{RAM}} = N_{\max} \times \left( D \times \text{sizeof}(\text{float32}) + \text{sizeof}(\text{int32}) + \text{sizeof}(\text{float32}) + \text{sizeof}(\text{int64}) + \text{sizeof}(\text{int32}) \right)$$

For default values ($N_{\max} = 5,000$, $D = 128$):
- States: $5,000 \times 128 \times 4\text{ bytes} = 2,560,000\text{ bytes} \approx 2.44\text{ MB}$
- Actions: $5,000 \times 4\text{ bytes} = 20,000\text{ bytes} \approx 19.53\text{ KB}$
- Rewards: $5,000 \times 4\text{ bytes} = 20,000\text{ bytes} \approx 19.53\text{ KB}$
- Insertion Ticks: $5,000 \times 8\text{ bytes} = 40,000\text{ bytes} \approx 39.06\text{ KB}$
- Usage Counts: $5,000 \times 4\text{ bytes} = 20,000\text{ bytes} \approx 19.53\text{ KB}$
- **Total Static Buffer Footprint:** $\approx \mathbf{2.54\text{ MB}}$

Total system heap allocation during active prequential evaluation (including PyTorch Double DQN MLP and TensorFlow inference engines) remains strictly bounded under **10 MB RAM** ($9.92\text{ MB}$ on MNIST, $9.93\text{ MB}$ on CIFAR-10), satisfying LMOS operational sustainability guidelines.

---

**Navigation:**
- Previous: [Installation and Quick Start](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/docs/getting-started/quickstart.md)
- Up: [Getting Started Index](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/docs/getting-started/README.md)
- Next: [Training Pipeline Guide](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/docs/guides/training-pipeline.md)

---

Licensed under the GNU General Public License v3.0. See [LICENSE](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/LICENSE) for details.
