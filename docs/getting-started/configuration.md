# Configuration Reference

> Part of the [Trustworthy Edge AI: RL-Driven Active Memory Management](../../README.md) documentation.
> Parent: [Getting Started Index](README.md) | Up: [Documentation Index](../README.md)

---

## 1. Overview and Design Philosophy

The entire codebase relies on a **centralized configuration module** located at [`src/config.py`](../../src/config.py). This file serves as the single source of truth for architectural dimensions, storage paths, hyperparameters, and operational flags across all subsystems:
- Parametric CNN training (`scripts/train_cnn.py`)
- Out-of-Distribution (OOD) profiling and calibration (`scripts/profile_latent.py`)
- Non-parametric episodic memory allocation (`src/models/knn_bandit_agent.py`)
- Deep reinforcement learning agent and experience replay (`src/models/rl_agent.py`)
- Curriculum Learning reward system (`src/models/reward_manager.py`)
- Online non-stationary streaming simulation (`training/train_rl_online_simulation.py`)
- Multi-baseline experimental evaluation (`evaluate_hybrid_global.py`)

### Architectural Principles of `config.py`
1. **Zero Hardcoded Magic Numbers:** Operational scripts import constants directly from `src.config` rather than defining local defaults.
2. **Deterministic Reproducibility:** Global seeds (`RANDOM_SEED`) are propagated across NumPy, TensorFlow, and PyTorch runtime engines.
3. **Dynamic Path Resolution:** All filesystem paths are derived dynamically relative to `PROJECT_ROOT`, guaranteeing cross-platform portability on Linux, macOS, and Windows.
4. **Decoupled Architecture:** Subsystem dimensions (e.g., `LATENT_DIM`, `RL_STATE_DIM`, `RL_N_ACTIONS`) are declared centrally to prevent circular import dependencies between models, agents, and simulation drivers.

---

## 2. Complete Configuration Parameters Master Table

The table below catalogs all 35 configuration parameters defined in `src/config.py`:

| Parameter | Default Value | Python Type | Subsystem / Phase | Functional Description |
|:----------|:--------------|:------------|:------------------|:-----------------------|
| `PROJECT_ROOT` | Dynamic `abspath` | `str` | System Paths | Absolute path to the repository root directory on the host filesystem. |
| `RANDOM_SEED` | `42` | `int` | General | Master pseudo-random number generator seed enforcing reproducibility. |
| `MAHALANOBIS_THRESHOLD` | `12.5` | `float` | OOD Detection | Calibrated threshold $\tau$ separating nominal in-distribution inputs from anomalies. |
| `NOISE_SWEEP` | `[0.0, 0.2, 0.4, 0.6, 0.8]` | `list[float]` | Evaluation | Standard Gaussian perturbation scales $\sigma$ used for noise stress sweeps. |
| `THRESHOLD_SWEEP_START` | `5.0` | `float` | Evaluation | Lower bound of the Mahalanobis threshold exploration interval. |
| `THRESHOLD_SWEEP_END` | `30.0` | `float` | Evaluation | Upper bound of the Mahalanobis threshold exploration interval. |
| `THRESHOLD_SWEEP_STEP` | `2.5` | `float` | Evaluation | Step increment for generating candidate thresholds $\tau \in [5.0, 30.0]$. |
| `DATA_DIR` | `.../data/MNIST/raw` | `str` | Data Pipeline | Directory housing the original binary MNIST ubyte dataset files. |
| `OUTPUT_DIR` | `.../outputs` | `str` | File Storage | Base root directory for all generated metrics, checkpoints, and figures. |
| `CHECKPOINT_DIR` | `.../outputs/checkpoints` | `str` | File Storage | Subdirectory storing TensorFlow CNN checkpoints and PyTorch `.pt` models. |
| `LOG_DIR` | `.../outputs/logs` | `str` | File Storage | Subdirectory storing training execution logs and console transcripts. |
| `KNN_K` | `30` | `int` | Phase 1: Memory | Number of nearest neighbors queried during distance-weighted voting. |
| `KNN_N_ACTIONS` | `10` | `int` | Phase 1: Memory | Number of discrete output classification categories (0 to 9 digits). |
| `MEMORY_CAPACITY` | `5000` | `int` | Phase 1: Memory | Maximum exemplar capacity bound $N_{\max}$ of pre-allocated contiguous arrays. |
| `LATENT_DIM` | `128` | `int` | CNN / Memory | Dimensionality of the CNN bottleneck representation vector $z \in \mathbb{R}^{128}$. |
| `KNN_K_NEIGHBORS` | `30` | `int` | Phase 1: Memory | Hyperparameter defining nearest neighbors count for online simulation retrieval. |
| `RL_STATE_DIM` | `5` | `int` | Phase 2: RL Agent | Dimension of the normalized state vector fed to the Double DQN policy net. |
| `RL_N_ACTIONS` | `4` | `int` | Phase 2: RL Agent | Cardinality of the discrete eviction action space $\{0, 1, 2, 3\}$. |
| `CURRICULUM_ALPHA_DECAY`| `0.995` | `float` | Phase 2: Reward | Multiplicative decay per step for Curriculum reward interpolation $\alpha$. |
| `SLIDING_VALIDATION_BUFFER_SIZE` | `100` | `int` | Phase 2: Reward | Capacity of circular validation buffer tracking empirical accuracy $R_{\text{acc}}$. |
| `REPLAY_BUFFER_CAPACITY` | `10000` | `int` | Phase 2: RL Agent | Capacity of the binary SumTree backing Prioritized Experience Replay. |
| `CNN_BATCH_SIZE` | `256` | `int` | CNN Training | Mini-batch sample size utilized during parametric feature extractor training. |
| `CNN_EPOCHS` | `10` | `int` | CNN Training | Total training epochs for the custom CNN on nominal training data. |
| `CNN_LEARNING_RATE` | `0.01` | `float` | CNN Training | Initial SGD optimization learning rate for CNN convolutional and dense layers. |
| `SIMULATION_STEPS` | `50000` | `int` | Phase 3: Simulation | Total streaming prequential evaluation steps in the online simulation. |
| `SIMULATION_NOISE_RATE` | `0.1` | `float` | Phase 3: Simulation | Bernoulli probability $p_{\text{noise}}$ of injecting noise perturbation at step $t$. |
| `SIMULATION_NOISE_LEVEL`| `0.6` | `float` | Phase 3: Simulation | Standard deviation $\sigma$ of additive Gaussian noise perturbation $\mathcal{N}(0, \sigma^2)$. |
| `SIMULATION_LOG_INTERVAL`| `500` | `int` | Phase 3: Simulation | Frequency of structured progress telemetry emission to console and CSV. |
| `SIMULATION_CSV_PATH` | `.../outputs/train_rl_simulation_log.csv` | `str` | Phase 3: Simulation | Destination CSV path for online streaming simulation telemetry logs. |
| `RL_AGENT_CHECKPOINT_PATH` | `.../outputs/checkpoints/rl_agent_phase3.pt` | `str` | Phase 3: Simulation | Serialized PyTorch checkpoint path for the trained Double DQN agent. |
| `MAHALANOBIS_PP_PROFILES_PATH` | `.../outputs/mahalanobis_pp_profiles.npz` | `str` | OOD Detection | Serialized centroids and regularized precision matrices for Mahalanobis++. |
| `MEMORY_BANK_10D_PATH` | `.../outputs/knn_memory_bank.npz` | `str` | Phase 1: Memory | Legacy 10D Softmax probability memory bank archive path. |
| `MEMORY_BANK_128D_PATH` | `.../outputs/knn_memory_bank_128d.npz` | `str` | Phase 1: Memory | Primary 128D latent feature episodic memory bank archive path. |
| `MAHALANOBIS_PROFILES_PATH` | `.../outputs/mahalanobis_profiles.npz` | `str` | OOD Detection | Legacy un-normalized Mahalanobis statistical profiles archive path. |
| `MEMORY_MAPPING_PATH` | `.../outputs/knn_memory_mapping.json` | `str` | Phase 1: Memory | JSON mapping linking latent memory indices to source metadata. |

---

## 3. Subsystem Breakdown and Detailed Explanations

### 3.1. General Environment and Path Constants

```python
# src/config.py
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RANDOM_SEED = 42

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "MNIST", "raw")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
```

- **`PROJECT_ROOT`:** Automatically identified using two levels of `os.path.dirname` above `src/config.py`. Allows scripts to be invoked from any working directory without path failure.
- **`RANDOM_SEED`:** Fixed to `42`. Seeded during startup via:
  ```python
  np.random.seed(RANDOM_SEED)
  tf.random.set_seed(RANDOM_SEED)
  torch.manual_seed(RANDOM_SEED)
  ```
- **`DATA_DIR`:** Expected location for raw MNIST binary files (`train-images-idx3-ubyte`, etc.).
- **`OUTPUT_DIR`, `CHECKPOINT_DIR`, `LOG_DIR`:** Directory targets created automatically on first run to isolate runtime artifacts from source code.

---

### 3.2. CNN Parametric Feature Extractor

```python
# src/config.py
CNN_BATCH_SIZE = 256
CNN_EPOCHS = 10
CNN_LEARNING_RATE = 0.01
LATENT_DIM = 128
```

- **`CNN_BATCH_SIZE = 256`:** Balances gradient estimation stability with cache utilization on Edge GPUs and multi-core CPUs.
- **`CNN_EPOCHS = 10`:** Delivers convergence on nominal MNIST ($> 98.5\%$ validation accuracy) in custom SGD without overfitting to nominal artifacts.
- **`CNN_LEARNING_RATE = 0.01`:** Step size for the from-scratch SGD optimizer (`src/scratch/optimizers.py`), applied to weights initialized with He Normal and Glorot Uniform distributions.
- **`LATENT_DIM = 128`:** The critical architectural bottleneck. Dense layer `dense_1` projects the 1,600D flattened convolutional activations down to $\mathbb{R}^{128}$ before final 10D softmax classification. This 128D space acts as the shared geometric manifold for OOD detection and episodic memory retrieval.

---

### 3.3. Out-of-Distribution Detection (Mahalanobis++)

```python
# src/config.py
MAHALANOBIS_THRESHOLD = 12.5
MAHALANOBIS_PP_PROFILES_PATH = os.path.join(OUTPUT_DIR, "mahalanobis_pp_profiles.npz")
NOISE_SWEEP = [0.0, 0.2, 0.4, 0.6, 0.8]
THRESHOLD_SWEEP_START = 5.0
THRESHOLD_SWEEP_END = 30.0
THRESHOLD_SWEEP_STEP = 2.5
```

- **`MAHALANOBIS_THRESHOLD = 12.5`:** The operational threshold $\tau$. When an incoming image's $L_2$-normalized latent feature $z_{\text{norm}}$ produces minimum Mahalanobis distance $d_M(z_{\text{norm}}, c) \le \tau$, the input is classified as **In-Distribution (ID)** and handled by the parametric CNN. If $d_M > \tau$, it is flagged as **Out-of-Distribution (OOD)** or degraded, triggering diversion to the episodic memory subsystem.
- **`MAHALANOBIS_PP_PROFILES_PATH`:** Stores the serialized dictionary containing:
  - `centroids`: Class mean vectors $\mu_c \in \mathbb{R}^{10 \times 128}$ on the unit hypersphere.
  - `precisions`: Inverted, Ledoit-Wolf shrunk covariance matrices $\Sigma_c^{-1} \in \mathbb{R}^{10 \times 128 \times 128}$.
  - `threshold`: Calibrated empirical 95th percentile distance ($12.5$).
- **`NOISE_SWEEP`:** Evaluation noise levels spanning clean conditions ($\sigma=0.0$) through severe degradation ($\sigma=0.8$).
- **`THRESHOLD_SWEEP_*`:** Range $[5.0, 30.0]$ with step $2.5$ used during sensitivity analysis in `evaluate_global.py`.

---

### 3.4. Non-Parametric Episodic Memory (Phase 1)

```python
# src/config.py
MEMORY_CAPACITY = 5000
KNN_K = 30
KNN_K_NEIGHBORS = 30
KNN_N_ACTIONS = 10
MEMORY_BANK_128D_PATH = os.path.join(OUTPUT_DIR, "knn_memory_bank_128d.npz")
```

- **`MEMORY_CAPACITY = 5000`:** Hard upper bound $N_{\max}$ on the number of exemplar vectors retained in the episodic memory. When this bound is reached, insertion of new observations requires evicting an existing slot.
- **`KNN_K = 30` / `KNN_K_NEIGHBORS = 30`:** Number of nearest neighbors queried via vectorized Euclidean distance using `np.argpartition`. Predictions are formed via inverse distance weighting:
  $$w_i = \frac{1}{\|z - z_i\|_2 + \epsilon}, \quad \hat{y} = \arg\max_{a} \sum_{i: y_i = a} w_i \cdot r_i$$
- **`KNN_N_ACTIONS = 10`:** Output action space for digit classification ($0 \dots 9$).
- **`MEMORY_BANK_128D_PATH`:** File path where the pre-allocated memory bank is saved and loaded via `np.savez_compressed` / `np.load`.

---

### 3.5. Active RL Agent and Curriculum Rewards (Phase 2)

```python
# src/config.py
RL_STATE_DIM = 5
RL_N_ACTIONS = 4
CURRICULUM_ALPHA_DECAY = 0.995
SLIDING_VALIDATION_BUFFER_SIZE = 100
REPLAY_BUFFER_CAPACITY = 10000
RL_AGENT_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "rl_agent_phase3.pt")
```

- **`RL_STATE_DIM = 5`:** The 5-dimensional environment state vector observed by the RL agent before an eviction decision:
  1. $s_0$: Normalized Mahalanobis distance $\min(d_M / 50.0, 1.0)$
  2. $s_1$: Normalized local label entropy $H / \log(k)$
  3. $s_2$: Normalized distance to nearest neighbor $\min(d_{\min} / 10.0, 1.0)$
  4. $s_3$: Binary CNN prediction error indicator $\mathbb{I}(\hat{y}_{\text{CNN}} \neq y)$
  5. $s_4$: Buffer occupancy ratio $N_{\text{current}} / N_{\max}$
- **`RL_N_ACTIONS = 4`:** Discrete actions available to the Double DQN:
  - `Action 0`: **Ignore** (retain memory unchanged, reject candidate)
  - `Action 1`: **FIFO** (evict exemplar with minimum `_insertion_ticks`)
  - `Action 2`: **LFU** (evict exemplar with minimum `_usage_counts`, tie-break by oldest)
  - `Action 3`: **Redundancy** (evict geometric nearest neighbor of the same class)
- **`CURRICULUM_ALPHA_DECAY = 0.995`:** Exponential decay governing Curriculum Learning reward transition:
  $$R_t = \alpha_t \cdot R_{\text{geom}} + (1 - \alpha_t) \cdot R_{\text{acc}}, \quad \alpha_{t+1} = \max(\alpha_t \times 0.995, 0.01)$$
- **`SLIDING_VALIDATION_BUFFER_SIZE = 100`:** Circular buffer holding the most recent 100 verified samples used to compute real-time empirical accuracy $R_{\text{acc}}$.
- **`REPLAY_BUFFER_CAPACITY = 10000`:** Size of the binary SumTree data structure backing Prioritized Experience Replay (PER), sampling transitions proportional to TD-error priority $p_i = |\delta_i| + 10^{-5}$.

---

### 3.6. Online Streaming Simulation (Phase 3)

```python
# src/config.py
SIMULATION_STEPS = 50000
SIMULATION_NOISE_RATE = 0.1
SIMULATION_NOISE_LEVEL = 0.6
SIMULATION_LOG_INTERVAL = 500
SIMULATION_CSV_PATH = os.path.join(OUTPUT_DIR, "train_rl_simulation_log.csv")
```

- **`SIMULATION_STEPS = 50000`:** Total number of sequential streaming events processed in the test-then-train prequential simulation protocol.
- **`SIMULATION_NOISE_RATE = 0.1`:** $10\%$ probability per step that the incoming image undergoes non-stationary Gaussian noise perturbation, simulating intermittent sensor degradation.
- **`SIMULATION_NOISE_LEVEL = 0.6`:** Standard deviation $\sigma = 0.6$ of the Gaussian perturbation $\tilde{x} = \text{clip}(x + \mathcal{N}(0, \sigma^2), 0, 1)$.
- **`SIMULATION_LOG_INTERVAL = 500`:** Emits structured progress rows to `SIMULATION_CSV_PATH` every 500 steps, recording step index, rolling reward, TD loss, epsilon exploration, and memory class distribution.

---

## 4. Edge AI Memory Footprint Sizing

Because the episodic memory is engineered for Edge AI hardware (such as Raspberry Pi 4/5, NVIDIA Jetson Orin Nano, or embedded NPU accelerators), its RAM footprint is strictly bounded and pre-allocated.

### RAM Sizing Formula

The total memory consumed by the `KNNBanditAgent128D` pre-allocated NumPy storage is given by:

$$M_{\text{RAM}} = N_{\max} \times \left( D \times \text{sizeof}(\text{float32}) + \text{sizeof}(\text{int32}) + \text{sizeof}(\text{float32}) + \text{sizeof}(\text{int64}) + \text{sizeof}(\text{int32}) \right) \text{ bytes}$$

Substituting default parameters ($N_{\max} = 5,000$, $D = 128$):
- `_states`: $5,000 \times 128 \times 4\text{ bytes} = 2,560,000\text{ bytes} \approx 2.44\text{ MB}$
- `_actions`: $5,000 \times 4\text{ bytes} = 20,000\text{ bytes} \approx 19.53\text{ KB}$
- `_rewards`: $5,000 \times 4\text{ bytes} = 20,000\text{ bytes} \approx 19.53\text{ KB}$
- `_insertion_ticks`: $5,000 \times 8\text{ bytes} = 40,000\text{ bytes} \approx 39.06\text{ KB}$
- `_usage_counts`: $5,000 \times 4\text{ bytes} = 20,000\text{ bytes} \approx 19.53\text{ KB}$

$$\mathbf{M_{\text{total}}} = 2,660,000\text{ bytes} \approx \mathbf{2.54\text{ MB}}$$

The complete episodic memory buffer requires **only 2.54 MB of RAM**, making it exceptionally well-suited for embedded microcontrollers and edge gateways with strict memory budgets.

| Hardware Tier | Available System RAM | Recommended `MEMORY_CAPACITY` | Memory Buffer Footprint |
|:--------------|:---------------------|:------------------------------|:------------------------|
| Ultra-low Edge (Cortex-A53) | 512 MB - 1 GB | `1000` | $\approx 0.51$ MB |
| **Standard Edge (Jetson Nano / RPi 4)** | **2 GB - 4 GB** | **`5000` (Default)** | **$\approx 2.54$ MB** |
| High-performance Edge (Jetson Orin) | 8 GB - 16 GB | `10000` | $\approx 5.07$ MB |
| Industrial Edge Server | 32 GB+ | `25000` | $\approx 12.68$ MB |

---

## 5. Modifying and Overriding Configuration

### Method 1: Editing `src/config.py` Directly (Persistent)
To permanently adjust a parameter for an experiment, modify the variable assignment directly in `src/config.py`:
```python
# src/config.py
MEMORY_CAPACITY = 10000         # Expand memory capacity to 10k slots
MAHALANOBIS_THRESHOLD = 15.0   # Relax OOD sensitivity
```

### Method 2: Command-Line Argument Overrides (Runtime)
Operational scripts support command-line overrides that take precedence over the defaults in `src/config.py`:

```bash
# Override simulation steps, noise rate, and capacity at runtime
python -m training.train_rl_online_simulation \
    --steps 25000 \
    --capacity 2500 \
    --noise-rate 0.15 \
    --noise-level 0.7
```

### Method 3: Programmatic Import and Dynamic Modification
When writing custom evaluation scripts or Jupyter / Python notebooks:

```python
# custom_experiment.py
from src import config

# Inspect defaults
print("Default capacity:", config.MEMORY_CAPACITY)

# Override dynamically in memory before instantiating modules
config.MEMORY_CAPACITY = 2000
config.MAHALANOBIS_THRESHOLD = 10.0

from src.models.knn_bandit_agent import KNNBanditAgent128D
agent = KNNBanditAgent128D(capacity=config.MEMORY_CAPACITY)
```

---

**Navigation:**
- Previous: [Installation and Quick Start](quickstart.md)
- Up: [Getting Started Index](README.md)
- Next: [Documentation Index](../README.md)

---
Licensed under the GNU General Public License v3.0. See [LICENSE](../../LICENSE) for details.
