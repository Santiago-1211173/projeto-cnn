# Configuration Module API Reference

> Part of the [Trustworthy Edge AI: RL-Driven Active Memory Management](../../README.md) documentation.  
> Parent: [API Reference Index](README.md) | Up: [Documentation Index](../README.md)

---

## Module Overview

The `src.config` module provides a single source of truth for global configuration parameters, filesystem paths, hyperparameter registers, and memory bounds across all subsystems of the architecture. Centralizing these values prevents cyclic dependencies, avoids magic numbers, and guarantees reproducible execution across training, online streaming simulation, and benchmark evaluation.

- **Source File:** `src/config.py`
- **Import Statement:**
  ```python
  from src import config
  # Or specific imports:
  from src.config import MEMORY_CAPACITY, LATENT_DIM, MAHALANOBIS_THRESHOLD
  ```

---

## Complete Parameter Inventory

The table below lists all 35 parameters exposed by `src/config.py`.

| Parameter | Type | Default Value | Functional Domain | Description |
|:----------|:-----|:--------------|:------------------|:------------|
| `PROJECT_ROOT` | `str` | `os.path.dirname(os.path.dirname(...))` | Filesystem | Absolute path to the repository root directory |
| `RANDOM_SEED` | `int` | `42` | Reproducibility | Global seed applied to NumPy, TensorFlow, and PyTorch |
| `MAHALANOBIS_THRESHOLD` | `float` | `12.5` | Routing | Distance threshold separating In-Distribution (ID) from Out-of-Distribution (OOD) |
| `NOISE_SWEEP` | `List[float]` | `[0.0, 0.2, 0.4, 0.6, 0.8]` | Evaluation | Perturbation standard deviations for noise evaluation sweep |
| `THRESHOLD_SWEEP_START` | `float` | `5.0` | Evaluation | Lower boundary for Mahalanobis routing sensitivity analysis |
| `THRESHOLD_SWEEP_END` | `float` | `30.0` | Evaluation | Upper boundary for Mahalanobis routing sensitivity analysis |
| `THRESHOLD_SWEEP_STEP` | `float` | `2.5` | Evaluation | Step size for sweeping Mahalanobis distance thresholds |
| `DATA_DIR` | `str` | `projeto-cnn/data/MNIST/raw` | Filesystem | Directory containing raw binary MNIST byte files |
| `OUTPUT_DIR` | `str` | `projeto-cnn/outputs` | Filesystem | Root output directory for metrics, plots, and logs |
| `CHECKPOINT_DIR` | `str` | `projeto-cnn/outputs/checkpoints` | Filesystem | Directory storing trained model checkpoints |
| `LOG_DIR` | `str` | `projeto-cnn/outputs/logs` | Filesystem | Directory for execution telemetry and run logs |
| `KNN_K` | `int` | `30` | Episodic Memory | Number of nearest neighbors for legacy inference |
| `KNN_N_ACTIONS` | `int` | `10` | Episodic Memory | Cardinality of the classification action space (10 MNIST digits) |
| `MEMORY_CAPACITY` | `int` | `5000` | Episodic Memory | Maximum capacity of the episodic memory buffer (vectors) |
| `LATENT_DIM` | `int` | `128` | Feature Extractor | Dimensionality of the CNN bottleneck representation space |
| `KNN_K_NEIGHBORS` | `int` | `30` | Episodic Memory | Number of nearest neighbors queried during k-NN retrieval |
| `RL_STATE_DIM` | `int` | `5` | Reinforcement Learning | Dimensionality of the normalized state vector fed to QNetwork |
| `RL_N_ACTIONS` | `int` | `4` | Reinforcement Learning | Discrete memory management actions: 0=Ignore, 1=FIFO, 2=LFU, 3=Redundant |
| `CURRICULUM_ALPHA_DECAY` | `float` | `0.995` | Reward System | Multiplicative step decay for curriculum interpolation parameter $\alpha$ |
| `SLIDING_VALIDATION_BUFFER_SIZE` | `int` | `100` | Reward System | Maximum number of hard OOD samples held in circular validation buffer |
| `REPLAY_BUFFER_CAPACITY` | `int` | `10000` | Reinforcement Learning | Maximum transition capacity of the Prioritized Experience Replay buffer |
| `CNN_BATCH_SIZE` | `int` | `256` | CNN Training | Mini-batch size for SGD optimization of the feature extractor |
| `CNN_EPOCHS` | `int` | `10` | CNN Training | Total training epochs for the custom CNN |
| `CNN_LEARNING_RATE` | `float` | `0.01` | CNN Training | Learning rate for custom SGD optimizer |
| `SIMULATION_STEPS` | `int` | `50000` | Online Simulation | Total prequential streaming steps per simulation run |
| `SIMULATION_NOISE_RATE` | `float` | `0.1` | Online Simulation | Fraction ($10\%$) of incoming stream subjected to noise perturbation |
| `SIMULATION_NOISE_LEVEL` | `float` | `0.6` | Online Simulation | Gaussian perturbation intensity $\sigma$ applied to corrupted stream samples |
| `SIMULATION_LOG_INTERVAL` | `int` | `500` | Online Simulation | Step frequency for metric logging and CSV serialization |
| `SIMULATION_CSV_PATH` | `str` | `outputs/train_rl_simulation_log.csv` | Filesystem | File path for streaming prequential metrics log |
| `RL_AGENT_CHECKPOINT_PATH` | `str` | `outputs/checkpoints/rl_agent_phase3.pt` | Filesystem | Path for saving and restoring trained Double DQN weights |
| `MAHALANOBIS_PP_PROFILES_PATH` | `str` | `outputs/mahalanobis_pp_profiles.npz` | Filesystem | Path for serialized Mahalanobis++ centroids and precision matrices |
| `MEMORY_BANK_10D_PATH` | `str` | `outputs/knn_memory_bank.npz` | Filesystem | Path for legacy 10D episodic memory archive |
| `MEMORY_BANK_128D_PATH` | `str` | `outputs/knn_memory_bank_128d.npz` | Filesystem | Path for 128D pre-populated episodic memory bank |
| `MAHALANOBIS_PROFILES_PATH` | `str` | `outputs/mahalanobis_profiles.npz` | Filesystem | Path for standard Mahalanobis covariance profiles |
| `MEMORY_MAPPING_PATH` | `str` | `outputs/knn_memory_mapping.json` | Filesystem | Path for class mapping metadata JSON |

---

## Detailed Specifications by Functional Subsystem

### 1. General & Directory Path Resolution
Paths are resolved relative to `PROJECT_ROOT`, ensuring platform independence across Windows and Linux environments without hardcoded drive letters.

```python
# src/config.py
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "MNIST", "raw")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
```

### 2. Episodic Memory Sizing & Edge AI Bounds
The memory footprint is deterministically bounded to adhere to the strict physical constraints of embedded Edge AI hardware:

$$\text{Memory Buffer RAM} = C \times \left( D \times 4\text{ bytes} + 4\text{ bytes} + 4\text{ bytes} + 8\text{ bytes} + 4\text{ bytes} \right)$$

With $C = \text{MEMORY\_CAPACITY} = 5,000$ and $D = \text{LATENT\_DIM} = 128$:
- $\text{State Matrix } (5000 \times 128 \times \text{float32}) = 2.56\text{ MB}$
- $\text{Metadata Arrays } (5000 \times [4 + 4 + 8 + 4]\text{ bytes}) = 0.10\text{ MB}$
- **Total Static Buffer Footprint:** $\approx 2.66\text{ MB}$, fitting comfortably within low-power microcontroller or embedded SoC SRAM/DRAM limits.

### 3. Out-of-Distribution Calibration
- `MAHALANOBIS_THRESHOLD = 12.5`: Default decision threshold separating nominal inputs from anomalous samples. Calibrated at the 95th percentile of in-distribution validation distances.
- `NOISE_SWEEP = [0.0, 0.2, 0.4, 0.6, 0.8]`: Standard evaluation grid to assess degradation across varying degrees of input degradation.

### 4. Reinforcement Learning & PER Bounds
- `RL_STATE_DIM = 5`: Matches the 5 normalized telemetry signals: Mahalanobis distance, neighborhood Shannon entropy, minimum k-NN distance, prediction error, and memory RAM occupancy.
- `RL_N_ACTIONS = 4`: Defines the 4 active memory management actions: `0: Ignore`, `1: FIFO`, `2: LFU`, `3: Redundant`.
- `REPLAY_BUFFER_CAPACITY = 10000`: Caps transition storage in the Prioritized Experience Replay buffer, requiring $\approx 10.4\text{ MB}$ of host memory.

---

## Usage Example

```python
# example_usage.py
from src.config import (
    MEMORY_CAPACITY,
    LATENT_DIM,
    MAHALANOBIS_THRESHOLD,
    RL_STATE_DIM,
    RL_N_ACTIONS,
)

print(f"Memory Capacity: {MEMORY_CAPACITY} vectors of dim {LATENT_DIM}")
print(f"Mahalanobis Routing Threshold: {MAHALANOBIS_THRESHOLD}")
print(f"RL State Vector Dimension: {RL_STATE_DIM} | Action Space: {RL_N_ACTIONS}")
```

---

## Cross-References

- For operational guidelines on tuning configuration values, see [Getting Started: Configuration Reference](../getting-started/configuration.md).
- For the architectural rationale behind memory bounds, see [Episodic Memory Architecture](../architecture/episodic-memory.md).
- For routing threshold derivations, see [Out-of-Distribution Detection](../architecture/ood-detection.md).

---

**Navigation:**
- Previous: [API Reference Index](README.md)
- Up: [API Reference Index](README.md)
- Next: [Custom CNN Module API Reference](custom-cnn.md)

---
Licensed under the GNU General Public License v3.0. See [LICENSE](../../LICENSE) for details.
