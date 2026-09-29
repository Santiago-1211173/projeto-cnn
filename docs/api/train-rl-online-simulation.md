# Online RL Simulation Module API Reference

> Part of the [Trustworthy Edge AI: RL-Driven Active Memory Management](../../README.md) documentation.  
> Parent: [API Reference Index](README.md) | Up: [Documentation Index](../README.md)

---

## Module Overview

The `training.train_rl_online_simulation` module orchestrates streaming prequential (test-then-train) simulation under non-stationary environments. It coordinates data generation, Gaussian sensor corruption, feature extraction via the custom CNN, Out-of-Distribution (OOD) routing via Mahalanobis++, episodic memory updates (`KNNBanditAgent128D`), Curriculum Learning reward evaluation (`RewardManager`), and active eviction training via Double DQN (`RLAgent`).

- **Source File:** `training/train_rl_online_simulation.py`
- **Import Statement:**
  ```python
  from training.train_rl_online_simulation import (
      inject_noise,
      MahalanobisPlusPlus,
      OnlineStreamPipeline,
      TrainRLOnlineSimulation,
      run_simulation,
  )
  ```

---

## Public Functions

### `inject_noise(batch: np.ndarray, noise_level: float) -> np.ndarray`

Simulates sensor degradation and concept drift by injecting zero-mean additive Gaussian perturbation noise into an image array:

$$\mathbf{x}_{\text{corrupt}} = \text{clip}\left(\mathbf{x} + \mathcal{N}(0, \sigma^2 \mathbf{I}), 0.0, 1.0\right)$$

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `batch` | `np.ndarray` | *required* | Input image array normalized to $[0.0, 1.0]$. Shape $(N, 28, 28, 1)$ or $(28, 28, 1)$ |
| `noise_level` | `float` | *required* | Gaussian standard deviation $\sigma$ |

**Returns:** `np.ndarray` -- Perturbed image array with `dtype=np.float32`, bounded strictly in $[0.0, 1.0]$.

---

### `run_simulation(n_episodes: int = 1, noise_injection_rate: float = SIMULATION_NOISE_RATE, capacity: int = MEMORY_CAPACITY, n_steps: Optional[int] = None, output_csv_path: Optional[str] = SIMULATION_CSV_PATH) -> Dict[str, List[float]]`

Functional entry point satisfying the programmatic contract defined in the EAAI action plan.

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `n_episodes` | `int` | `1` | Number of simulation episodes |
| `noise_injection_rate` | `float` | `0.1` | Fraction of stream subjected to noise perturbation |
| `capacity` | `int` | `5000` | Maximum episodic memory capacity |
| `n_steps` | `Optional[int]` | `None` | Total steps to execute (falls back to `SIMULATION_STEPS = 50000`) |
| `output_csv_path` | `Optional[str]` | `outputs/train_rl_simulation_log.csv` | Target path for telemetry CSV log |

**Returns:** `Dict[str, List[float]]` mapping metric keys (`"rewards"`, `"losses"`, `"alphas"`, `"epsilons"`, `"sizes"`, `"steps"`) to histories.

---

## Class: `MahalanobisPlusPlus`

```python
class MahalanobisPlusPlus:
```

Out-of-Distribution detector implementing feature-normalized Mahalanobis distance with Ledoit-Wolf analytic covariance shrinkage:

$$\mathbf{z}_{\text{norm}} = \frac{\mathbf{z}}{\|\mathbf{z}\|_2 + 10^{-8}}$$
$$d_M(\mathbf{z}_{\text{norm}}, c) = \sqrt{(\mathbf{z}_{\text{norm}} - \boldsymbol{\mu}_c)^T \boldsymbol{\Sigma}_c^{-1} (\mathbf{z}_{\text{norm}} - \boldsymbol{\mu}_c)}$$

### Constructor

#### `__init__(n_classes: int = KNN_N_ACTIONS, latent_dim: int = LATENT_DIM, threshold: float = MAHALANOBIS_THRESHOLD) -> None`

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `n_classes` | `int` | `10` | Number of semantic target classes |
| `latent_dim` | `int` | `128` | Latent representation dimension |
| `threshold` | `float` | `12.5` | OOD decision boundary threshold |

### Public Methods

#### `fit(latent_features: np.ndarray, labels: np.ndarray) -> None`
Estimates class-conditional centroids $\boldsymbol{\mu}_c$ and well-conditioned precision matrices $\boldsymbol{\Sigma}_c^{-1}$ using `sklearn.covariance.LedoitWolf`.

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `latent_features` | `np.ndarray` | *required* | Unnormalized feature matrix of shape $(N, 128)$ |
| `labels` | `np.ndarray` | *required* | Class label array of shape $(N,)$ |

#### `compute_distance(latent_feature: np.ndarray, target_class: Optional[int] = None) -> float`
Calculates Mahalanobis++ distance for a single vector. If `target_class` is `None`, evaluates against all $C$ classes and returns the minimum: $\min_{c} d_M(\mathbf{z}_{\text{norm}}, c)$.

#### `compute_distances_batch(latent_features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]`
Vectorized evaluation across a mini-batch of vectors.
- **Returns:** `Tuple[np.ndarray, np.ndarray]` containing:
  - `min_distances`: Array of shape $(N,)$ with minimum distance to the closest centroid.
  - `closest_classes`: Array of shape $(N,)$ with the corresponding class index.

#### `is_out_of_distribution(latent_feature: np.ndarray) -> Tuple[bool, float]`
Evaluates if a sample exceeds `self.threshold`.
- **Returns:** `(is_ood: bool, min_distance: float)`.

#### `calibrate_threshold(clean_features: np.ndarray, percentile: float = 95.0) -> float`
Sets `self.threshold` to the specified percentile of in-distribution distances.

#### `save(path: str = MAHALANOBIS_PP_PROFILES_PATH) -> None`
Saves centroids and precision matrices to compressed `.npz` archive.

#### `load(path: str = MAHALANOBIS_PP_PROFILES_PATH) -> None`
Restores fitted profiles from disk.

---

## Class: `OnlineStreamPipeline`

```python
class OnlineStreamPipeline:
```

Front-end data pipeline managing CNN feature extraction and chunked execution to prevent CPU-GPU transfer bottlenecks.

### Constructor

#### `__init__(data_dir: str = DATA_DIR, checkpoint_dir: str = CHECKPOINT_DIR) -> None`
Initializes `RawModel`, restores weights from checkpoint, and loads training data into memory.

### Public Methods

#### `extract_features_batch(images: np.ndarray, batch_size: int = 512) -> Tuple[np.ndarray, np.ndarray, np.ndarray]`
Feeds images through the CNN in chunks.
- **Returns:** `Tuple[np.ndarray, np.ndarray, np.ndarray]`:
  1. `latent_features`: `np.ndarray` of shape $(N, 128)$
  2. `predicted_labels`: `np.ndarray` of shape $(N,)$
  3. `probabilities`: `np.ndarray` of shape $(N, 10)$

---

## Class: `TrainRLOnlineSimulation`

```python
class TrainRLOnlineSimulation:
```

Full prequential streaming simulation engine coordinating active memory management under concept drift.

### Constructor

#### `__init__(capacity: int = MEMORY_CAPACITY, k_neighbors: int = KNN_K_NEIGHBORS, latent_dim: int = LATENT_DIM, buffer_size: int = SLIDING_VALIDATION_BUFFER_SIZE, alpha_decay: float = CURRICULUM_ALPHA_DECAY, replay_capacity: int = REPLAY_BUFFER_CAPACITY, mahalanobis_threshold: float = MAHALANOBIS_THRESHOLD, device: Optional[str] = "cpu") -> None`

Instantiates `KNNBanditAgent128D`, `RewardManager`, `RLAgent`, and `MahalanobisPlusPlus`.

### Public Methods

#### `initialize_detector(pipeline: OnlineStreamPipeline, num_samples: int = 5000) -> None`
Fits the `MahalanobisPlusPlus` detector using a clean subset of training images and calibrates threshold to the 95th percentile.

#### `execute_simulation(n_episodes: int = 1, n_steps: int = SIMULATION_STEPS, noise_injection_rate: float = SIMULATION_NOISE_RATE, noise_level: float = SIMULATION_NOISE_LEVEL, batch_size: int = 32, epsilon_start: float = 1.0, epsilon_end: float = 0.05, log_interval: int = SIMULATION_LOG_INTERVAL, train_frequency: int = 4, output_csv_path: Optional[str] = SIMULATION_CSV_PATH, save_agent_path: Optional[str] = RL_AGENT_CHECKPOINT_PATH) -> Dict[str, List[float]]`

Executes the streaming test-then-train loop.

**Streaming Step Protocol:**
1. Incoming sample $\mathbf{x}_t$ is perturbed with probability `noise_injection_rate`.
2. Passed through CNN: $\mathbf{z}_t, \hat{y}_t, \mathbf{p}_t$.
3. Compute Mahalanobis++ distance $d_M$ and local Shannon entropy $H$.
4. Check anomaly condition: $d_M > \theta$ or $\hat{y}_t \ne y_t$.
5. If memory not full: insert sample directly.
6. If memory full and anomaly detected:
   - Form 5D state vector $\mathbf{s}_t$.
   - Select active eviction action $a_t \in \{0, 1, 2, 3\}$.
   - Execute eviction policy on episodic memory.
   - Compute curriculum reward $R_t$.
   - Store transition in PER buffer and train Double DQN every `train_frequency` decisions.
7. Periodically log metrics to console and CSV.

---

## Command-Line Interface

`training/train_rl_online_simulation.py` can be executed directly from the terminal:

```bash
python training/train_rl_online_simulation.py \
    --steps 50000 \
    --capacity 5000 \
    --noise-rate 0.1 \
    --noise-level 0.6 \
    --log-interval 500 \
    --output-csv outputs/train_rl_simulation_log.csv \
    --save-agent outputs/checkpoints/rl_agent_phase3.pt
```

---

## Usage Example

```python
# example_simulation.py
from training.train_rl_online_simulation import run_simulation

# Run a quick 1,000-step simulation
results = run_simulation(
    n_episodes=1,
    n_steps=1000,
    noise_injection_rate=0.2,
    output_csv_path="outputs/test_simulation.csv",
)

print(f"Executed {len(results['steps'])} steps.")
print(f"Final memory occupancy: {results['sizes'][-1]}")
```

---

## Cross-References

- For the mathematical theory of Mahalanobis++ OOD detection, see [OOD Detection Architecture](../architecture/ood-detection.md).
- For step-by-step reproduction instructions, see [Full Training Pipeline Guide](../guides/training-pipeline.md).

---

**Navigation:**
- Previous: [Reward Manager Module API Reference](reward-manager.md)
- Up: [API Reference Index](README.md)
- Next: [Baseline Comparison Analysis](../results/baseline-comparison.md)

---
Licensed under the GNU General Public License v3.0. See [LICENSE](../../LICENSE) for details.
