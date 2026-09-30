# Reward Manager Module API Reference

> Part of the [Active Episodic Memory Management via Reinforcement Learning for Robust CNN Inference on Out-of-Distribution Data](../../README.md) documentation.  
> Parent: [API Reference Index](README.md) | Up: [Documentation Index](../README.md)

---

## Module Overview

The `src.models.reward_manager` module orchestrates curriculum learning signals for active episodic memory management. Because evaluating empirical task accuracy over an entire dataset is prohibitively expensive for streaming Edge AI, the reward manager dynamically blends a rapid, instantaneous Geometric Proxy ($R_{\text{geom}}$, evaluating local Euclidean dispersion and redundancy avoidance) with empirical validation accuracy ($R_{\text{acc}}$) measured over a circular, pre-allocated Sliding Validation Buffer of hard Out-of-Distribution (OOD) instances.

- **Source File:** `src/models/reward_manager.py`
- **Import Statement:**
  ```python
  from src.models.reward_manager import RewardManager
  ```

---

## Class: `RewardManager`

```python
class RewardManager:
```

### Attributes

| Attribute | Type | Description |
|:----------|:-----|:------------|
| `buffer_size` | `int` | Maximum capacity of the sliding validation buffer (default: 100) |
| `alpha_decay` | `float` | Multiplicative decay applied to $\alpha$ per reward evaluation (default: 0.995) |
| `min_alpha` | `float` | Lower floor for $\alpha$ to preserve residual geometric regularization (default: 0.01) |
| `latent_dim` | `int` | Dimensionality of stored representations (default: 128) |
| `alpha` | `float` | Current curriculum balance parameter in $[\text{min\_alpha}, 1.0]$ |
| `_val_states` | `np.ndarray` | Pre-allocated circular buffer of shape `(buffer_size, latent_dim)` with `dtype=np.float32` |
| `_val_labels` | `np.ndarray` | Pre-allocated labels array of shape `(buffer_size,)` with `dtype=np.int32` |
| `_val_size` | `int` | Current count of active validation instances ($0 \le \text{size} \le \text{buffer\_size}$) |
| `_val_ptr` | `int` | Circular insertion pointer index in $[0, \text{buffer\_size} - 1]$ |

---

## Constructor

#### `__init__(buffer_size: int = SLIDING_VALIDATION_BUFFER_SIZE, alpha_decay: float = CURRICULUM_ALPHA_DECAY, min_alpha: float = 0.01, latent_dim: int = LATENT_DIM) -> None`

Pre-allocates memory blocks for streaming validation samples and sets initial curriculum weight $\alpha_0 = 1.0$.

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `buffer_size` | `int` | `100` | Capacity of sliding validation buffer |
| `alpha_decay` | `float` | `0.995` | Step decay multiplier for curriculum parameter $\alpha$ |
| `min_alpha` | `float` | `0.01` | Minimum threshold below which $\alpha$ cannot decay |
| `latent_dim` | `int` | `128` | Latent vector dimensionality |

---

## Public Methods

#### `update_validation_buffer(state: np.ndarray, true_label: int, predicted_label: int) -> None`

Inserts a hard, anomalous, or misclassified sample into the circular sliding validation buffer. This ensures that the validation set continuously adapts to non-stationary concept drift without unbounded memory growth.

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `state` | `np.ndarray` | *required* | Latent feature vector of shape `(latent_dim,)` |
| `true_label` | `int` | *required* | Ground truth integer class index |
| `predicted_label` | `int` | *required* | Integer class label predicted by the system prior to memory update |

#### `compute_reward(evicted_index: int, memory: KNNBanditAgent128D, new_state: np.ndarray, characteristic_dist: float = 1.0, max_eval_samples: int = 15) -> float`

Computes the composite, curriculum-weighted reward for an active memory management decision:

$$R_{\text{total}} = \alpha \cdot R_{\text{geom}} + (1 - \alpha) \cdot R_{\text{acc}}$$
$$\alpha_{t+1} = \max\left(\text{min\_alpha}, \alpha_t \cdot \text{alpha\_decay}\right)$$

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `evicted_index` | `int` | *required* | Slot index overwritten in memory ($\ge 0$), or `-1` if action was `Ignore` |
| `memory` | `KNNBanditAgent128D` | *required* | Live instance of the episodic memory buffer |
| `new_state` | `np.ndarray` | *required* | Candidate latent representation of shape `(latent_dim,)` |
| `characteristic_dist` | `float` | `1.0` | Scaling parameter $\tau$ for geometric distance transformation |
| `max_eval_samples` | `int` | `15` | Subsample limit evaluated against memory for bounded execution latency |

**Returns:** `float` -- Bounded composite reward strictly constrained to $[-1.0, 1.0]$.

#### `get_current_alpha() -> float`
Returns the current curriculum interpolation weight $\alpha \in [\text{min\_alpha}, 1.0]$.

#### `get_buffer_stats() -> Dict[str, Any]`
Returns operational diagnostics of the validation buffer:
- `"validation_size"`: `int`
- `"buffer_capacity"`: `int`
- `"occupancy_pct"`: `float`
- `"current_alpha"`: `float`

---

## Internal Mathematical Mechanics

### 1. Geometric Proxy Evaluation (`_compute_geometric_proxy`)

Evaluates Euclidean dispersion against existing memory representations without label dependency:

- **Case A: Insertion Occurred (`evicted_index >= 0`):**
  Identifies distance $d_{\min}$ to the nearest neighbor among stored vectors (excluding `evicted_index`). Rewards space expansion and penalizes storing redundant duplicates:
  $$R_{\text{geom}} = 2 \left( \frac{d_{\min}}{d_{\min} + \tau} \right) - 1 \in [-1.0, 1.0]$$
  - If $d_{\min} = 0$ (exact duplicate stored): $R_{\text{geom}} = -1.0$.
  - If $d_{\min} = \tau$: $R_{\text{geom}} = 0.0$.
  - If $d_{\min} \gg \tau$ (novel cluster coverage): $R_{\text{geom}} \to +1.0$.

- **Case B: Rejection Occurred (`evicted_index < 0`, Action `Ignore`):**
  Rewards the rejection of redundant vectors and penalizes rejecting novel samples:
  $$R_{\text{geom}} = 1 - 2 \left( \frac{d_{\min}}{d_{\min} + \tau} \right) \in [-1.0, 1.0]$$

### 2. Sliding Validation Accuracy (`_compute_validation_accuracy_reward`)

Evaluates $k$-NN accuracy over a sub-sampled slice of size $\min(\text{val\_size}, \text{max\_eval\_samples})$:

$$R_{\text{acc}} = 2 \cdot \text{Accuracy}_{\text{val}} - 1 \in [-1.0, 1.0]$$
- $100\%$ validation accuracy $\to R_{\text{acc}} = +1.0$
- $50\%$ validation accuracy $\to R_{\text{acc}} = 0.0$
- $0\%$ validation accuracy $\to R_{\text{acc}} = -1.0$

---

## Usage Example

```python
# example_reward_manager.py
import numpy as np
from src.models.knn_bandit_agent import KNNBanditAgent128D
from src.models.reward_manager import RewardManager

# Instantiate components
memory = KNNBanditAgent128D(capacity=500, k=5, latent_dim=128)
reward_mgr = RewardManager(buffer_size=50, alpha_decay=0.995, min_alpha=0.01)

# Populate validation buffer with hard OOD failure cases
for i in range(20):
    hard_sample = np.random.randn(128).astype(np.float32)
    reward_mgr.update_validation_buffer(hard_sample, true_label=i % 10, predicted_label=(i + 1) % 10)

# Memory has some samples
for i in range(50):
    memory.add_experience(np.random.randn(128).astype(np.float32), action=i % 10, reward=1.0)

# Compute reward for an eviction decision
candidate_vec = np.random.randn(128).astype(np.float32)
evicted_slot = memory.evict_oldest(candidate_vec, new_action=3, new_reward=1.0)

reward = reward_mgr.compute_reward(
    evicted_index=evicted_slot,
    memory=memory,
    new_state=candidate_vec,
)

print(f"Computed Reward: {reward:+.4f} | Current Alpha: {reward_mgr.get_current_alpha():.4f}")
```

---

## Cross-References

- For the theoretical foundation of curriculum reward transitions, see [Reward System Architecture](../architecture/reward-system.md).
- For integration within the online simulation, see [Online RL Simulation Reference](train-rl-online-simulation.md).

---

**Navigation:**
- Previous: [RL Agent Module API Reference](rl-agent.md)
- Up: [API Reference Index](README.md)
- Next: [Online RL Simulation Module API Reference](train-rl-online-simulation.md)

---
Licensed under the GNU General Public License v3.0. See [LICENSE](../../LICENSE) for details.
