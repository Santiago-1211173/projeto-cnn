# Episodic Memory Agent API Reference

> Part of the [Trustworthy Edge AI: RL-Driven Active Memory Management](../../README.md) documentation.  
> Parent: [API Reference Index](README.md) | Up: [Documentation Index](../README.md)

---

## Module Overview

The `src.models.knn_bandit_agent` module implements `KNNBanditAgent128D`, a high-performance instance-based episodic memory system tailored for resource-constrained Edge AI environments. Built entirely with contiguous pre-allocated NumPy arrays, it completely eliminates dynamic list resizing and memory fragmentation, ensuring strictly bounded RAM usage ($O(1)$ updates) and fast vectorized $k$-Nearest Neighbors retrieval using `np.argpartition`. In addition to standard retrieval, the agent implements three mechanical eviction policies (FIFO, LFU, and Redundancy) driven by active reinforcement learning.

- **Source File:** `src/models/knn_bandit_agent.py`
- **Import Statement:**
  ```python
  from src.models.knn_bandit_agent import KNNBanditAgent128D, KNNBanditAgent
  ```

---

## Class: `KNNBanditAgent128D`

```python
class KNNBanditAgent128D:
```

*Backward-compatible alias:* `KNNBanditAgent = KNNBanditAgent128D`

### Attributes

| Attribute | Type | Description |
|:----------|:-----|:------------|
| `capacity` | `int` | Maximum number of experience vectors stored in memory buffer |
| `k` | `int` | Default number of nearest neighbors queried during retrieval |
| `latent_dim` | `int` | Dimensionality of stored feature vectors (default: 128) |
| `n_actions` | `int` | Number of discrete action/class categories (default: 10) |
| `size` | `int` | Current number of active experiences stored ($0 \le \text{size} \le \text{capacity}$) |
| `tick_counter` | `int` | Monotonically increasing logical clock tracking insertion order |
| `_states` | `np.ndarray` | Flat array of shape `(capacity, latent_dim)` with `dtype=np.float32` |
| `_actions` | `np.ndarray` | Array of shape `(capacity,)` with `dtype=np.int32` storing class labels |
| `_rewards` | `np.ndarray` | Array of shape `(capacity,)` with `dtype=np.float32` storing historical feedback |
| `_insertion_ticks` | `np.ndarray` | Array of shape `(capacity,)` with `dtype=np.int64` storing logical insertion time |
| `_usage_counts` | `np.ndarray` | Array of shape `(capacity,)` with `dtype=np.int32` tracking retrieval hit frequency |

---

## Constructor

#### `__init__(capacity: int = MEMORY_CAPACITY, k: int = KNN_K_NEIGHBORS, latent_dim: int = LATENT_DIM, n_actions: int = KNN_N_ACTIONS, **kwargs: Any) -> None`

Initializes and pre-allocates contiguous memory blocks for representations and tracking metadata.

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `capacity` | `int` | `5000` | Maximum vector capacity ($C$) |
| `k` | `int` | `30` | Default neighborhood size for k-NN queries |
| `latent_dim` | `int` | `128` | Feature representation dimensionality ($D$) |
| `n_actions` | `int` | `10` | Number of classification labels / action indices |
| `**kwargs` | `Any` | `{}` | Absorbs legacy keyword arguments for backwards compatibility |

---

## Properties

#### `memory_size -> int`
Backward-compatible property returning `self.size`.

---

## Memory Insertion Methods

#### `add_experience(state: np.ndarray, action: int, reward: float) -> None`

Appends a single experience to memory. If the buffer is full (`size >= capacity`), automatically invokes `evict_oldest()` (FIFO) as the default mechanical eviction policy.

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `state` | `np.ndarray` | *required* | Feature vector of shape `(latent_dim,)` or `(1, latent_dim)` |
| `action` | `int` | *required* | Integer class label or action index |
| `reward` | `float` | *required* | Scalar feedback reward associated with experience |

#### `add_experience_batch(states: np.ndarray, actions: np.ndarray, rewards: np.ndarray) -> None`

Sequentially appends an array of experiences into memory.

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `states` | `np.ndarray` | *required* | Batch array of shape `(N, latent_dim)` |
| `actions` | `np.ndarray` | *required* | Array of class labels of shape `(N,)` |
| `rewards` | `np.ndarray` | *required* | Array of rewards of shape `(N,)` |

---

## Nearest Neighbor Retrieval Methods

#### `get_nearest_neighbors(query: np.ndarray, k: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]`

Executes vectorized Euclidean distance calculation over active memory slice `[:self.size]`. Uses `np.argpartition` for $\mathcal{O}(N)$ partition selection followed by sorting of the top-$k$ nearest neighbors.

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `query` | `np.ndarray` | *required* | Query feature vector of shape `(latent_dim,)` |
| `k` | `Optional[int]` | `None` | Number of neighbors to return; falls back to `self.k` |

**Returns:** `Tuple[np.ndarray, np.ndarray]`
- `indices`: `np.ndarray` of shape `(k,)` containing memory slot indices of the nearest neighbors.
- `distances`: `np.ndarray` of shape `(k,)` containing sorted Euclidean distances.

**Raises:** `RuntimeError` if called when `self.size == 0`.

---

## Active Eviction Policy Methods

#### `evict_oldest(new_state: np.ndarray, new_action: int, new_reward: float) -> int`

Implements First-In-First-Out (FIFO) eviction. Identifies the slot with the minimal insertion tick, overwrites it with new data, resets usage count to 0, advances `tick_counter`, and returns the target slot index:

$$\text{idx}^* = \arg\min_{i \in [0, \text{size}-1]} \text{insertion\_ticks}[i]$$

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `new_state` | `np.ndarray` | *required* | New latent vector to insert |
| `new_action` | `int` | *required* | New action label |
| `new_reward` | `float` | *required* | New reward value |

**Returns:** `int` -- Index of the overwritten slot in memory.

#### `evict_least_frequently_used(new_state: np.ndarray, new_action: int, new_reward: float) -> int`

Implements Least Frequently Used (LFU) eviction. Finds the slot with the minimal retrieval hit count. Breaks ties by selecting the oldest entry among candidates:

$$\text{idx}^* = \arg\min_{i \in \mathcal{C}_{\text{min}}} \text{insertion\_ticks}[i], \quad \mathcal{C}_{\text{min}} = \{j \mid \text{usage}[j] = \min(\text{usage})\}$$

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `new_state` | `np.ndarray` | *required* | New latent vector to insert |
| `new_action` | `int` | *required* | New action label |
| `new_reward` | `float` | *required* | New reward value |

**Returns:** `int` -- Index of the overwritten slot in memory.

#### `evict_most_redundant(new_state: np.ndarray, new_action: int, new_reward: float) -> int`

Implements Redundancy Eviction based on geometric density. Computes Euclidean distances between `new_state` and stored entries sharing the identical class label (`_actions == new_action`). Replaces the nearest geometric duplicate:

$$\text{idx}^* = \arg\min_{j \in \{i \mid \text{action}[i] = a_{\text{new}}\}} \|\mathbf{z}_{\text{new}} - \mathbf{z}_j\|_2$$

If no prior sample with label `new_action` exists, falls back to `evict_least_frequently_used()`.

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `new_state` | `np.ndarray` | *required* | New latent vector to insert |
| `new_action` | `int` | *required* | New action label |
| `new_reward` | `float` | *required* | New reward value |

**Returns:** `int` -- Index of the overwritten slot in memory.

---

## Decision & Value Estimation Methods

#### `increment_usage(indices: np.ndarray | List[int] | int) -> None`

Increments the retrieval counter for one or multiple memory slots.

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `indices` | `np.ndarray \| List[int] \| int` | *required* | Memory slot index or indices |

#### `build_index() -> None`

No-op method kept for backwards compatibility. Vectorized NumPy routines dynamically query memory slices without external tree builds.

#### `get_expected_rewards(state: np.ndarray) -> np.ndarray`

Estimates distance-weighted expected reward for each of the $A$ possible classes across the top-$k$ nearest neighbors:

$$Q(s, a) = \sum_{j \in \mathcal{N}_k(s), a_j = a} r_j \cdot \frac{1}{d_j + 10^{-8}}$$

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `state` | `np.ndarray` | *required* | Query state vector of shape `(latent_dim,)` |

**Returns:** `np.ndarray` of shape `(n_actions,)` with estimated expected rewards.

#### `get_expected_rewards_batch(states: np.ndarray) -> np.ndarray`

Batch version of `get_expected_rewards`. Returns array of shape `(N, n_actions)`.

#### `get_action(state: np.ndarray, epsilon: float = 0.0) -> int`

Selects action via $\epsilon$-greedy policy over expected rewards.

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `state` | `np.ndarray` | *required* | Query state vector |
| `epsilon` | `float` | `0.0` | Random exploration probability in $[0.0, 1.0]$ |

**Returns:** `int` -- Selected class / action index.

#### `get_action_batch(states: np.ndarray, epsilon: float = 0.0) -> np.ndarray`

Batch $\epsilon$-greedy action selection. Returns `np.ndarray` of shape `(N,)`.

---

## Serialization & Diagnostics

#### `save(path: str) -> None`

Compresses and serializes all memory arrays and metadata into a compressed `.npz` archive.

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `path` | `str` | *required* | Output path (e.g., `outputs/knn_memory_bank_128d.npz`) |

#### `load(path: str) -> None`

Restores memory buffer state, shapes, and metadata from a compressed `.npz` archive.

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `path` | `str` | *required* | Source `.npz` archive path |

#### `get_memory_stats() -> Dict[str, Any]`

Computes statistical diagnostics on current memory contents.

**Returns:** `Dict[str, Any]` containing:
- `"size"`: `int` (active vector count)
- `"capacity"`: `int` (maximum buffer capacity)
- `"occupancy_pct"`: `float` (percentage filled)
- `"reward_mean"`: `float` (mean stored reward)
- `"reward_positive_pct"`: `float` (percentage of rewards $> 0$)
- `"actions_distribution"`: `Dict[int, int]` (frequency distribution per class label)

---

## Usage Example

```python
# example_episodic_memory.py
import numpy as np
from src.models.knn_bandit_agent import KNNBanditAgent128D

# Initialize memory buffer
memory = KNNBanditAgent128D(capacity=500, k=10, latent_dim=128, n_actions=10)

# Insert 100 synthetic experiences
for i in range(100):
    vec = np.random.randn(128).astype(np.float32)
    label = i % 10
    memory.add_experience(vec, action=label, reward=1.0)

# Query nearest neighbors
query = np.random.randn(128).astype(np.float32)
indices, distances = memory.get_nearest_neighbors(query, k=5)
print(f"Nearest indices: {indices}")
print(f"Distances: {distances}")

# Predict class
pred_class = memory.get_action(query)
print(f"Predicted class: {pred_class}")

# Inspect statistics
stats = memory.get_memory_stats()
print(f"Occupancy: {stats['occupancy_pct']:.1f}% ({stats['size']}/{stats['capacity']})")
```

---

## Cross-References

- For the architectural rationale and benchmarks, see [Episodic Memory Architecture](../architecture/episodic-memory.md).
- For details on how the RL agent drives these eviction methods, see [RL Agent Architecture](../architecture/rl-agent.md).

---

**Navigation:**
- Previous: [Custom CNN Module API Reference](custom-cnn.md)
- Up: [API Reference Index](README.md)
- Next: [RL Agent Module API Reference](rl-agent.md)

---
Licensed under the GNU General Public License v3.0. See [LICENSE](../../LICENSE) for details.
