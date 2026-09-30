# Deep Reinforcement Learning Agent API Reference

> Part of the [Active Episodic Memory Management via Reinforcement Learning for Robust CNN Inference on Out-of-Distribution Data](../../README.md) documentation.  
> Parent: [API Reference Index](README.md) | Up: [Documentation Index](../README.md)

---

## Module Overview

The `src.models.rl_agent` module implements the active memory management decision engine for the semiparametric system. It provides a lightweight Double Deep Q-Network (Double DQN) with Prioritized Experience Replay (PER), backed by an array-based binary `SumTree`. Operating on a normalized 5-dimensional telemetry state vector, the agent outputs discrete memory eviction decisions to curate the episodic buffer during runtime concept drift.

- **Source File:** `src/models/rl_agent.py`
- **Import Statement:**
  ```python
  from src.models.rl_agent import SumTree, PrioritizedReplayBuffer, QNetwork, RLAgent
  ```

---

## Class: `SumTree`

```python
class SumTree:
```

An array-based binary sum tree providing $\mathcal{O}(\log C)$ transition priority updates and cumulative priority sampling.

### Tree Layout Convention
For a buffer with leaf capacity $C$:
- Total internal array length: $2C - 1$
- Leaf nodes reside in range: $[C - 1, 2C - 2]$
- Tree root (index $0$) stores total cumulative priority: $\sum_{i} p_i^\alpha$
- Left child of node $i$: $2i + 1$, Right child: $2i + 2$, Parent: $\lfloor (i - 1) / 2 \rfloor$

### Constructor

#### `__init__(capacity: int) -> None`

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `capacity` | `int` | *required* | Maximum number of leaf nodes / stored transitions ($C$) |

### Public Methods & Properties

#### `total_priority -> float` (Property)
Returns the root value representing the sum of all leaf priorities.

#### `update(tree_idx: int, priority: float) -> None`
Updates the priority of a leaf node and propagates the difference upward to the root.

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `tree_idx` | `int` | *required* | Node index within tree array $[C-1, 2C-2]$ |
| `priority` | `float` | *required* | New priority value $p_i^\alpha$ |

#### `get_leaf(value: float) -> Tuple[int, float, int]`
Traverses the tree to locate the leaf corresponding to cumulative priority query `value`.

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `value` | `float` | *required* | Priority sample query in $[0, \text{total\_priority}]$ |

**Returns:** `Tuple[int, float, int]`
- `tree_idx`: Array index in the tree structure.
- `priority`: Scalar priority stored at the leaf.
- `data_idx`: Corresponding index in the replay data buffer in $[0, C - 1]$.

---

## Class: `PrioritizedReplayBuffer`

```python
class PrioritizedReplayBuffer:
```

Proportional Prioritized Experience Replay buffer utilizing `SumTree` and contiguous NumPy storage arrays.

### Constructor

#### `__init__(capacity: int = REPLAY_BUFFER_CAPACITY, state_dim: int = RL_STATE_DIM, alpha: float = 0.6, beta_0: float = 0.4, beta_increment: float = 1e-4, epsilon: float = 1e-5) -> None`

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `capacity` | `int` | `10000` | Maximum transition capacity |
| `state_dim` | `int` | `5` | State feature dimension |
| `alpha` | `float` | `0.6` | Priority exponent $P(i) \propto p_i^\alpha$ |
| `beta_0` | `float` | `0.4` | Initial importance sampling weight exponent |
| `beta_increment` | `float` | `1e-4` | Linear increment applied to $\beta$ per sampling call |
| `epsilon` | `float` | `1e-5` | Small positive constant ensuring $p_i > 0$ |

### Public Methods

#### `store(state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None`
Inserts a transition into the circular buffer with maximal priority ($p_{\max}^\alpha$) ensuring initial exploration.

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `state` | `np.ndarray` | *required* | Starting state of shape `(state_dim,)` |
| `action` | `int` | *required* | Action index |
| `reward` | `float` | *required* | Feedback reward |
| `next_state` | `np.ndarray` | *required* | Resulting state of shape `(state_dim,)` |
| `done` | `bool` | *required* | Episode termination flag |

#### `sample(batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]`
Samples a prioritized mini-batch and computes normalized importance sampling weights:

$$w_i = \left(N \cdot P(i)\right)^{-\beta} / \max_j w_j$$

**Returns:** `Tuple` containing:
1. `states`: `np.ndarray` of shape `(batch_size, state_dim)`
2. `actions`: `np.ndarray` of shape `(batch_size,)`
3. `rewards`: `np.ndarray` of shape `(batch_size,)`
4. `next_states`: `np.ndarray` of shape `(batch_size, state_dim)`
5. `dones`: `np.ndarray` of shape `(batch_size,)`
6. `weights`: `np.ndarray` of shape `(batch_size,)`
7. `tree_indices`: `np.ndarray` of shape `(batch_size,)`

#### `update_priorities(tree_indices: np.ndarray, td_errors: np.ndarray) -> None`
Updates transition priorities using absolute Temporal Difference errors: $p_i = |\delta_i| + \epsilon$.

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `tree_indices` | `np.ndarray` | *required* | Indices returned by `sample()` |
| `td_errors` | `np.ndarray` | *required* | Absolute TD errors $|\hat{Q} - y|$ |

---

## Class: `QNetwork`

```python
class QNetwork(nn.Module):
```

Lightweight Multi-Layer Perceptron (MLP) for Q-value estimation, designed for deterministic sub-millisecond forward passes on Edge CPUs/GPUs.

### Architecture

```text
Linear(in_features=state_dim, out_features=64) -> ReLU()
Linear(in_features=64, out_features=64)        -> ReLU()
Linear(in_features=64, out_features=n_actions)
```

### Constructor

#### `__init__(state_dim: int = RL_STATE_DIM, n_actions: int = RL_N_ACTIONS) -> None`

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `state_dim` | `int` | `5` | Dimensionality of state representation |
| `n_actions` | `int` | `4` | Number of discrete output actions |

#### `forward(x: torch.Tensor) -> torch.Tensor`
Executes forward pass. Returns tensor of shape `(B, n_actions)`.

---

## Class: `RLAgent`

```python
class RLAgent:
```

Double DQN Agent coordinating policy execution, target network synchronization, and prioritized experience updates.

### Constructor

#### `__init__(state_dim: int = RL_STATE_DIM, n_actions: int = RL_N_ACTIONS, lr: float = 1e-3, gamma: float = 0.99, target_update_interval: int = 100, buffer_capacity: int = REPLAY_BUFFER_CAPACITY, device: Optional[str] = None) -> None`

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `state_dim` | `int` | `5` | State vector dimension |
| `n_actions` | `int` | `4` | Action cardinality |
| `lr` | `float` | `1e-3` | Adam optimizer learning rate |
| `gamma` | `float` | `0.99` | Bellman discount factor |
| `target_update_interval` | `int` | `100` | Frequency (gradient steps) of target network synchronization |
| `buffer_capacity` | `int` | `10000` | PER buffer capacity |
| `device` | `Optional[str]` | `None` | Execution device (`"cpu"` or `"cuda"`) |

---

## Public Methods

#### `get_state_vector(mahalanobis_dist: float, local_entropy: float, min_knn_dist: float, prediction_error: float, ram_occupancy: float) -> np.ndarray`

Normalizes telemetry signals into a unit-bounded 5D state vector with `dtype=np.float32`.

$$\mathbf{s} = \left[ \min\left(\frac{d_M}{50}, 1\right), \min\left(\frac{H}{\ln 10}, 1\right), \min\left(\frac{d_{\text{min}}}{10}, 1\right), \text{clip}(e, 0, 1), \text{clip}\left(\frac{\text{size}}{\text{cap}}, 0, 1\right) \right]^T$$

| Parameter | Type | Default | Scaling Boundary | Description |
|:----------|:-----|:--------|:-----------------|:------------|
| `mahalanobis_dist` | `float` | *required* | $[0, 50.0]$ | Mahalanobis++ distance from class centroids |
| `local_entropy` | `float` | *required* | $[0, \ln(10) \approx 2.3026]$ | Shannon entropy of k-NN neighborhood |
| `min_knn_dist` | `float` | *required* | $[0, 10.0]$ | Euclidean distance to nearest neighbor in memory |
| `prediction_error` | `float` | *required* | $[0, 1.0]$ | Binary misclassification flag or probability delta |
| `ram_occupancy` | `float` | *required* | $[0, 1.0]$ | Memory buffer fill ratio ($\text{size} / \text{capacity}$) |

**Returns:** `np.ndarray` of shape `(5,)` and `dtype=np.float32`.

#### `select_action(state: np.ndarray, epsilon: float = 0.1) -> int`

Selects an action using an $\epsilon$-greedy exploration policy.

| Action Index | Semantic Action | Eviction Mechanism Invoked in `KNNBanditAgent128D` |
|:-------------|:----------------|:---------------------------------------------------|
| `0` | Ignore | None (reject incoming sample from episodic memory) |
| `1` | FIFO | `evict_oldest()` |
| `2` | LFU | `evict_least_frequently_used()` |
| `3` | Redundant | `evict_most_redundant()` |

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `state` | `np.ndarray` | *required* | 5D state vector of shape `(5,)` |
| `epsilon` | `float` | `0.1` | Exploration rate in $[0.0, 1.0]$ |

**Returns:** `int` in $\{0, 1, 2, 3\}$.

#### `store_transition(state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None`

Forwards experience tuple into internal Prioritized Experience Replay buffer.

#### `update_weights(batch_size: int = 32) -> Optional[float]`

Executes one gradient optimization step using Double Q-learning with importance-weighted Smooth L1 (Huber) loss:

$$y_i = r_i + (1 - d_i) \gamma Q_{\text{target}}\left(s'_i, \arg\max_{a'} Q_{\text{policy}}(s'_i, a')\right)$$
$$\mathcal{L} = \frac{1}{B} \sum_{i=1}^B w_i \cdot \text{Smooth}_{L1}\left(Q_{\text{policy}}(s_i, a_i) - y_i\right)$$

Gradients are clipped at `max_norm=10.0`. Synchronizes `target_net` every `target_update_interval` steps.

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `batch_size` | `int` | `32` | Number of transitions sampled from PER |

**Returns:** `Optional[float]` -- Scalar loss value, or `None` if buffer contains fewer than `batch_size` transitions.

#### `save(path: str) -> None`
Serializes policy network, target network, optimizer state, and training step count to a `.pt` checkpoint file.

#### `load(path: str) -> None`
Restores agent weights and optimizer state from a `.pt` checkpoint file.

#### `get_agent_stats() -> Dict[str, Any]`
Returns operational telemetry dictionary:
- `"train_step"`: `int`
- `"buffer_size"`: `int`
- `"buffer_capacity"`: `int`
- `"current_beta"`: `float`
- `"max_priority"`: `float`

---

## Usage Example

```python
# example_rl_agent.py
import numpy as np
from src.models.rl_agent import RLAgent

# Instantiate agent on CPU
agent = RLAgent(state_dim=5, n_actions=4, lr=1e-3, device="cpu")

# Construct state vector
s = agent.get_state_vector(
    mahalanobis_dist=14.2,
    local_entropy=0.85,
    min_knn_dist=2.1,
    prediction_error=1.0,
    ram_occupancy=0.98,
)

# Select action
action = agent.select_action(s, epsilon=0.05)
print(f"Chosen Action: {action} (0=Ignore, 1=FIFO, 2=LFU, 3=Redundant)")

# Simulate transition storage and learning step
s_next = agent.get_state_vector(12.1, 0.45, 2.5, 0.0, 0.98)
agent.store_transition(s, action, reward=0.8, next_state=s_next, done=False)

# When buffer has >= 32 samples:
loss = agent.update_weights(batch_size=32)
```

---

## Cross-References

- For the full mathematical derivation of Double DQN and PER, see [RL Agent Architecture](../architecture/rl-agent.md).
- For curriculum reward interactions, see [Reward System Architecture](../architecture/reward-system.md).

---

**Navigation:**
- Previous: [Episodic Memory Agent API Reference](knn-bandit-agent.md)
- Up: [API Reference Index](README.md)
- Next: [Reward Manager Module API Reference](reward-manager.md)

---
Licensed under the GNU General Public License v3.0. See [LICENSE](../../LICENSE) for details.
