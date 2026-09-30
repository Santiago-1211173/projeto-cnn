# API Reference Index

> Part of the [Active Episodic Memory Management via Reinforcement Learning for Robust CNN Inference on Out-of-Distribution Data](../../README.md) documentation.  
> Parent: [Documentation Index](../README.md) | Up: [Repository Root](../../README.md)

---

## Overview

This directory provides the definitive application programming interface (API) reference for the core modules, classes, mathematical estimators, and orchestration pipelines implemented in the repository.

Every module specification documents:
- Concrete class hierarchies, constructor signatures, and parameter typing.
- Public method contracts, input/output tensor specifications, and failure handling.
- Memory allocation constraints, computational complexity guarantees, and Edge AI operational boundaries.
- Copy-pasteable Python code snippets verifying programmatic interoperability.

---

## Module Inventory

| Module Document | Source File | Key Public Classes / Functions | Primary Responsibility |
|:----------------|:------------|:-------------------------------|:-----------------------|
| [`config.md`](config.md) | `src/config.py` | Centralized Constants & Paths | Global system configuration, architectural dimensions, and hyperparameter registry |
| [`custom-cnn.md`](custom-cnn.md) | `src/models/custom_cnn.py` | `RawModel` | From-scratch CNN feature extractor yielding 128D latent representations and softmax posteriors |
| [`knn-bandit-agent.md`](knn-bandit-agent.md) | `src/models/knn_bandit_agent.py` | `KNNBanditAgent128D`, `KNNBanditAgent` | Capacity-bounded episodic memory buffer with vectorized k-NN and active eviction policies |
| [`rl-agent.md`](rl-agent.md) | `src/models/rl_agent.py` | `SumTree`, `PrioritizedReplayBuffer`, `QNetwork`, `RLAgent` | Double DQN active memory manager with Prioritized Experience Replay (PER) |
| [`reward-manager.md`](reward-manager.md) | `src/models/reward_manager.py` | `RewardManager` | Curriculum Learning reward orchestrator blending geometric proxy with validation accuracy |
| [`train-rl-online-simulation.md`](train-rl-online-simulation.md) | `training/train_rl_online_simulation.py` | `MahalanobisPlusPlus`, `OnlineStreamPipeline`, `TrainRLOnlineSimulation`, `inject_noise`, `run_simulation` | Mahalanobis++ OOD detector, streaming feature pipeline, and prequential drift simulation |

---

## Architectural Conventions

### 1. Static Allocation & Deterministic Memory (Edge AI Compliance)
To prevent `OutOfMemoryError` (OOM) and dynamic heap fragmentation on embedded Linux runtimes (e.g., Jetson Orin Nano, Raspberry Pi 5), dynamic list extensions (`list.append()`) are strictly banned in memory retrieval and replay paths.
- `KNNBanditAgent128D` pre-allocates static contiguous NumPy arrays for all states, actions, rewards, and tick counters at instantiation.
- `PrioritizedReplayBuffer` pre-allocates flat NumPy storage buffers and an array-based binary `SumTree`.
- `RewardManager` maintains a circular pre-allocated sliding validation buffer.

### 2. Computational Complexity Guarantees

| Operation | Component | Complexity | Mechanism |
|:----------|:----------|:-----------|:----------|
| Latent Feature Extraction | `RawModel` | $\mathcal{O}(B \cdot H \cdot W \cdot C)$ | Vectorized custom TensorFlow convolution layers |
| OOD Distance Computation | `MahalanobisPlusPlus` | $\mathcal{O}(N \cdot C \cdot D^2)$ | Batch quadratic form evaluation with regularized precision |
| Nearest Neighbor Retrieval | `KNNBanditAgent128D` | $\mathcal{O}(M \cdot D + k \log k)$ | Vectorized Euclidean distance + `np.argpartition` |
| Memory Slot Eviction | `KNNBanditAgent128D` | $\mathcal{O}(M)$ or $\mathcal{O}(M_c \cdot D)$ | In-place slot overwriting without array reallocation |
| Prioritized Transition Sampling | `PrioritizedReplayBuffer` | $\mathcal{O}(B \log C)$ | Binary `SumTree` cumulative priority descent |
| Q-Value Decision Inference | `RLAgent` | $\mathcal{O}(1)$ | 2-layer MLP forward pass ($\approx 9\text{k}$ FLOPs) |

### 3. Type Annotations and Standards
All public interfaces adhere to Python 3.10+ typing standards via `from __future__ import annotations`, using explicit unions (`Union[A, B]` or `A | B`), typed NumPy arrays, and PyTorch / TensorFlow tensor signatures.

---

## Navigation Order

For structured inspection of the component APIs, follow this logical progression:
1. [`config.md`](config.md) -- Understand all global constants and memory limits.
2. [`custom-cnn.md`](custom-cnn.md) -- Inspect the feature extraction front-end.
3. [`knn-bandit-agent.md`](knn-bandit-agent.md) -- Review the episodic memory data structure.
4. [`rl-agent.md`](rl-agent.md) -- Review the active Double DQN decision engine.
5. [`reward-manager.md`](reward-manager.md) -- Examine the curriculum reward calculation.
6. [`train-rl-online-simulation.md`](train-rl-online-simulation.md) -- Explore the prequential streaming simulation engine.

---

**Navigation:**
- Up: [Documentation Index](../README.md)
- Next: [Configuration Module API Reference](config.md)

---
Licensed under the GNU General Public License v3.0. See [LICENSE](../../LICENSE) for details.
