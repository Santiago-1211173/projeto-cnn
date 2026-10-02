# API Reference Index

> Part of the [Active Episodic Memory Management via Reinforcement Learning for Robust CNN Inference on Out-of-Distribution Data](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/README.md) documentation.  
> Parent: [Documentation Index](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/docs/README.md) | Up: [Repository Root](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/README.md)

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

| Module Document | Source File(s) | Key Public Classes / Functions | Primary Responsibility |
|:---|:---|:---|:---|
| [`config.md`](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/docs/api/config.md) | `src/config.py` | Centralized Constants & Paths | Global system configuration, architectural dimensions, and hyperparameter registry |
| [`custom-cnn.md`](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/docs/api/custom-cnn.md) | `src/models/custom_cnn.py` | `RawModel` | From-scratch MNIST CNN feature extractor yielding 128D latent representations and Softmax posteriors |
| [`cifar10.md`](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/docs/api/cifar10.md) | `src/cifar10/*.py` | `RawModelCIFAR10`, `DualUncertaintyArbiter`, `ResidualBlock`, `Adam` | ResNet-9 backbone (91.18% acc), Dual Uncertainty Arbiter, layer primitives, and from-scratch Adam optimizer |
| [`knn-bandit-agent.md`](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/docs/api/knn-bandit-agent.md) | `src/models/knn_bandit_agent.py` | `KNNBanditAgent128D`, `KNNBanditAgent` | Capacity-bounded episodic memory buffer with vectorized k-NN and active eviction policies |
| [`rl-agent.md`](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/docs/api/rl-agent.md) | `src/models/rl_agent.py` | `SumTree`, `PrioritizedReplayBuffer`, `QNetwork`, `RLAgent` | Double DQN active memory manager with Prioritized Experience Replay (PER) |
| [`reward-manager.md`](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/docs/api/reward-manager.md) | `src/models/reward_manager.py` | `RewardManager` | Curriculum Learning reward orchestrator blending geometric proxy with validation accuracy |
| [`train-rl-online-simulation.md`](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/docs/api/train-rl-online-simulation.md) | `training/train_rl_online_simulation.py` | `MahalanobisPlusPlus`, `OnlineStreamPipeline`, `TrainRLOnlineSimulation`, `inject_noise`, `run_simulation` | Mahalanobis++ OOD detector, streaming feature pipeline, and prequential drift simulation |

---

## Architectural Conventions

### 1. Static Allocation & Deterministic Memory (Edge AI Compliance)
To prevent `OutOfMemoryError` (OOM) and dynamic heap fragmentation on embedded Linux runtimes (e.g., Jetson Orin Nano, Raspberry Pi 5), dynamic list extensions (`list.append()`) are strictly banned in memory retrieval and replay paths.
- `KNNBanditAgent128D` pre-allocates static contiguous NumPy arrays for all states, actions, rewards, and tick counters at instantiation.
- `PrioritizedReplayBuffer` pre-allocates flat NumPy storage buffers and an array-based binary `SumTree`.
- `RewardManager` maintains a circular pre-allocated sliding validation buffer.

### 2. Computational Complexity Guarantees

| Operation | Component | Complexity | Mechanism |
|:---|:---|:---|:---|
| Latent Feature Extraction (MNIST) | `RawModel` | $\mathcal{O}(B \cdot H \cdot W \cdot C)$ | Vectorized custom TensorFlow convolution layers |
| Latent Feature Extraction (CIFAR-10) | `RawModelCIFAR10` | $\mathcal{O}(B \cdot \sum_l H_l W_l C_l^2)$ | Residual blocks + Global Average Pooling |
| OOD Distance Computation | `MahalanobisPlusPlus` / `DualUncertaintyArbiter` | $\mathcal{O}(N \cdot C \cdot D^2)$ | Batch quadratic form evaluation with regularized precision |
| Predictive Entropy Computation | `DualUncertaintyArbiter` | $\mathcal{O}(N \cdot C)$ | Vectorized Shannon entropy evaluation |
| Nearest Neighbor Retrieval | `KNNBanditAgent128D` | $\mathcal{O}(M \cdot D + k \log k)$ | Vectorized Euclidean distance + `np.argpartition` |
| Memory Slot Eviction | `KNNBanditAgent128D` | $\mathcal{O}(M)$ or $\mathcal{O}(M_c \cdot D)$ | In-place slot overwriting without array reallocation |
| Prioritized Transition Sampling | `PrioritizedReplayBuffer` | $\mathcal{O}(B \log C)$ | Binary `SumTree` cumulative priority descent |
| Q-Value Decision Inference | `RLAgent` | $\mathcal{O}(1)$ | 2-layer MLP forward pass ($\approx 9\text{k}$ FLOPs) |

### 3. Type Annotations and Standards
All public interfaces adhere to Python 3.10+ typing standards via `from __future__ import annotations`, using explicit unions (`Union[A, B]` or `A | B`), typed NumPy arrays, and PyTorch / TensorFlow tensor signatures.

---

## Navigation Progression

For structured inspection of component APIs, follow this logical progression:
1. [`config.md`](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/docs/api/config.md) -- Review all global constants, dataset paths, and memory limits.
2. [`custom-cnn.md`](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/docs/api/custom-cnn.md) -- Inspect the MNIST 4-layer feature extraction front-end.
3. [`cifar10.md`](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/docs/api/cifar10.md) -- Inspect the CIFAR-10 ResNet-9 backbone and Dual Uncertainty Arbiter.
4. [`knn-bandit-agent.md`](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/docs/api/knn-bandit-agent.md) -- Review the episodic memory data structure.
5. [`rl-agent.md`](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/docs/api/rl-agent.md) -- Review the active Double DQN decision engine.
6. [`reward-manager.md`](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/docs/api/reward-manager.md) -- Examine the curriculum reward calculation.
7. [`train-rl-online-simulation.md`](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/docs/api/train-rl-online-simulation.md) -- Explore the prequential streaming simulation engine.

---

**Navigation:**
- Up: [Documentation Index](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/docs/README.md)
- Next: [Configuration Module API Reference](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/docs/api/config.md)

---

Licensed under the GNU General Public License v3.0. See [LICENSE](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/LICENSE) for details.
