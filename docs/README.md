# Documentation Index

> Part of the [Trustworthy Edge AI: RL-Driven Active Memory Management](../README.md) documentation.  
> Parent: [Root README](../README.md)

---

## Overview

This directory contains the complete technical documentation for the RL-driven active episodic memory management system designed for robust Out-of-Distribution (OOD) routing in resource-constrained Edge AI environments. The documentation covers system architecture, installation and configuration procedures, reproduction guides, API references, and experimental benchmark results submitted to *Engineering Applications of Artificial Intelligence* (EAAI).

All technical documentation within this hierarchy is authored in academic English, strictly follows standard Markdown formatting, and excludes decorative emojis.

---

## Documentation Map

The table below outlines all sections of the technical documentation hierarchy, their directory locations, scope, and implementation status.

| Section | Directory Path | Description | Status |
|:--------|:---------------|:------------|:-------|
| [Root Overview](../README.md) | `.` | Top-level project introduction, abstract, key results, repository structure, and reproduction pipeline. | `[COMPLETED]` |
| [System Architecture](architecture/README.md) | `docs/architecture/` | Deep-dive architectural specifications for the CNN feature extractor, Mahalanobis++ OOD detector, k-NN episodic memory, Double DQN agent, reward system, and data flow. | `[COMPLETED]` |
| [Getting Started](getting-started/README.md) | `docs/getting-started/` | System prerequisites, dependency installation, raw MNIST dataset layout, quickstart verification, and configuration parameter reference. | `[PENDING]` |
| [Usage Guides](guides/README.md) | `docs/guides/` | Procedural workflows for the end-to-end training pipeline, online drift simulation, 5-baseline evaluation, and publication figure generation. | `[PENDING]` |
| [API Reference](api/README.md) | `docs/api/` | Comprehensive technical reference documenting classes, methods, data structures, tensor shapes, and type annotations for all core modules. | `[PENDING]` |
| [Experimental Results](results/README.md) | `docs/results/` | In-depth analysis of the 5-baseline benchmark across noise levels, latency/memory profiles, drift restoration, and metric definitions. | `[PENDING]` |
| [Scientific Literature](Literatura/README.md) | `docs/Literatura/` | Curated literature taxonomy, foundational research guidelines, and unified scientific glossary covering 12 domain pillars. | `[COMPLETED]` |

---

## Recommended Reading Order

For researchers and engineers examining this codebase for review, reproduction, or adaptation, the following reading progression is recommended:

1. **Architecture Overview (`architecture/README.md`):**  
   Understand the semiparametric design philosophy combining a parametric CNN with a non-parametric episodic memory buffer, mediated by an OOD routing gate.

2. **End-to-End Data Flow (`architecture/data-flow.md`):**  
   Trace the lifecycle of a tensor through feature extraction, covariance-based Mahalanobis scoring, threshold comparison, memory retrieval, and RL eviction selection.

3. **Getting Started (`getting-started/README.md`):**  
   Configure the local environment, verify required runtimes (Python 3.10+, PyTorch 2.0+, TensorFlow 2.10.1), and position the raw MNIST binary assets.

4. **Training Pipeline Guide (`guides/training-pipeline.md`):**  
   Follow the reproducible 5-step sequence to train the CNN, profile the latent space, populate the initial memory bank, execute online RL simulation, and run benchmark evaluation.

5. **API Reference (`api/README.md`):**  
   Review module interfaces, type contracts, and configuration hyperparameters when modifying model components or extending memory eviction heuristics.

6. **Experimental Results (`results/README.md`):**  
   Examine empirical findings comparing the RL active memory system (B4) against Parametric-only (B0), Unbounded Memory (B1), FIFO (B2), and LFU (B3) baselines.

---

## Scientific Literature Repository

The `docs/Literatura/` directory preserves the curated scientific research base established prior to implementation. It contains:

- Curated literature repositories across **12 foundational AI pillars**: Active Perception, Deep Learning, Early Exit, Episodic Memory, Hopfield Networks, Out-of-Distribution Detection, Contextual Bandits, Q-Learning, Reinforcement Learning, RL + LLMs, Continual Learning, and Edge AI.
- A comprehensive [Unified Scientific Glossary](Literatura/GLOSSARIO.md) providing formal definitions, theoretical foundations, and mathematical formulations from A to Z.
- Specialized research agent guidelines for each literature topic.

> [!NOTE]
> The literature collection is maintained in its original Portuguese research format as an archival foundation and operates independently of the English technical documentation hierarchy.

---

**Navigation:**
- Up: [Root README](../README.md)

---

Licensed under the GNU General Public License v3.0. See [LICENSE](../LICENSE) for details.
