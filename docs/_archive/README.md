# Historical Documentation Archive

> Part of the [Active Episodic Memory Management via Reinforcement Learning for Robust CNN Inference on Out-of-Distribution Data](../../README.md) documentation.  
> Parent: [Documentation Index](../README.md)

---

## Overview

This directory preserves the original pre-refactoring technical documentation, development notes, and operational logs created during early engineering phases. All technical contents, empirical findings, and mathematical specifications from these files have been unified, modernized, and expanded within the official English documentation hierarchy.

These files are retained as historical records for full development traceability.

---

## Archived Files Inventory

| Archived File | Original Topic | Superceded By |
|:--------------|:---------------|:--------------|
| [`arquitetura_e_fluxo_cnn.md`](arquitetura_e_fluxo_cnn.md) | Original CNN architecture and layer parameters | [`docs/architecture/cnn-feature-extractor.md`](../architecture/cnn-feature-extractor.md) |
| [`fluxo_dados_sistema_hibrido.md`](fluxo_dados_sistema_hibrido.md) | Hybrid inference data flow and state machine | [`docs/architecture/data-flow.md`](../architecture/data-flow.md) |
| [`knn_bandit_agent_128d.md`](knn_bandit_agent_128d.md) | 128D k-NN episodic memory and eviction heuristics | [`docs/architecture/episodic-memory.md`](../architecture/episodic-memory.md) & [`docs/api/knn-bandit-agent.md`](../api/knn-bandit-agent.md) |
| [`train_rl_128d.md`](train_rl_128d.md) | Phase 2 RL Double DQN training notes | [`docs/architecture/rl-agent.md`](../architecture/rl-agent.md) & [`docs/api/rl-agent.md`](../api/rl-agent.md) |
| [`train_rl_online_simulation.md`](train_rl_online_simulation.md) | Phase 3 online simulation and chaos injection | [`docs/guides/training-pipeline.md`](../guides/training-pipeline.md) & [`docs/api/train-rl-online-simulation.md`](../api/train-rl-online-simulation.md) |
| [`evaluate_hybrid_global.md`](evaluate_hybrid_global.md) | Hybrid system global evaluation protocol | [`docs/guides/evaluation.md`](../guides/evaluation.md) & [`docs/results/baseline-comparison.md`](../results/baseline-comparison.md) |
| [`caso_de_estudo_eaai.md`](caso_de_estudo_eaai.md) | Case study notes for EAAI submission | [`docs/results/baseline-comparison.md`](../results/baseline-comparison.md) & [`docs/results/metrics-reference.md`](../results/metrics-reference.md) |

---

**Navigation:**
- Up: [Documentation Index](../README.md)

---

Licensed under the GNU General Public License v3.0. See [LICENSE](../../LICENSE) for details.
