# Documentation Index

> Part of the [Active Episodic Memory Management via Reinforcement Learning for Robust CNN Inference on Out-of-Distribution Data](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/README.md) documentation.  
> Parent: [Root README](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/README.md)

---

## Overview

This directory contains the complete technical documentation for the active semiparametric episodic memory architecture designed for robust Out-of-Distribution (OOD) routing and active cache curation in resource-constrained Edge AI systems. The documentation encompasses mathematical foundations, hardware budgets, tri-dataset empirical validations spanning three distinct complexity regimes:
1. **Low-Dimensional Stylized Regime:** MNIST handwritten digits (10 classes, custom 4-layer CNN with 225,034 parameters, Mahalanobis++ unit hypersphere covariance estimation).
2. **High-Dimensional Natural Regime:** CIFAR-10 natural RGB images (10 classes, ResNet-9 backbone with 6.57M parameters, 10-class Dual Uncertainty Arbiter fusing Mahalanobis distance with Shannon entropy).
3. **High-Entropy Fine-Grained Natural Regime:** CIFAR-100 fine-grained natural RGB images (100 classes, ResNet-18 V2 backbone with 11,250,532 parameters, 74.27% Top-1 accuracy, 100-class Dual Uncertainty Arbiter calibrated at $\tau_M = 8.69, \tau_H = 2.09\text{ nats}$).

All three vision backbones preserve an **invariant 128-dimensional latent bottleneck** ($z \in \mathbb{R}^{128}$), enabling identical non-parametric episodic memory banks ($C = 5,000$ slots in contiguous memory) and Reinforcement Learning decision agents (Double DQN with Prioritized Experience Replay). The documentation includes complete API contracts, procedural workflows, and journal publication benchmarks prepared for *Engineering Applications of Artificial Intelligence* (EAAI, Elsevier).

All technical documentation adheres to academic English standards, utilizes clean standard Markdown formatting, and strictly excludes decorative emojis.

---

## Documentation Map

The table below outlines all primary sections in the technical documentation hierarchy, their directory locations, scope, and status:

| Section | Directory Path | Description | Status |
|:---|:---|:---|:---|
| [Root Overview](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/README.md) | `.` | Top-level project abstract, dataset support matrix, side-by-side benchmark summary, architecture flowchart, and reproduction commands across MNIST, CIFAR-10, and CIFAR-100. | `[COMPLETED]` |
| [System Architecture](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/docs/architecture/README.md) | [`docs/architecture/`](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/docs/architecture/README.md) | Architectural deep-dives: CNN backbones (MNIST, CIFAR-10, and CIFAR-100 ResNet-18 V2), Mahalanobis++ & Dual Uncertainty OOD arbiters, episodic memory, and Double DQN agent. | `[COMPLETED]` |
| [Getting Started](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/docs/getting-started/README.md) | [`docs/getting-started/`](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/docs/getting-started/README.md) | Host prerequisites, environment installation, MNIST binary setup, CIFAR-10 & CIFAR-100 automated loaders, quickstart reproduction, and configuration registry. | `[COMPLETED]` |
| [Usage Guides](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/docs/guides/README.md) | [`docs/guides/`](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/docs/guides/README.md) | Step-by-step procedural workflows for the 5-stage training pipelines (MNIST, CIFAR-10, and CIFAR-100), baseline evaluations, and XAI visualization tools. | `[COMPLETED]` |
| [API Reference](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/docs/api/README.md) | [`docs/api/`](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/docs/api/README.md) | Technical reference documenting classes, tensor dimensions, methods, and complexity bounds for core modules, CIFAR-10, and CIFAR-100 packages. | `[COMPLETED]` |
| [CIFAR-10 Module API](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/docs/api/cifar10.md) | [`docs/api/cifar10.md`](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/docs/api/cifar10.md) | API specification for `RawModelCIFAR10`, `DualUncertaintyArbiter`, residual layers, and pure TensorFlow Adam optimizer. | `[COMPLETED]` |
| [CIFAR-100 Module API](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/docs/api/cifar100.md) | [`docs/api/cifar100.md`](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/docs/api/cifar100.md) | API specification for `RawModelCIFAR100V2` (11.25M params, 74.27% accuracy), legacy `RawModelCIFAR100`, `load_cifar100_backbone`, 100-class arbiter, and optimizers. | `[COMPLETED]` |
| [Experimental Results Hub](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/docs/results/README.md) | [`docs/results/`](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/docs/results/README.md) | Central experimental hub providing access to per-dataset benchmarks, telemetry, dashboards, and metrics definitions. | `[COMPLETED]` |
| [MNIST Baseline Benchmark](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/docs/results/baseline-comparison.md) | [`docs/results/baseline-comparison.md`](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/docs/results/baseline-comparison.md) | Quantitative 5-baseline evaluation on MNIST under 5 noise levels, cache hit rates, LMOS efficiency, and class balance. | `[COMPLETED]` |
| [CIFAR-10 Baseline Benchmark](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/docs/results/baseline-comparison-cifar10.md) | [`docs/results/baseline-comparison-cifar10.md`](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/docs/results/baseline-comparison-cifar10.md) | Quantitative 5-baseline evaluation on CIFAR-10, natural manifold dynamics, dual uncertainty routing, and eviction balance. | `[COMPLETED]` |
| [CIFAR-100 Baseline Benchmark](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/docs/results/baseline-comparison-cifar100.md) | [`docs/results/baseline-comparison-cifar100.md`](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/docs/results/baseline-comparison-cifar100.md) | Quantitative 5-baseline evaluation on CIFAR-100, fine-grained multi-class dynamics, anti-pollution gating, and class starvation elimination. | `[COMPLETED]` |
| [Cross-Dataset Synthesis](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/docs/results/cross-dataset-analysis.md) | [`docs/results/cross-dataset-analysis.md`](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/docs/results/cross-dataset-analysis.md) | Flagship scientific synthesis confronting MNIST vs. CIFAR-10 vs. CIFAR-100, side-by-side performance matrices, and core thesis validation. | `[COMPLETED]` |
| [Comparative Literature Validation](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/docs/results/literature-validation.md) | [`docs/results/literature-validation.md`](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/docs/results/literature-validation.md) | Systematic confrontation of empirical results against published scientific literature, root cause analysis, and statistical rigor across all three benchmarks. | `[COMPLETED]` |
| [Scientific Literature](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/docs/Literatura/README.md) | [`docs/Literatura/`](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/docs/Literatura/README.md) | Curated literature taxonomy, foundational research guidelines, and unified scientific glossary covering 12 domain pillars. | `[COMPLETED]` |
| [Historical Archive](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/docs/_archive/README.md) | [`docs/_archive/`](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/docs/_archive/README.md) | Historical pre-refactoring technical notes and architectural records preserved for developmental traceability. | `[ARCHIVED]` |

---

## Recommended Reading Order

For researchers and peer reviewers examining this codebase, the recommended progression guides through both the low-dimensional pilot, the natural image scaling, and the fine-grained high-entropy benchmark:

### Track A: Architectural Foundations
1. **[System Architecture](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/docs/architecture/README.md):** Understand the semiparametric design philosophy and the **invariant 128D latent bottleneck** linking heterogeneous CNN backbones to the shared episodic memory and RL governor.
2. **[CNN Feature Extractor](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/docs/architecture/cnn-feature-extractor.md):** Examine the custom from-scratch MNIST CNN (225k parameters), upgraded CIFAR-10 ResNet-9 backbone (6.57M parameters), and the promoted CIFAR-100 ResNet-18 V2 backbone (11.25M parameters, 74.27% accuracy).
3. **[Out-of-Distribution Detection](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/docs/architecture/ood-detection.md):** Review the Mahalanobis++ hypersphere detector on MNIST, the 10-class Dual Uncertainty Arbiter on CIFAR-10, and the calibrated 100-class Dual Uncertainty Arbiter ($\tau_M = 8.69, \tau_H = 2.09\text{ nats}$) on CIFAR-100.
4. **[End-to-End Data Flow](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/docs/architecture/data-flow.md):** Trace sample progression through inference, uncertainty gating, episodic rescue, and RL eviction.

### Track B: Setup and Execution
5. **[Installation & Quickstart](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/docs/getting-started/quickstart.md):** Set up runtime dependencies and verify the environment across all three datasets (MNIST, CIFAR-10, and CIFAR-100).
6. **[Configuration Reference](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/docs/getting-started/configuration.md):** Review global constants, dataset paths, memory bounds, and hyperparameter registries for MNIST, CIFAR-10, and CIFAR-100.
7. **[Training Pipeline Guide](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/docs/guides/training-pipeline.md):** Follow execution workflows for MNIST, CIFAR-10, and CIFAR-100 pipelines.
8. **[Running Evaluation](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/docs/guides/evaluation.md):** Execute the 5-baseline prequential streaming benchmark on MNIST, CIFAR-10, or CIFAR-100.

### Track C: Scientific Verification & Results
9. **[Cross-Dataset Scientific Synthesis](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/docs/results/cross-dataset-analysis.md):** Read the overarching synthesis comparing low-complexity, natural, and fine-grained regimes.
10. **[CIFAR-100 Benchmark Analysis](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/docs/results/baseline-comparison-cifar100.md):** Inspect detailed quantitative metrics, anti-pollution shielding, and class starvation elimination on 100 classes.
11. **[CIFAR-10 Benchmark Analysis](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/docs/results/baseline-comparison-cifar10.md):** Inspect detailed quantitative metrics, eviction balance, and Pareto trade-offs on natural images.
12. **[MNIST Benchmark Analysis](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/docs/results/baseline-comparison.md):** Examine the baseline results and extreme noise robustness on stylized digits.
13. **[Literature Validation](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/docs/results/literature-validation.md):** Review formal statistical testing (McNemar $p < 0.001$), confidence intervals, and alignment with peer-reviewed literature across MNIST, CIFAR-10, and CIFAR-100.

---

## Scientific Literature Repository

The [`docs/Literatura/`](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/docs/Literatura/README.md) directory preserves the curated scientific research foundation established prior to implementation:

- Curated repositories across **12 foundational AI pillars**: Active Perception, Deep Learning, Early Exit, Episodic Memory, Hopfield Networks, Out-of-Distribution Detection, Contextual Bandits, Q-Learning, Reinforcement Learning, RL + LLMs, Continual Learning, and Edge AI.
- A comprehensive [Unified Scientific Glossary](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/docs/Literatura/GLOSSARIO.md) providing formal definitions, theoretical foundations, and mathematical formulations.

> [!NOTE]
> The literature collection is maintained in its original Portuguese research format as an archival foundation and operates independently of the English technical documentation hierarchy.

---

**Navigation:**
- Up: [Root README](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/README.md)

---

Licensed under the GNU General Public License v3.0. See [LICENSE](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/LICENSE) for details.
