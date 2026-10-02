# Experimental Results

> Part of the [Active Episodic Memory Management via Reinforcement Learning for Robust CNN Inference on Out-of-Distribution Data](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/README.md) documentation.  
> Parent: [Documentation Index](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/docs/README.md) | Up: [Repository Root](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/README.md)

---

## 1. Overview and Executive Summary

This section presents the empirical validation and comparative benchmark of the proposed RL-driven active episodic memory architecture against four established system baselines. Conducted as the experimental core for submission to *Engineering Applications of Artificial Intelligence* (EAAI, Elsevier), this benchmark assesses system robustness, latency, memory bounds, catastrophic forgetting, and operational sustainability under severe non-stationary concept drift and out-of-distribution (OOD) sensory degradation across two distinct complexity regimes:
1. **Low-Dimensional Stylized Regime (MNIST):** 28x28x1 grayscale handwritten digits, using a custom 4-layer CNN (225k parameters).
2. **High-Dimensional Natural Regime (CIFAR-10):** 32x32x3 natural color images, using an upgraded ResNet-9 backbone (6.57M parameters).

### EAAI Benchmark: Key Metric Highlights

#### 1. MNIST Benchmark Summary
| Metric | B0 (Pure CNN) | B2 (FIFO) | B3 (LFU) | B4 (Proposed RL) |
|:---|:---:|:---:|:---:|:---:|
| Accuracy under Noise ($\sigma = 0.4$) | 51.40% | 51.50% | 51.50% | **55.70%** (+4.20%) |
| Accuracy under Noise ($\sigma = 0.6$) | 30.00% | 29.30% | 29.30% | **38.90%** (+9.60%) |
| Accuracy under Noise ($\sigma = 0.8$) | 20.30% | 17.60% | 17.60% | **27.30%** (+9.70%) |
| Overall Stream Accuracy | 58.10% | 57.18% | 57.18% | **61.66%** (+4.48%) |
| Cache Hit Rate ($k$-NN rescue) | 0.00% | 56.98% | 56.98% | **61.48%** (+4.50%) |
| Eviction KL Divergence (nats) | 0.0000 | 0.5003 | 0.5003 | **0.0028** ($177\times$ lower) |
| RAM Peak Allocation (MB) | 0.0008 MB | 7.47 MB | 7.47 MB | **9.92 MB** (Bounded) |
| Mean Processing Latency (ms) | 0.004 ms | 3.272 ms | 3.302 ms | **6.671 ms** (Real-time) |

#### 2. CIFAR-10 Benchmark Summary
| Metric | B0 (Pure CNN) | B2 (FIFO) | B3 (LFU) | B4 (Proposed RL) |
|:---|:---:|:---:|:---:|:---:|
| Clean Accuracy ($\sigma = 0.0$) | 91.40% | 92.10% | 92.10% | **92.20%** (+0.80% dividend) |
| Severe Noise Accuracy ($\sigma = 0.6$) | 10.00% | 9.60% | 9.60% | **10.10%** (+0.50%) |
| Extreme Noise Accuracy ($\sigma = 0.8$) | 10.70% | 11.20% | 11.50% | **11.90%** (+0.70%) |
| Overall Stream Accuracy | 27.14% | 27.14% | 27.20% | **27.50%** (Top performer) |
| Cache Hit Rate ($k$-NN rescue) | 0.00% | 13.16% | 13.23% | **13.60%** (+0.44%) |
| Eviction KL Divergence (nats) | 0.0000 | 0.8850 | 0.6144 | **0.0007** ($>1,200\times$ lower) |
| RAM Peak Allocation (MB) | 0.0008 MB | 7.48 MB | 7.48 MB | **9.93 MB** (Bounded) |
| Mean Processing Latency (ms) | 2.358 ms | 5.248 ms | 5.436 ms | **6.940 ms** (Real-time) |

---

## 2. Experimental Setup and Protocol

The benchmark is conducted in accordance with the streaming evaluation methodology formalized in [`evaluate_hybrid_global.py`](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/evaluate_hybrid_global.py) and [`scripts/cifar10/evaluate_baselines.py`](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/scripts/cifar10/evaluate_baselines.py).

### 2.1. Benchmark Datasets and Partitioning
- **MNIST:** $28 \times 28 \times 1$ grayscale images, seeded with 5,000 clean exemplars from training data; tested across 5,000 sequential test samples.
- **CIFAR-10:** $32 \times 32 \times 3$ natural RGB images, seeded with 5,000 clean prototypes ($k=10$); tested across 5,000 streaming test samples.
- **Noise Sweep:** Gaussian perturbation $\sigma \in \{0.0, 0.2, 0.4, 0.6, 0.8\}$ evaluated across 1,000 samples per tier.

### 2.2. Prequential Evaluation Protocol
Following Haug et al. (2022) and Wu et al. (2026), the evaluation runs in test-then-train mode:
1. **Predict:** Sample $x_t$ is mapped to $z_t \in \mathbb{R}^{128}$ and evaluated by the Uncertainty Arbiter. In-distribution samples are classified by the CNN; out-of-distribution or uncertain samples are routed to episodic memory where $k$-NN voting emits $\hat{y}_t$.
2. **Evaluate:** Prediction $\hat{y}_t$ is scored against ground-truth label $y_t$.
3. **Curate:** When an OOD or misclassification event occurs, the active RL controller considers buffer insertion. Under buffer saturation ($N = 5,000$), the agent selects an eviction action.

---

## 3. Evaluated System Baselines

1. **B0: Pure CNN (No Episodic Memory):** Monolithic CNN classifying all inputs directly via Softmax.
2. **B1: Infinite Memory Hybrid ($N \to \infty$):** Unbounded memory expansion establishing empirical retention ceilings.
3. **B2: Hybrid with Strict FIFO Eviction (`evict_oldest`):** Standard circular buffer eviction discarding oldest prototype.
4. **B3: Hybrid with LFU Eviction (`evict_least_frequently_used`):** Frequency-aware eviction discarding least queried prototype.
5. **B4: Hybrid with Active RL Eviction (Double DQN + PER, Proposed):** State-dependent eviction adaptively selecting among Action 0 (Ignore/Filter), Action 1 (FIFO), Action 2 (LFU), or Action 3 (Redundancy pruning).

---

## 4. Directory Structure and Sub-Documents

This documentation section is organized into three core analytical reports:

| Document | Primary Focus | Key Contents |
|:---|:---|:---|
| [Baseline Comparison Analysis (MNIST)](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/docs/results/baseline-comparison.md) | Quantitative comparison (MNIST) | Complete 5-baseline metrics table on MNIST, per-regime breakdown, comparative analysis, and statistical significance. |
| [Baseline Comparison Analysis (CIFAR-10)](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/docs/results/baseline-comparison-cifar10.md) | Quantitative comparison (CIFAR-10) | Complete 5-baseline metrics table on CIFAR-10, natural image manifold analysis, LMOS evaluation, and class preservation. |
| [Cross-Dataset Scientific Synthesis](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/docs/results/cross-dataset-analysis.md) | Flagship synthesis | Systematic confrontation between MNIST and CIFAR-10 complexity regimes, side-by-side matrices, McNemar test, and validation of 4 core theses. |
| [Engineering Metrics Reference](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/docs/results/metrics-reference.md) | Metric definitions | Mathematical formulations, physical units, algorithmic implementation mappings, and literature citations for all 7 metrics. |
| [Comparative Literature Validation](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/docs/results/literature-validation.md) | Literature confrontation | Systematic mapping of empirical findings against seminal publications, root cause analysis, statistical rigor validation, and EAAI submission guidelines. |

### Artifact Cross-References
- **MNIST Artifacts:**
  - Metrics JSON: [`outputs/mnist/eaai_metrics.json`](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/outputs/mnist/eaai_metrics.json)
  - Metrics CSV: [`outputs/mnist/eaai_metrics.csv`](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/outputs/mnist/eaai_metrics.csv)
  - Dashboard: [`outputs/mnist/eaai_evaluation_dashboard.png`](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/outputs/mnist/eaai_evaluation_dashboard.png)
- **CIFAR-10 Artifacts:**
  - Metrics JSON: [`outputs/cifar10/eaai_metrics.json`](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/outputs/cifar10/eaai_metrics.json)
  - Metrics CSV: [`outputs/cifar10/eaai_metrics.csv`](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/outputs/cifar10/eaai_metrics.csv)
  - Dashboard: [`outputs/cifar10/eaai_evaluation_dashboard.png`](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/outputs/cifar10/eaai_evaluation_dashboard.png)

---

## 5. Reproducing Experimental Results

```bash
# Execute evaluation on MNIST across all 5 baselines:
python evaluate_hybrid_global.py --dataset mnist --capacity 5000 --samples-per-level 1000 --output-dir outputs/mnist/

# Execute evaluation on CIFAR-10 across all 5 baselines:
python scripts/cifar10/evaluate_baselines.py --samples-per-level 1000 --noise-levels 0.0 0.2 0.4 0.6 0.8
```

For detailed options and parameter configurations, refer to the [Evaluation Guide](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/docs/guides/evaluation.md).

---

**Navigation:**
- Previous: [API Reference Index](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/docs/api/README.md)
- Up: [Documentation Index](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/docs/README.md)
- Next: [Baseline Comparison Analysis (MNIST)](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/docs/results/baseline-comparison.md)
- Also: [Baseline Comparison Analysis (CIFAR-10)](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/docs/results/baseline-comparison-cifar10.md) | [Cross-Dataset Analysis](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/docs/results/cross-dataset-analysis.md)

---

Licensed under the GNU General Public License v3.0. See [LICENSE](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/LICENSE) for details.
