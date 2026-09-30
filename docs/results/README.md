# Experimental Results

> Part of the [Trustworthy Edge AI: RL-Driven Active Memory Management](../../README.md) documentation.
> Parent: [Documentation Index](../README.md)

---

## 1. Overview and Executive Summary

This section presents the empirical validation and comparative benchmark of the proposed RL-driven active episodic memory architecture against four established system baselines. Conducted as the experimental core for submission to *Engineering Applications of Artificial Intelligence* (EAAI, Elsevier), this benchmark assesses system robustness, latency, memory bounds, catastrophic forgetting, and operational sustainability under severe non-stationary concept drift and out-of-distribution (OOD) sensory degradation.

Traditional deep neural networks exhibit catastrophic precision collapse and unwarranted overconfidence when deployed on edge devices subjected to sensor noise, lens occlusion, or environmental drift. The experimental findings documented herein demonstrate that combining a parametric feature extractor (custom convolutional neural network) with an active, capacity-bounded non-parametric episodic memory ($k$-NN bandit governed by Double DQN with Prioritized Experience Replay) mitigates performance collapse, maintaining operational efficacy under noise conditions that degrade standalone deep models by over 77%.

```
========================================================================================================
                           EAAI BENCHMARK: KEY METRIC HIGHLIGHTS
========================================================================================================
Metric                            B0 (Pure CNN)   B2 (FIFO)       B3 (LFU)        B4 (Proposed RL)
--------------------------------------------------------------------------------------------------------
Accuracy under Noise (sigma=0.4)   51.40%          51.50%          51.50%          55.70% (+4.20%)
Accuracy under Noise (sigma=0.6)   30.00%          29.30%          29.30%          38.90% (+9.60%)
Accuracy under Noise (sigma=0.8)   20.30%          17.60%          17.60%          27.30% (+9.70%)
Overall Stream Accuracy            58.10%          57.18%          57.18%          61.66% (+4.48%)
Cache Hit Rate (k-NN rescue)        0.00%          56.98%          56.98%          61.48% (+4.50%)
Eviction KL Divergence (nats)      0.0000          0.5003          0.5003          0.0028 (177x lower)
Forgetting Rate (%/transition)     19.42%          19.60%          19.60%          17.10% (-2.50%)
Drift Restoration Time (steps)     419.0           428.2           428.2           383.4  (-44.8 steps)
RAM Peak Allocation (MB)            0.0008 MB       7.47 MB         7.47 MB         9.92 MB (Bounded)
Mean Processing Latency (ms)        0.004 ms        3.272 ms        3.302 ms        6.671 ms (Real-time)
========================================================================================================
```

---

## 2. Experimental Setup and Protocol

The benchmark is conducted in accordance with the streaming evaluation methodology formalized in [evaluate_hybrid_global.py](../../evaluate_hybrid_global.py) and configured via [src/config.py](../../src/config.py).

### 2.1. Benchmark Dataset and Partitioning

- **Benchmark Dataset:** Standard MNIST handwritten digits ($28 \times 28$ grayscale images), representing a canonical spatial classification benchmark for validating semiparametric edge architectures.
- **Reference Seed Memory:** 5,000 clean exemplars extracted from the training partition (`x_train[:5000]`), mapped through the CNN to initialize the episodic memory bank with known ground-truth prototypes.
- **Evaluation Partition:** 5,000 independent samples drawn sequentially from the native test set (`t10k`), strictly segregated from all training and memory seeding stages.
- **Regime Disjunction:** 1,000 samples evaluated per noise tier, ensuring balanced statistical representation across all perturbation regimes.

### 2.2. Perturbation Regimes and Concept Drift Simulation

The streaming evaluation simulates progressive sensory corruption via additive Gaussian perturbation projected onto the valid input hypercube:

$$\tilde{x} = \operatorname{clip}(x + \mathcal{N}(0, \sigma^2 \mathbf{I}), 0.0, 1.0)$$

Evaluation traverses five distinct noise standard deviations:

| Regime Index | Noise Level ($\sigma$) | Sensory Interpretation | Physical Analogy |
|:---|:---:|:---|:---|
| Regime 0 | $\sigma = 0.0$ | Clean Baseline | Nominal sensor capture under optimal lighting. |
| Regime 1 | $\sigma = 0.2$ | Mild Noise | High ISO sensor grain, minor optical thermal noise. |
| Regime 2 | $\sigma = 0.4$ | Moderate Noise | Environmental degradation, low illumination, atmospheric blur. |
| Regime 3 | $\sigma = 0.6$ | Severe Noise | Partial lens obstruction, severe sensor distortion. |
| Regime 4 | $\sigma = 0.8$ | Extreme Noise | Destructive interference, near-total signal corruption. |

### 2.3. Prequential Evaluation Protocol

The evaluation adheres to the standard **prequential (test-then-train)** protocol for streaming machine learning (Haug et al., 2022; Wu et al., 2026):
1. **Predict:** For each arriving sample $x_t$, the system computes latent representation $z_t = f_\theta(x_t) \in \mathbb{R}^{128}$ and Mahalanobis distance $D_M(z_t)$. If $D_M(z_t) \le \tau$, the parametric CNN output $\hat{y}_t = \arg\max \operatorname{Softmax}(W z_t)$ is emitted; otherwise, the instance is routed to episodic memory where $k$-NN retrieval emits $\hat{y}_t = \operatorname{Mode}(\mathcal{N}_k(z_t))$.
2. **Evaluate:** The prediction $\hat{y}_t$ is scored against ground-truth label $y_t$.
3. **Curate:** If an OOD event occurs ($D_M(z_t) > \tau$) or the CNN prediction failed ($\hat{y}_t \neq y_t$), the memory controller considers admitting $z_t$. Under memory saturation ($\text{size} = \text{capacity}$), the active controller executes an eviction policy to maintain memory bounded at $C = 5,000$ vectors.

### 2.4. Hardware Constraints and Sizing

| Parameter | Value | Hardware / Algorithmic Significance |
|:---|:---|:---|
| Memory Capacity ($C$) | 5,000 | Strict RAM constraint for microcontroller / edge accelerator buffers. |
| Latent Dimension ($D$) | 128 | Compact bottleneck vector optimized for tensor core alignment. |
| $k$-Nearest Neighbors ($k$) | 30 | Neighborhood size for weighted distance consensus voting. |
| OOD Threshold ($\tau$) | 95.0 percentile | Calibrated on clean latent training distribution via Ledoit-Wolf shrinkage. |
| State Vector Dimension | 5 | Normalized features: Mahalanobis distance, entropy, nearest neighbor distance, error flag, RAM occupancy. |
| Action Space | 4 | Discrete actions: 0 (Ignore/Filter), 1 (FIFO), 2 (LFU), 3 (Redundant). |

---

## 3. Evaluated System Baselines

The benchmark contrasts five independent operational strategies:

1. **B0: Pure CNN (No Episodic Memory)**  
   The standalone convolutional neural network operates without edge caching or memory rescue. All samples are classified directly via final dense projection and Softmax, representing traditional monolithic edge deployment.

2. **B1: Infinite Memory Hybrid (Unbounded Theoretical Bound)**  
   A hybrid system with unrestricted memory expansion ($C = 50,000$). Every OOD sample and classification error is appended without eviction. This baseline establishes the theoretical performance ceiling and illustrates the memory leak hazard of unbounded caches on edge hardware.

3. **B2: Hybrid with Strict FIFO Eviction (`evict_oldest`)**  
   A capacity-bounded hybrid ($C = 5,000$) utilizing classical first-in, first-out circular buffer eviction. When full, the exemplar with the minimum insertion tick is discarded.

4. **B3: Hybrid with Least Frequently Used Eviction (`evict_least_frequently_used`)**  
   A capacity-bounded hybrid ($C = 5,000$) employing LFU cache replacement. When full, the exemplar accessed fewest times during $k$-NN rescue is discarded, breaking ties by oldest insertion tick.

5. **B4: Hybrid with RL Active Memory Management (Double DQN + PER, Proposed)**  
   A capacity-bounded hybrid ($C = 5,000$) where memory admission and eviction are orchestrated by a reinforcement learning agent trained via Curriculum Learning. The agent observes the 5D operational state and selects among filtering outliers (Action 0), blind temporal eviction (Action 1), access-frequency eviction (Action 2), or geometric class redundancy eviction (Action 3).

---

## 4. Key Scientific Breakthroughs

The experimental results validate four major hypotheses formulated for edge robotic and vision deployment:

1. **Robustness Under Severe Corruption:**  
   In severe noise regimes ($\sigma \ge 0.4$), the parametric CNN collapses to near-random accuracy ($30.00\%$ at $\sigma = 0.6$ and $20.30\%$ at $\sigma = 0.8$). The proposed active agent (B4) achieves **$38.90\%$** at $\sigma = 0.6$ and **$27.30\%$** at $\sigma = 0.8$, outperforming standard FIFO/LFU heuristics by **$+9.60\%$** and **$+9.70\%$** absolute accuracy, respectively.

2. **Mitigation of Cache Pollution:**  
   Blind eviction heuristics (B2 and B3) indiscriminately admit corrupted, uninformative representations, rapidly degrading memory bank fidelity and lowering retrieval precision to $56.98\%$. The RL agent learns to invoke **Action 0 (Ignore)** on destructive noise outliers, safeguarding clean exemplar clusters and elevating cache hit rate to **$61.48\%$**.

3. **Distribution Matching and Class Preservation:**  
   Under non-stationary streaming drift, blind heuristics disproportionately evict classes that appear less frequently in localized noise windows. B4 maintains an eviction distribution divergence of just **$0.0028\text{ nats}$** relative to a uniform class distribution—a **$177\times$ reduction** in distribution distortion compared to FIFO and LFU ($0.5003\text{ nats}$).

4. **Edge Operational Sustainability (LMOS):**  
   While the unbounded baseline (B1) accumulates **$35.20\text{ MB}$** of RAM without accuracy benefit ($57.20\%$ mean accuracy), B4 enforces a deterministic memory peak of **$9.92\text{ MB}$** (including the PyTorch DQN runtime and NumPy buffers). Per-inference latency remains strictly bounded at **$6.67\text{ ms}$**, satisfying real-time constraints ($>149\text{ frames/second}$) for embedded robotics.

---

## 5. Directory Structure and Sub-Documents

This documentation section is organized into the following specialized documents:

| Document | Primary Focus | Key Contents |
|:---|:---|:---|
| [Baseline Comparison Analysis](baseline-comparison.md) | Quantitative comparison | Complete 5-baseline metrics table, per-regime breakdown, comparative analysis, and statistical significance discussion. |
| [Engineering Metrics Reference](metrics-reference.md) | Metric definitions | Mathematical formulations, physical units, algorithmic implementation mappings, and literature citations for all 7 metrics. |
| [Comparative Literature Validation](literature-validation.md) | Literature confrontation | Systematic mapping of empirical findings against seminal publications, root cause analysis, statistical rigor validation, and EAAI submission guidelines. |

### Artifact Cross-References

The primary experimental data and visualization artifacts generated by the test runner are stored in the project outputs directory:

- Structured metrics JSON: [`outputs/eaai_metrics.json`](../../outputs/eaai_metrics.json)
- Tabular metrics CSV: [`outputs/eaai_metrics.csv`](../../outputs/eaai_metrics.csv)
- Publication 4-panel dashboard: [`outputs/eaai_evaluation_dashboard.png`](../../outputs/eaai_evaluation_dashboard.png)
- Training trajectory log: [`outputs/train_rl_simulation_log.csv`](../../outputs/train_rl_simulation_log.csv)

---

## 6. Reproducing Experimental Results

To reproduce all reported metrics and regenerate the publication dashboard using the pre-trained checkpoints:

```bash
# Execute evaluation across all 5 baselines with 1,000 samples per noise tier
python evaluate_hybrid_global.py --capacity 5000 --samples-per-level 1000 --output-dir outputs/
```

For detailed options, parameter configurations, and hardware sizing instructions, refer to the [Evaluation Guide](../guides/evaluation.md).

---

**Navigation:**
- Previous: [Online Simulation Reference](../api/train-rl-online-simulation.md)
- Up: [Documentation Index](../README.md)
- Next: [Baseline Comparison Analysis](baseline-comparison.md)
- Also: [Comparative Literature Validation](literature-validation.md)

---
Licensed under the GNU General Public License v3.0. See [LICENSE](../../LICENSE) for details.
