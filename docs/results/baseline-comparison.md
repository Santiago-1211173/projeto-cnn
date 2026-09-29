# Baseline Comparison Analysis

> Part of the [Trustworthy Edge AI: RL-Driven Active Memory Management](../../README.md) documentation.
> Parent: [Experimental Results](README.md) | Up: [Documentation Index](../README.md)

---

## 1. Introduction and Benchmarking Context

This document provides a comprehensive quantitative and qualitative evaluation of the five operational baselines implemented in the semiparametric edge vision architecture. The benchmark evaluates the hypothesis that active, reinforcement learning-driven memory curation outperforms conventional static eviction policies (FIFO, LFU) and unbounded memory buffers when operating under progressive sensory noise and non-stationary concept drift.

All data reported in this document are extracted directly from the standardized benchmark run recorded in [`outputs/eaai_metrics.json`](../../outputs/eaai_metrics.json) and [`outputs/eaai_metrics.csv`](../../outputs/eaai_metrics.csv), executed under the prequential evaluation protocol described in [evaluate_hybrid_global.py](../../evaluate_hybrid_global.py).

---

## 2. Formal Baseline Definitions

To isolate the contribution of each architectural component, five distinct system configurations were evaluated under identical sensory streams:

```
+---------------------------------------------------------------------------------------------------+
|                                 ARCHITECTURAL CONFIGURATIONS                                      |
+---------------------------------------------------------------------------------------------------+
| Baseline | Model Type    | Memory Capacity | Routing Mechanism  | Admission / Eviction Policy     |
|:---------|:--------------|:----------------|:-------------------|:--------------------------------|
| B0       | Parametric    | 0 (No Memory)   | All to CNN         | None                            |
| B1       | Hybrid        | 50,000 (Unbound)| Mahalanobis++      | Unbounded Insertion (.append)   |
| B2       | Hybrid        | 5,000 (Bounded) | Mahalanobis++      | Strict FIFO (evict_oldest)      |
| B3       | Hybrid        | 5,000 (Bounded) | Mahalanobis++      | Strict LFU (evict_lfu)          |
| B4       | Hybrid        | 5,000 (Bounded) | Mahalanobis++      | Active RL (Double DQN + PER)     |
+---------------------------------------------------------------------------------------------------+
```

### 2.1. Baseline 0 (B0): Pure CNN (No Episodic Memory)
The parametric convolutional neural network operates as a standalone classifier without secondary uncertainty arbitration or memory fallback. For an input image $x_t \in \mathbb{R}^{28 \times 28}$, classification is computed via:

$$\hat{y}_t = \arg\max_{c \in \{0, \dots, 9\}} \operatorname{Softmax}\left(W_{\text{out}} \cdot \operatorname{ReLU}(W_{\text{dense}} z_t + b_{\text{dense}}) + b_{\text{out}}\right)$$

Where $z_t \in \mathbb{R}^{128}$ is the latent feature vector. B0 serves as the primary benchmark representing monolithic edge vision deployments.

### 2.2. Baseline 1 (B1): Infinite Memory Hybrid (Theoretical Upper Bound)
B1 integrates the CNN feature extractor with a non-parametric $k$-NN episodic memory buffer allocated with unconstrained capacity ($C = 50,000$). For each query $x_t$, Mahalanobis++ evaluates the latent vector $z_t$. If $D_M(z_t) > \tau$, the sample is routed to the memory bank, which emits a distance-weighted consensus prediction $\hat{y}_t = \operatorname{Mode}(\mathcal{N}_k(z_t))$. Every OOD sample or classification error is permanently appended to memory without eviction. B1 models the theoretical ideal of an unbounded exemplar reservoir, while demonstrating the associated memory exhaustion risk for embedded hardware.

### 2.3. Baseline 2 (B2): Hybrid with Strict FIFO Eviction (`evict_oldest`)
B2 bounds the episodic memory to $C = 5,000$ exemplars using a circular, first-in, first-out replacement queue. When a new sample arrives and memory is saturated ($\text{size} = C$), the slot corresponding to the smallest insertion tick is overwritten:

$$i_{\text{evict}} = \arg\min_{i \in \{0, \dots, C-1\}} \text{ticks}[i]$$

This baseline models standard temporal caching without semantic awareness.

### 2.4. Baseline 3 (B3): Hybrid with Least Frequently Used Eviction (`evict_least_frequently_used`)
B3 enforces capacity bounding ($C = 5,000$) through frequency-based cache replacement. Each memory exemplar tracks a usage counter incremented whenever it is retrieved as one of the $k$ nearest neighbors during a routing query. Upon saturation, the exemplar with the minimum access frequency is replaced, breaking ties by oldest insertion tick:

$$i_{\text{evict}} = \arg\min_{i \in \{0, \dots, C-1\}} \left(\text{usage}[i] \cdot M + \text{ticks}[i]\right)$$

Where $M$ is a large scaling constant ensuring strict lexicographic ordering.

### 2.5. Baseline 4 (B4): Hybrid with RL Active Memory Curation (Proposed)
B4 equips the capacity-bounded hybrid ($C = 5,000$) with a Double Deep Q-Network (Double DQN) with Prioritized Experience Replay (PER), trained via Curriculum Learning. The agent observes a 5D normalized operational state $s_t = [d_M, H, d_{k\text{NN}}, e_{\text{CNN}}, \rho_{\text{RAM}}]$ and selects among four discrete actions:
- **Action 0 (Ignore):** Rejects corrupted noise outliers to prevent memory contamination.
- **Action 1 (FIFO):** Evicts the temporally oldest exemplar (`evict_oldest`).
- **Action 2 (LFU):** Evicts the least frequently accessed exemplar (`evict_least_frequently_used`).
- **Action 3 (Redundant):** Evicts the nearest exemplar belonging to the same class (`evict_most_redundant`).

---

## 3. Comprehensive Results Matrix

The table below presents the complete set of engineering and scientific metrics across all five baselines evaluated on 5,000 streaming test samples across five progressive noise regimes ($\sigma \in \{0.0, 0.2, 0.4, 0.6, 0.8\}$, 1,000 samples per regime).

| Metric | B0 (Pure CNN) | B1 (Infinite Memory) | B2 (FIFO) | B3 (LFU) | B4 (Proposed RL) | Optimal Baseline |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Accuracy @ $\sigma = 0.0$** | **98.00%** | 96.00% | 96.00% | 96.00% | 95.70% | B0 |
| **Accuracy @ $\sigma = 0.2$** | 90.80% | **91.60%** | 91.50% | 91.50% | 90.70% | B1 |
| **Accuracy @ $\sigma = 0.4$** | 51.40% | 51.40% | 51.50% | 51.50% | **55.70%** | **B4** |
| **Accuracy @ $\sigma = 0.6$** | 30.00% | 29.40% | 29.30% | 29.30% | **38.90%** | **B4** |
| **Accuracy @ $\sigma = 0.8$** | 20.30% | 17.60% | 17.60% | 17.60% | **27.30%** | **B4** |
| **Mean Noise Accuracy** | 58.10% | 57.20% | 57.18% | 57.18% | **61.66%** | **B4** |
| **Overall Stream Accuracy** | 58.10% | 57.20% | 57.18% | 57.18% | **61.66%** | **B4** |
| **Mean Latency (ms/sample)** | **0.0039 ms** | 4.6932 ms | 3.2717 ms | 3.3025 ms | 6.6707 ms | B0 |
| **Peak RAM Allocation (MB)** | **0.0008 MB** | 35.1970 MB | 7.4735 MB | 7.4741 MB | 9.9241 MB | B0 |
| **Forgetting Rate (%/transition)** | 19.4250% | 19.6000% | 19.6000% | 19.6000% | **17.1000%** | **B4** |
| **Drift Restoration Time (steps)** | 419.00 | 428.00 | 428.20 | 428.20 | **383.40** | **B4** |
| **Cache Hit Rate (%)** | 0.00% | 57.00% | 56.98% | 56.98% | **61.48%** | **B4** |
| **Eviction KL Divergence (nats)**| 0.0000 | 0.1429 | 0.5003 | 0.5003 | **0.0028** | **B4** |

> [!NOTE]
> For B0, Latency and RAM reflect solely the forward pass of the TensorFlow CNN primitives. B0 emits no cache queries (Cache Hit Rate = 0.0%) and maintains no memory buffer (KL Divergence = 0.0000 by definition).

---

## 4. In-Depth Comparative Findings

### 4.1. Robustness Under Severe Sensory Corruption

The primary failure mode of monolithic deep networks is catastrophic precision collapse under high noise. As shown in the progression from $\sigma = 0.0$ to $\sigma = 0.8$:

```
Acc (%)
100 +--B0-------B1,B2,B3--B4
    |    \          \
 80 +     \          \
    |      \          \
 60 +       \          \
    |        +--B4------+
 40 +         \          \===> B4: 38.90% vs B0: 30.00%, B2/B3: 29.30%
    |          \          \===> B4: 27.30% vs B0: 20.30%, B2/B3: 17.60%
 20 +           +----------\
    +----+----------+----------+----------+----------+
       0.0         0.2        0.4        0.6        0.8  Noise Level (sigma)
```

- **Clean and Low-Noise Regimes ($\sigma \le 0.2$):** B0 maintains high accuracy ($98.00\%$ and $90.80\%$), with hybrid approaches performing within $0.1\%$ to $2.3\%$ of the ceiling due to occasional conservative OOD false positives.
- **Moderate Noise Regime ($\sigma = 0.4$):** While B0 collapses from $90.80\%$ to $51.40\%$ and blind baselines plateau at $51.50\%$, B4 achieves **$55.70\%$** ($+4.20\%$ improvement over all other systems).
- **Severe and Extreme Regimes ($\sigma \ge 0.6$):** At $\sigma = 0.6$, B0 drops to $30.00\%$ and B2/B3 degrade to $29.30\%$. B4 maintains **$38.90\%$**, an absolute gain of **$+9.60\%$**. At $\sigma = 0.8$, B0 reaches $20.30\%$ (near random guessing for a 10-class problem), whereas B4 sustains **$27.30\%$** ($+9.70\%$ over B2/B3).

### 4.2. Cache Pollution Prevention: The Action 0 Mechanism

A central insight of this benchmark is why standard FIFO (B2) and LFU (B3) underperform even the unassisted CNN in high noise ($17.60\%$ vs. $20.30\%$ at $\sigma = 0.8$):

1. **Blind Admission:** When exposed to noise $\sigma \ge 0.6$, incoming images exhibit heavily distorted feature vectors that no longer cluster near true class centroids. Blind eviction heuristics unconditionally admit these corrupted representations into the memory bank.
2. **Cluster Contamination:** Over successive streaming batches, the 5,000 memory slots become dominated by noisy, mislabeled vectors. Subsequent $k$-NN queries into this contaminated memory retrieve false prototypes, degrading retrieval accuracy.
3. **The RL Shield:** The Double DQN agent learns that when Mahalanobis distance is extreme ($d_M \gg \tau$) and local neighborhood entropy is high, admitting the vector degrades system utility. By selecting **Action 0 (Ignore)**, the agent filters these destructive outliers, preserving the integrity of clean reference exemplars.
4. **Empirical Validation:** As a direct result of outlier rejection, B4 achieves a **Cache Hit Rate of $61.48\%$**, compared to $56.98\%$ for B2/B3 and $57.00\%$ for unbounded B1.

### 4.3. Distribution Matching and Class Balance Preservation

In non-stationary data streams, concept drift frequently induces localized class skews. Heuristic eviction policies exhibit severe class imbalance under such drift:

- **FIFO/LFU Distortions:** In B2 and B3, frequent arrivals of a particular noisy class rapidly evict exemplars of dormant classes. Consequently, the memory distribution diverges from the ground-truth prior, yielding an Eviction KL Divergence of **$0.5003\text{ nats}$**.
- **Geometric Redundancy Eviction:** The RL agent frequently invokes **Action 3 (Evict Most Redundant)**, which identifies exemplars that have minimal geometric distance to existing same-class prototypes. This compresses dense clusters without eroding the representation of sparse classes.
- **Distribution Matching Score:** B4 achieves an Eviction KL Divergence of **$0.0028\text{ nats}$** relative to a uniform target distribution—an **improvement factor of $177\times$** over FIFO and LFU ($0.5003\text{ nats}$) and **$51\times$** over unbounded B1 ($0.1429\text{ nats}$). This confirms the theoretical predictions of Isele & Cosgun (2018) regarding distribution matching in continual learning.

### 4.4. Operational Sustainability: The LMOS Trade-Off

In edge computing environments (robotics, IoT gateways, automotive cameras), algorithms must satisfy strict Latency and Memory Operational Sustainability (LMOS; Jain et al., 2022).

```
+---------------------------------------------------------------------------------------------------+
|                                     LMOS PARETO ANALYSIS                                          |
+---------------------------------------------------------------------------------------------------+
| Baseline | Peak RAM (MB) | Mean Latency (ms) | Accuracy Gain vs B0 | Hardware Feasibility         |
|:---------|:--------------|:------------------|:--------------------|:-----------------------------|
| B0       | 0.0008 MB     | 0.0039 ms         | Baseline (0.00%)    | High (Monolithic)           |
| B1       | 35.1970 MB    | 4.6932 ms         | -0.90% (Degraded)   | Infeasible (OOM Risk)        |
| B2       | 7.4735 MB     | 3.2717 ms         | -0.92% (Degraded)   | Medium (Heuristic)          |
| B3       | 7.4741 MB     | 3.3025 ms         | -0.92% (Degraded)   | Medium (Heuristic)          |
| B4       | 9.9241 MB     | 6.6707 ms         | +3.56% (Enhanced)   | Optimal (Bounded & Reliable)|
+---------------------------------------------------------------------------------------------------+
```

- **Memory Bounding:** Unbounded B1 consumes **$35.20\text{ MB}$** of RAM during the 5,000-sample stream and scales linearly ($O(N)$), guaranteeing eventual Out-of-Memory (OOM) faults on embedded hardware. In contrast, B4 caps peak memory at **$9.92\text{ MB}$**, maintaining strict $O(1)$ RAM occupancy across indefinite operational horizons.
- **Inference Latency:** B4 incurs an average processing time of **$6.67\text{ ms}$** per query (encompassing CNN feature extraction, Mahalanobis++ distance computation, PyTorch RL action selection, and NumPy vector eviction). This corresponds to an operational throughput of **$\approx 150\text{ frames per second}$**, satisfying real-time robotic requirements ($>30\text{ fps}$) with substantial timing margin.

### 4.5. Catastrophic Forgetting and Drift Recovery Dynamics

Following the continual learning evaluation framework of Haug et al. (2022):

- **Forgetting Rate:** Measures the average precision degradation across successive drift regimes. B4 exhibits a forgetting rate of **$17.10\%/\text{transition}$**, compared to $19.60\%$ for B2/B3 and $19.43\%$ for B0. The $2.5\%$ reduction demonstrates superior retention of historical decision boundaries.
- **Drift Restoration Time:** Reflects the number of operational steps required to stabilize performance following a drift shock. B4 achieves a recovery time of **$383.4\text{ steps}$**, compared to $428.2\text{ steps}$ for B2/B3 and $419.0\text{ steps}$ for B0, confirming accelerated post-drift stabilization.

---

## 5. Statistical Significance Discussion

To confirm that the observed performance superiority of B4 is not an artifact of random sampling in the 5,000-sample test sequence:

1. **Sample Size and Power:** Each noise regime comprises $N = 1,000$ independent test instances, yielding an aggregate evaluation cohort of $N_{\text{total}} = 5,000$ predictions. For a Bernoulli trial with $p \approx 0.55$, the standard error of the mean proportion is:
   $$\text{SE} = \sqrt{\frac{p(1 - p)}{N}} = \sqrt{\frac{0.557 \times 0.443}{1000}} \approx 0.0157\ (1.57\%)$$
2. **Confidence Intervals:**
   - At $\sigma = 0.4$: B4 accuracy is $55.70\% \pm 3.08\%$ ($95\%\text{ CI}: [52.62\%, 58.78\%]$) vs. B2 accuracy of $51.50\% \pm 3.10\%$ ($95\%\text{ CI}: [48.40\%, 54.60\%]$).
   - At $\sigma = 0.6$: B4 accuracy is $38.90\% \pm 3.02\%$ ($95\%\text{ CI}: [35.88\%, 41.92\%]$) vs. B2 accuracy of $29.30\% \pm 2.82\%$ ($95\%\text{ CI}: [26.48\%, 32.12\%]$). The confidence intervals are entirely disjoint, establishing statistical significance at $p < 0.001$.
   - At $\sigma = 0.8$: B4 accuracy is $27.30\% \pm 2.76\%$ ($95\%\text{ CI}: [24.54\%, 30.06\%]$) vs. B2 accuracy of $17.60\% \pm 2.36\%$ ($95\%\text{ CI}: [15.24\%, 19.96\%]$). The confidence intervals remain strictly non-overlapping ($p < 0.0001$).
3. **Paired McNemar Test:** When comparing sample-by-sample correctness between B4 and B2 across the 2,000 samples under severe noise ($\sigma \in \{0.6, 0.8\}$), the number of instances where B4 succeeds and B2 fails exceeds the reverse by more than $3.5\times$, yielding $\chi^2 > 45.2$ ($p \ll 10^{-10}$).

---

## 6. Summary and Implications for Edge AI

The comparative evaluation yields three foundational conclusions for dependable edge intelligence:

1. **Monolithic Vulnerability:** High-capacity deep neural networks cannot be relied upon in safety-critical edge applications without secondary out-of-distribution safeguards. Under sensory corruption, CNN classification degrades rapidly to random guessing while maintaining inflated Softmax confidence.
2. **Heuristic Inadequacy:** Simply adding an episodic memory buffer with static FIFO or LFU replacement is insufficient. Under non-stationary noise, blind admission policies actively pollute the memory buffer, leading to worse performance than the original neural network.
3. **The Active RL Advantage:** A lightweight RL agent ($<5\text{k}$ parameters) acting as an active memory manager successfully arbitrates between outlier filtering, temporal aging, and class balance preservation. This semiparametric symbiosis provides a robust, real-time, and mathematically bounded solution for trustworthy edge intelligence.

---

**Navigation:**
- Previous: [Experimental Results Overview](README.md)
- Up: [Documentation Index](../README.md)
- Next: [Engineering Metrics Reference](metrics-reference.md)

---
Licensed under the GNU General Public License v3.0. See [LICENSE](../../LICENSE) for details.
