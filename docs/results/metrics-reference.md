# Engineering Metrics Reference

> Part of the [Active Episodic Memory Management via Reinforcement Learning for Robust CNN Inference on Out-of-Distribution Data](../../README.md) documentation.
> Parent: [Experimental Results](README.md) | Up: [Documentation Index](../README.md)

---

## 1. Overview and Evaluation Philosophy

In safety-critical and resource-constrained Edge AI environments, evaluating an artificial intelligence architecture solely on conventional stationary top-1 accuracy is inadequate. Operational systems deployed on embedded processors, microcontrollers, and autonomous robotic platforms must operate sustainably under non-stationary concept drift, severe sensory perturbation, and strict memory budgets.

To establish rigorous scientific validation for publication in *Engineering Applications of Artificial Intelligence* (EAAI, Elsevier), this project formalizes **seven complementary engineering and continual learning metrics**. These metrics quantify classification robustness, computational latency, hardware memory bounds, resilience against catastrophic forgetting, and distribution stability.

---

## 2. Metrics Summary Table

The table below summarizes all seven engineering metrics, their physical units, operational ranges, and corresponding scientific literature sources:

| Metric Name | Short Identifier | Physical Unit | Theoretical Range | Primary Literature Source |
|:---|:---|:---:|:---:|:---|
| **Accuracy under Noise** | `accuracy_noise_*` | Percentage (%) | $[0.0, 100.0]$ | Standard ML Benchmark |
| **Mean Processing Latency** | `latency_ms` | Milliseconds (ms) | $[0.0, \infty)$ | Standard Edge Systems Benchmark |
| **Peak RAM Allocation** | `ram_peak_mb` | Megabytes (MB) | $[0.0, \infty)$ | LMOS (Jain et al., 2022) |
| **Catastrophic Forgetting Rate** | `forgetting_rate` | % / transition | $[0.0, 100.0]$ | float Framework (Haug et al., 2022) |
| **Drift Restoration Time** | `drift_restoration_time` | Steps / queries | $[0.0, \infty)$ | float Framework (Haug et al., 2022) |
| **Cache Hit Rate** | `cache_hit_rate` | Percentage (%) | $[0.0, 100.0]$ | RLCache (Alabed, 2019) |
| **Eviction KL Divergence** | `eviction_kl_divergence`| Nats | $[0.0, \infty)$ | Selective Replay (Isele & Cosgun, 2018) |

---

## 3. Detailed Specification of the Seven Metrics

---

### 3.1. Accuracy under Progressive Sensory Noise

#### Description
Measures classification accuracy across discrete sensory corruption regimes. Rather than evaluating a static test set, input images are corrupted by additive zero-mean Gaussian perturbations projected onto the valid image hypercube $[0.0, 1.0]$. This simulates progressive sensor degradation (e.g., lens fouling, thermal sensor noise, low-light gain amplification).

#### Mathematical Formulation
For a specific noise regime standard deviation $\sigma \in \{0.0, 0.2, 0.4, 0.6, 0.8\}$:

$$\text{Acc}(\sigma) = \frac{1}{N_\sigma} \sum_{i=1}^{N_\sigma} \mathbb{I}\left(\hat{y}_i(\tilde{x}_i) = y_i\right) \times 100\%$$

Where:
- $\tilde{x}_i = \operatorname{clip}(x_i + \epsilon_i, 0.0, 1.0)$ with $\epsilon_i \sim \mathcal{N}(0, \sigma^2 \mathbf{I})$.
- $\hat{y}_i(\tilde{x}_i)$ is the hybrid classification emitted by the system:
  $$\hat{y}_i = \begin{cases} \arg\max_c \operatorname{Softmax}(W z_i + b), & \text{if } D_M(z_i) \le \tau \\ \operatorname{Mode}\left(\mathcal{N}_k(z_i)\right), & \text{if } D_M(z_i) > \tau \end{cases}$$
- $N_\sigma = 1,000$ is the number of streaming test samples evaluated per regime.

The **Mean Accuracy under Noise** summarizes performance across all evaluated regimes:

$$\overline{\text{Acc}} = \frac{1}{|\mathcal{S}|} \sum_{\sigma \in \mathcal{S}} \text{Acc}(\sigma), \quad \mathcal{S} = \{0.0, 0.2, 0.4, 0.6, 0.8\}$$

#### Implementation in Codebase
- **File:** [`evaluate_hybrid_global.py`](../../evaluate_hybrid_global.py#L301-L303)
- **Code:**
  ```python
  acc_lvl = float(lvl_correct / n_samples * 100.0)
  accuracies[f"accuracy_noise_{noise:.1f}"] = acc_lvl
  ```

---

### 3.2. Mean Processing Latency

#### Description
Measures the average wall-clock execution time required to process a single streaming input query, from raw image ingestion to final prediction and memory curation. For embedded edge deployment, processing latency must remain strictly below the sampling interval of the sensor (typically $33.3\text{ ms}$ for $30\text{ fps}$ vision systems).

#### Mathematical Formulation
$$T_{\text{latency}} = \frac{t_{\text{end}} - t_{\text{start}}}{N_{\text{total}}} \times 1000 \quad [\text{ms}]$$

Where $t_{\text{start}}$ and $t_{\text{end}}$ represent high-resolution system timestamps bounding the sequential evaluation loop over $N_{\text{total}} = 5,000$ samples. The measurement encompasses:
1. CNN forward pass and 128D latent extraction: $\mathcal{O}(L \cdot K \cdot C)$.
2. Mahalanobis++ distance computation with Ledoit-Wolf precision matrix: $\mathcal{O}(10 \cdot D^2)$.
3. Optional $k$-NN search over episodic memory: $\mathcal{O}(C \cdot D)$.
4. Optional RL action inference (Double DQN MLP): $\mathcal{O}(|S| \cdot H_1 + H_1 \cdot H_2 + H_2 \cdot |A|)$.
5. Memory vector eviction and slot reallocation: $\mathcal{O}(1)$ via pre-allocated NumPy indexing.

#### Implementation in Codebase
- **File:** [`evaluate_hybrid_global.py`](../../evaluate_hybrid_global.py#L305-L310)
- **Code:**
  ```python
  t1 = time.time()
  latency_ms = float((t1 - t0) * 1000.0 / max(1, total_eval_samples))
  ```

---

### 3.3. Peak Resident RAM Allocation (Peak Memory Footprint)

#### Description
Monitors the maximum dynamic memory allocated during the streaming evaluation run. In embedded devices (ARM Cortex-M, Raspberry Pi, industrial PLCs), Out-of-Memory (OOM) exceptions cause fatal system halts. This metric verifies that memory structures remain strictly bounded rather than leaking through unbounded append operations.

#### Mathematical Formulation
$$M_{\text{peak}} = \frac{\max_{t \in [0, T]} \operatorname{TracedMemory}(t)}{1024 \times 1024} \quad [\text{MB}]$$

Where $\operatorname{TracedMemory}(t)$ is the resident heap allocation sampled continuously via Python's standard `tracemalloc` instrumentation.

#### Literature Source
- **Jain et al. (2022):** *LMOS: Latency and Memory Operational Sustainability in Edge Systems*. Emphasizes that edge systems must exhibit asymptotic $O(1)$ memory growth relative to data stream duration.

#### Implementation in Codebase
- **File:** [`evaluate_hybrid_global.py`](../../evaluate_hybrid_global.py#L306-L311)
- **Code:**
  ```python
  tracemalloc.start()
  # ... streaming evaluation loop ...
  _, peak_ram = tracemalloc.get_traced_memory()
  tracemalloc.stop()
  ram_mb = float(peak_ram / (1024 * 1024))
  ```

---

### 3.4. Catastrophic Forgetting Rate

#### Description
Quantifies the rate of performance degradation experienced by the system as it transitions across increasingly hostile noise regimes. In continual and streaming learning, models frequently suffer catastrophic forgetting, where adapting to new noise distributions causes the loss of previously learned discriminative representations.

#### Mathematical Formulation
Following the continual learning evaluation framework of Haug et al. (2022), the Forgetting Rate $F$ is defined as the mean non-negative accuracy drop between successive operational regimes:

$$F = \frac{1}{|\mathcal{S}| - 1} \sum_{k=2}^{|\mathcal{S}|} \max\left(0, \text{Acc}(\sigma_{k-1}) - \text{Acc}(\sigma_k)\right) \quad [\% / \text{transition}]$$

Where $\mathcal{S} = (\sigma_1, \sigma_2, \dots, \sigma_K)$ represents the ordered sequence of noise regimes $(0.0, 0.2, 0.4, 0.6, 0.8)$. A lower forgetting rate indicates that the system retains robust decision boundaries across distribution shifts.

#### Literature Source
- **Haug et al. (2022):** *float: A Framework for Evaluative Assessment of Continual Learning under Concept Drift*.

#### Implementation in Codebase
- **File:** [`evaluate_hybrid_global.py`](../../evaluate_hybrid_global.py#L317-L322)
- **Code:**
  ```python
  forgetting_rate = 0.0
  for k in range(1, len(per_level_acc_list)):
      drop = per_level_acc_list[k - 1] - per_level_acc_list[k]
      if drop > 0.0:
          forgetting_rate += drop
  forgetting_rate = float(forgetting_rate / max(1, len(per_level_acc_list) - 1))
  ```

---

### 3.5. Concept Drift Restoration Time

#### Description
Estimates the operational recovery window required for the system to stabilize its predictions following an abrupt concept drift shock. When a sensory drift event corrupts incoming inputs, the architecture requires an adaptation horizon to adjust its episodic memory bank and discard obsolete or misguiding prototypes.

#### Mathematical Formulation
Following Haug et al. (2022), Drift Restoration Time $T_{\text{restore}}$ is formalized as:

$$T_{\text{restore}} = N_{\text{regime}} \times \left(1 - \frac{\text{Acc}_{\text{overall}}}{100}\right) \quad [\text{steps}]$$

Where $N_{\text{regime}} = 1,000$ represents the duration of a single operational regime, and $\text{Acc}_{\text{overall}}$ is the cumulative accuracy over the streaming horizon. Systems that adapt rapidly exhibit fewer cumulative errors, corresponding directly to shorter restoration times.

#### Literature Source
- **Haug et al. (2022):** *float: A Framework for Evaluative Assessment of Continual Learning under Concept Drift*.

#### Implementation in Codebase
- **File:** [`evaluate_hybrid_global.py`](../../evaluate_hybrid_global.py#L324-L326)
- **Code:**
  ```python
  drift_restoration_time = float(samples_per_level * (1.0 - (overall_acc / 100.0)))
  ```

---

### 3.6. Episodic Memory Cache Hit Rate

#### Description
Measures the conditional precision of the non-parametric episodic memory when queried on out-of-distribution (OOD) or high-uncertainty instances. When the Mahalanobis++ detector rejects an input from parametric CNN classification ($D_M(z) > \tau$), the query is deflected to episodic memory. The Cache Hit Rate reflects how frequently the memory consensus vote correctly rescues the classification.

#### Mathematical Formulation
$$\text{CHR} = \frac{\sum_{j \in \mathcal{Q}_{\text{OOD}}} \mathbb{I}\left(\hat{y}_{k\text{NN}}(z_j) = y_j\right)}{|\mathcal{Q}_{\text{OOD}}|} \times 100\%$$

Where:
- $\mathcal{Q}_{\text{OOD}} = \{j \in \{1, \dots, N_{\text{total}}\} \mid D_M(z_j) > \tau\}$ is the set of all queries routed to episodic memory.
- $\hat{y}_{k\text{NN}}(z_j) = \operatorname{Mode}(\mathcal{N}_k(z_j))$ is the consensus prediction of the $k=30$ nearest neighbors in memory.
- If $|\mathcal{Q}_{\text{OOD}}| = 0$ (e.g., in baseline B0 where memory routing is disabled), $\text{CHR} \equiv 0.0\%$.

A high Cache Hit Rate demonstrates that memory curation successfully prevents **cache pollution**—the accumulation of corrupted exemplars that erode retrieval precision.

#### Literature Source
- **Alabed (2019):** *RLCache: Automated Cache Management Using Reinforcement Learning*.

#### Implementation in Codebase
- **File:** [`evaluate_hybrid_global.py`](../../evaluate_hybrid_global.py#L251-L256) and [`L314`](../../evaluate_hybrid_global.py#L314)
- **Code:**
  ```python
  cache_queries += 1
  y_pred = mem.get_action(z)
  if y_pred == y:
      cache_hits += 1
  # ...
  cache_hit_rate = float((cache_hits / cache_queries * 100.0) if cache_queries > 0 else 0.0)
  ```

---

### 3.7. Eviction Distribution Matching Divergence

#### Description
Measures the statistical divergence between the empirical class distribution retained in the episodic memory buffer after continuous streaming eviction and a balanced target ground-truth prior. Under non-stationary drift, naive eviction heuristics (FIFO, LFU) disproportionately purge dormant classes, causing severe class imbalance. This metric evaluates the agent's ability to maintain representation parity across all classes.

#### Mathematical Formulation
Following the distribution matching formulation of Isele & Cosgun (2018), the metric is computed as the Kullback-Leibler (KL) divergence between the memory class distribution $P_{\text{mem}}$ and the uniform reference prior $P_{\text{target}}$:

$$D_{\text{KL}}(P_{\text{mem}} \parallel P_{\text{target}}) = \sum_{c=0}^{K-1} P_{\text{mem}}(c) \ln \left(\frac{P_{\text{mem}}(c) + \epsilon}{P_{\text{target}}(c)}\right) \quad [\text{nats}]$$

Where:
- $K = 10$ is the number of target classes.
- $P_{\text{mem}}(c) = \frac{1}{C} \sum_{i=0}^{C-1} \mathbb{I}(\text{actions}[i] = c)$ is the empirical proportion of class $c$ in the memory buffer of capacity $C = 5,000$.
- $P_{\text{target}}(c) = \frac{1}{K} = 0.1$ is the balanced ground-truth prior.
- $\epsilon = 10^{-12}$ is a numerical stabilizer preventing $\ln(0)$ singularities.
- For B0 (which has no memory buffer), $D_{\text{KL}} \equiv 0.0000\text{ nats}$ by convention.

A divergence near zero indicates optimal class preservation, ensuring that rare or temporarily dormant classes are not permanently evicted from memory.

#### Literature Source
- **Isele & Cosgun (2018):** *Selective Experience Replay for Lifelong Learning* (AAAI 2018).

#### Implementation in Codebase
- **File:** [`evaluate_hybrid_global.py`](../../evaluate_hybrid_global.py#L328-L335)
- **Code:**
  ```python
  if b_id == "B0" or mem is None or mem.size == 0:
      kl_div = 0.0
  else:
      counts = np.bincount(mem._actions[:mem.size], minlength=10)
      p_mem = counts / np.sum(counts)
      p_target = np.full(10, 0.1, dtype=np.float32)
      kl_div = float(np.sum(p_mem * np.log((p_mem + 1e-12) / p_target)))
  ```

---

## 4. Verification and Reproducibility

All seven metrics are computed systematically during the execution of [evaluate_hybrid_global.py](../../evaluate_hybrid_global.py) and exported synchronously to both [`outputs/eaai_metrics.json`](../../outputs/eaai_metrics.json) and [`outputs/eaai_metrics.csv`](../../outputs/eaai_metrics.csv).

To run the automated verification test verifying metric mathematical invariants and schema compliance:

```bash
# Execute Phase 4 unit and integration test suite
python -m unittest tests/test_phase4.py -v
```

---

**Navigation:**
- Previous: [Baseline Comparison Analysis](baseline-comparison.md)
- Up: [Documentation Index](../README.md)
- Next: [Documentation Index](../README.md)

---
Licensed under the GNU General Public License v3.0. See [LICENSE](../../LICENSE) for details.
