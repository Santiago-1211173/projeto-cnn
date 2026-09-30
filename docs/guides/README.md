# Usage Guides

> Part of the [Active Episodic Memory Management via Reinforcement Learning for Robust CNN Inference on Out-of-Distribution Data](../../README.md) documentation.
> Parent: [Documentation Index](../README.md) | Up: [Root README](../../README.md)

---

## 1. Overview

This directory provides operational, step-by-step guides for executing the end-to-end workflows of the RL-driven active episodic memory system. The guides cover the full lifecycle of the semiparametric vision pipeline, from model training and latent space calibration to multi-baseline benchmarking and explainable AI (XAI) diagnostics.

The software architecture couples a custom Convolutional Neural Network (CNN) feature extractor with a capacity-bounded $k$-NN episodic memory buffer and a Double Deep Q-Network (Double DQN) with Prioritized Experience Replay (PER). Operating these components together requires adherence to specific sequential workflows to guarantee reproducibility, prevent data leakage, and ensure numerical stability under non-stationary Edge AI operational conditions.

---

## 2. Available Guides

The guides are organized into three primary operational domains:

| Document | Title | Purpose | Key Artifacts Produced |
|:---------|:------|:--------|:-----------------------|
| [training-pipeline.md](training-pipeline.md) | [Training Pipeline](training-pipeline.md) | Complete sequential workflow for training the parametric CNN, computing Mahalanobis++ OOD profiles, seeding the episodic memory bank, and training the Double DQN active memory agent via streaming online simulation. | `modelo_dissecado-*`, `mahalanobis_pp_profiles.npz`, `knn_memory_bank_128d.npz`, `rl_agent_weights-*`, `train_rl_simulation_log.csv` |
| [evaluation.md](evaluation.md) | [Running the EAAI Evaluation](evaluation.md) | Execution protocol for the 5-baseline comparative benchmark (B0 to B4) across progressive Gaussian noise levels, tracking 7 engineering and scientific metrics under prequential test-then-train evaluation. | `eaai_metrics.csv`, `eaai_metrics.json`, `eaai_evaluation_dashboard.png` |
| [visualization.md](visualization.md) | [Explainable AI and Visualization Tools](visualization.md) | Suite of diagnostic visualization tools including t-SNE latent collapse mapping, gradient-based saliency heatmaps, decision confidence profiling, and 128D episodic memory rescue dashboards. | `colapso_latente_tsne.png`, `mapa_saliencia.png`, `perfil_confianca_cnn_vs_knn.png`, `fluxo_correcoes_cnn_knn.png`, `episodic_memory_rescue_full.png` |

---

## 3. Recommended Workflow Sequence

For researchers and engineers seeking to reproduce the experimental results presented in the EAAI publication, the recommended execution sequence is as follows:

```mermaid
flowchart TD
    subgraph S1["Phase A: Prerequisites & Verification"]
        A1["Install Dependencies<br>requirements.txt"]
        A2["Extract MNIST Binaries<br>data/MNIST/raw/"]
        A3["Run Environment Verification<br>docs/getting-started/quickstart.md"]
        A1 --> A2 --> A3
    end

    subgraph S2["Phase B: Training Pipeline"]
        B1["Step 1: Train CNN<br>scripts/train_cnn.py"]
        B2["Step 2: Profile Latent Space<br>scripts/profile_latent.py"]
        B3["Step 3: Seed Episodic Memory<br>scripts/train_rl.py"]
        B4["Step 4: Online RL Simulation<br>training/train_rl_online_simulation.py"]
        B1 --> B2 --> B3 --> B4
    end

    subgraph S3["Phase C: Benchmark Evaluation"]
        C1["Run 5-Baseline Benchmark<br>evaluate_hybrid_global.py"]
        C2["Export Scientific Metrics<br>outputs/eaai_metrics.json"]
        C1 --> C2
    end

    subgraph S4["Phase D: Explainable AI & Analysis"]
        D1["t-SNE Latent Collapse<br>visualizations/make_tsne.py"]
        D2["Gradient Saliency Maps<br>visualizations/make_saliency.py"]
        D3["Episodic Memory Rescue<br>visualizations/make_memory_rescue.py"]
        D1 --> D2 --> D3
    end

    S1 --> S2 --> S3 --> S4
```

1. **Prerequisites & Setup:** Complete the onboarding steps in [Installation and Quick Start](../getting-started/quickstart.md) to ensure hardware acceleration, directory structures, and dataset files are properly configured.
2. **Execute Training Pipeline:** Follow [training-pipeline.md](training-pipeline.md) sequentially from Step 1 through Step 4. Do not omit the latent profiling step, as downstream routing depends on regularized covariance matrices.
3. **Execute Comparative Evaluation:** Follow [evaluation.md](evaluation.md) to evaluate the trained active memory agent against blind eviction baselines (FIFO, LFU, pure CNN, and infinite memory).
4. **Generate Visualizations:** Follow [visualization.md](visualization.md) to inspect model attention, latent cluster preservation, and memory rescue mechanics for presentation and manuscript figures.

---

## 4. Hardware and Runtime Considerations

The computational requirements vary across the different pipeline stages:

| Stage | Script | Typical Hardware | Memory Requirement | Estimated Duration |
|:------|:-------|:-----------------|:-------------------|:-------------------|
| CNN Training | `scripts/train_cnn.py` | CUDA GPU / Apple Silicon / CPU | ~1.5 GB VRAM / ~2 GB RAM | ~2-3 min (GPU), ~12 min (CPU) |
| Latent Profiling | `scripts/profile_latent.py` | CPU or CUDA GPU | ~1.0 GB RAM | ~20-30 seconds |
| Memory Seeding | `scripts/train_rl.py` | CPU or CUDA GPU | ~2.0 GB RAM | ~1-2 minutes |
| Online RL Simulation | `training/train_rl_online_simulation.py` | CUDA GPU / Multi-core CPU | ~2.5 GB RAM | ~15-25 min (50,000 steps) |
| Global Evaluation | `evaluate_hybrid_global.py` | CPU or CUDA GPU | ~1.5 GB RAM | ~2-4 minutes |
| t-SNE Visualization | `visualizations/make_tsne.py` | Multi-core CPU | ~2.0 GB RAM | ~1-2 minutes |
| Memory Rescue Dashboard | `visualizations/make_memory_rescue.py` | CPU or CUDA GPU | ~1.5 GB RAM | ~30-45 seconds |

---

**Navigation:**
- Previous: [Configuration Reference](../getting-started/configuration.md)
- Up: [Documentation Index](../README.md)
- Next: [Training Pipeline](training-pipeline.md)

---
Licensed under the GNU General Public License v3.0. See [LICENSE](../../LICENSE) for details.
