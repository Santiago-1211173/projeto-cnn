# Active Episodic Memory Management via Reinforcement Learning for Robust CNN Inference on Out-of-Distribution Data

**Robust Out-of-Distribution Routing for Resource-Constrained Semiparametric Vision Systems**

![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square)
![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square)
![TensorFlow 2.10.1](https://img.shields.io/badge/TensorFlow-2.10.1-FF6F00?style=flat-square)
![License GPL-3.0](https://img.shields.io/badge/License-GPL--3.0-blue?style=flat-square)
![NumPy](https://img.shields.io/badge/NumPy-1.24+-013243?style=flat-square)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?style=flat-square)

> Part of the [Active Episodic Memory Management via Reinforcement Learning for Robust CNN Inference on Out-of-Distribution Data](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/README.md) documentation.

---

## Abstract

This repository implements a semiparametric vision architecture combining deep Convolutional Neural Networks (CNNs) with a capacity-bounded $k$-Nearest Neighbors ($k$-NN) episodic memory, actively curated by a Reinforcement Learning (RL) agent under non-stationary Edge AI conditions. The system is empirically validated across two distinct complexity regimes:
1. **Low-Dimensional Stylized Regime (MNIST):** 28x28x1 grayscale handwritten digits, using a custom from-scratch CNN ([`RawModel`](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/src/models/custom_cnn.py)) with 225,034 parameters and Mahalanobis++ unit hypersphere covariance estimation.
2. **High-Dimensional Natural Regime (CIFAR-10):** 32x32x3 natural RGB images, using an upgraded ResNet-9 backbone ([`RawModelCIFAR10`](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/src/cifar10/model.py)) with residual connections and Batch Normalization (6.57M parameters), coupled with a Dual Uncertainty Arbiter ([`DualUncertaintyArbiter`](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/src/cifar10/ood_arbiter.py)) fusing unnormalized Mahalanobis distance with predictive Shannon entropy.

Crucially, both vision backbones preserve an **invariant 128-dimensional latent bottleneck** ($z \in \mathbb{R}^{128}$), allowing the downstream episodic memory buffer ([`KNNBanditAgent128D`](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/src/models/knn_bandit_agent.py)), Double DQN policy agent ([`RLAgent`](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/src/models/rl_agent.py)), and Curriculum Learning reward manager ([`RewardManager`](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/src/models/reward_manager.py)) to operate over an identical architectural contract. Under a prequential (test-then-train) streaming evaluation protocol under severe Gaussian noise injection ($\sigma \in \{0.0, 0.2, 0.4, 0.6, 0.8\}$), the RL-governed memory system (**B4**) systematically prevents cache pollution via Action 0 (Ignore/Filter) and eliminates class starvation via Action 3 (Redundancy pruning), operating within strictly bounded $O(1)$ memory constraints (< 10 MB RAM) and real-time inference latency (< 7 ms).

---

## Dataset Support Matrix

| Characteristic | MNIST Benchmark | CIFAR-10 Benchmark |
|:---|:---:|:---:|
| **Input Dimensions** | $28 \times 28 \times 1$ | $32 \times 32 \times 3$ |
| **Color Space** | Grayscale (1 channel) | RGB (3 channels) |
| **Classes** | 10 (Digits 0-9) | 10 (Natural Object Categories) |
| **Data Loader** | [`src/data/loader.py`](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/src/data/loader.py) (Raw binary ubyte) | [`scripts/cifar10/train_cnn.py`](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/scripts/cifar10/train_cnn.py) (`tf.keras.datasets.cifar10`) |
| **Preprocessing** | Rescale to $[0, 1]$ | Per-channel Standardization + Random Flip / Crop |
| **Feature Extractor** | [`RawModel`](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/src/models/custom_cnn.py) (From-scratch CNN) | [`RawModelCIFAR10`](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/src/cifar10/model.py) (ResNet-9 Backbone) |
| **Trainable Parameters** | 225,034 | 6,568,394 |
| **Nominal Clean Accuracy** | 98.74% | 91.18% |
| **Invariant Latent Space** | $128\text{D}$ Dense Bottleneck | $128\text{D}$ Dense Bottleneck |
| **OOD Arbiter Strategy** | Mahalanobis++ ($L_2$ Unit Hypersphere) | Dual Uncertainty (Unnormalized Mahalanobis + Entropy) |
| **Calibrated Thresholds** | $\tau = 12.5$ | $\tau_M = 16.0380$, $\tau_H = 0.7382$ |
| **Episodic Memory $k$** | $k = 30$ | $k = 10$ |
| **Evaluation Script** | [`evaluate_hybrid_global.py`](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/evaluate_hybrid_global.py) | [`scripts/cifar10/evaluate_baselines.py`](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/scripts/cifar10/evaluate_baselines.py) |

---

## Key Experimental Results

The architecture was evaluated against four baselines under five progressive noise regimes ($\sigma \in \{0.0, 0.2, 0.4, 0.6, 0.8\}$) across 5,000 streaming test samples with a fixed 5,000-vector memory capacity.

### 1. MNIST Benchmark (Low-Dimensional Regime)

| Baseline | Acc 0.0 | Acc 0.2 | Acc 0.4 | Acc 0.6 | Acc 0.8 | Mean Acc | Latency (ms) | Peak RAM (MB) | Cache Hit (%) | Eviction $D_{\text{KL}}$ |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| B0: Pure CNN | 98.0% | 90.8% | 51.4% | 30.0% | 20.3% | 58.10% | 0.004 | 0.001 | 0.0% | 0.000 |
| B1: Infinite Memory | 96.0% | 91.6% | 51.4% | 29.4% | 17.6% | 57.20% | 4.693 | 35.197 | 57.0% | 0.143 |
| B2: FIFO Eviction | 96.0% | 91.5% | 51.5% | 29.3% | 17.6% | 57.18% | 3.272 | 7.474 | 57.0% | 0.500 |
| B3: LFU Eviction | 96.0% | 91.5% | 51.5% | 29.3% | 17.6% | 57.18% | 3.302 | 7.474 | 57.0% | 0.500 |
| **B4: RL Active Memory** | **95.7%** | **90.7%** | **55.7%** | **38.9%** | **27.3%** | **61.66%** | 6.671 | 9.924 | **61.5%** | **0.003** |

### 2. CIFAR-10 Benchmark (High-Dimensional Natural Regime)

| Baseline | Acc 0.0 | Acc 0.2 | Acc 0.4 | Acc 0.6 | Acc 0.8 | Mean Acc | Latency (ms) | Peak RAM (MB) | Cache Hit (%) | Eviction $D_{\text{KL}}$ |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| B0: Pure CNN | 91.4% | 12.1% | 11.5% | 10.0% | 10.7% | 27.14% | 2.358 | 0.001 | 0.0% | 0.000 |
| B1: Infinite Memory | 92.1% | 11.1% | 11.7% | 9.6% | 11.2% | 27.14% | 7.054 | 52.283 | 13.16% | 0.279 |
| B2: FIFO Eviction | 92.1% | 11.1% | 11.7% | 9.6% | 11.2% | 27.14% | 5.248 | 7.481 | 13.16% | 0.885 |
| B3: LFU Eviction | 92.1% | 11.1% | 11.7% | 9.6% | 11.5% | 27.20% | 5.436 | 7.482 | 13.23% | 0.614 |
| **B4: RL Active Memory** | **92.2%** | **12.0%** | **11.3%** | **10.1%** | **11.9%** | **27.50%** | 6.940 | 9.933 | **13.60%** | **0.0007** |

### Key Scientific Findings:
- **Robustness Under Severe Noise:** On MNIST, B4 outperforms the best alternative baseline by +8.9% at $\sigma = 0.6$ and +7.0% at $\sigma = 0.8$. On CIFAR-10, B4 achieves superior overall accuracy (27.50%) over both B0 and B2 (27.14%).
- **Clean In-Distribution Generalization:** On CIFAR-10 clean data ($\sigma = 0.0$), episodic memory augmentation provides a **+0.80% accuracy dividend** (92.20% vs. 91.40% for standalone CNN).
- **Class Starvation Prevention:** B4 achieves an Eviction KL Divergence of **0.0028 nats** on MNIST and **0.0007 nats** on CIFAR-10—over **1,200x lower** than blind FIFO eviction (0.8850 nats), proving that Action 3 preserves balanced class representation under drift.
- **Bounded Hardware Footprint ($O(1)$ RAM):** Across both datasets, B4 maintains a strict memory ceiling (< 10 MB RAM), preventing the unbounded memory explosion of B1 (> 52 MB), while satisfying edge real-time latency budgets (< 7 ms per inference, > 140 fps).

For in-depth cross-dataset confrontation and statistical tests, see the [Cross-Dataset Scientific Synthesis](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/docs/results/cross-dataset-analysis.md).

---

## System Architecture

The semiparametric pipeline decouples static representation learning from dynamic online adaptation through four synchronized stages:

```mermaid
flowchart TD
    subgraph InputStage["1. Input Stage"]
        IN_M["MNIST Image<br>[28x28x1]"]
        IN_C["CIFAR-10 Image<br>[32x32x3]"]
    end

    subgraph Backbones["2. Parametric Vision Backbones"]
        CNN_M["Custom 4-Layer CNN<br>(src/models/custom_cnn.py)"]
        CNN_C["ResNet-9 Backbone<br>(src/cifar10/model.py)"]
        IN_M --> CNN_M
        IN_C --> CNN_C
    end

    subgraph InvariantBottleneck["3. Invariant 128D Latent Contract"]
        LATENT["128D Latent Vector z<br>+ 10-Class Softmax Posteriors p"]
        CNN_M --> LATENT
        CNN_C --> LATENT
    end

    subgraph OODStage["4. Out-of-Distribution Arbitration"]
        OOD_M{"Mahalanobis++<br>(d_M <= 12.5?)"}
        OOD_C{"Dual Uncertainty<br>(d_M <= 16.04 AND H <= 0.74?)"}
        LATENT -.-> OOD_M
        LATENT -.-> OOD_C
    end

    subgraph Execution["5. Decision & Memory Routing"]
        CNN_OUT["Nominal Prediction<br>y = argmax(p)"]
        MEM_ROUTE["Route to Episodic Memory<br>(knn_bandit_agent.py)"]
        OOD_M -->|In-Distribution| CNN_OUT
        OOD_M -->|OOD / Perturbed| MEM_ROUTE
        OOD_C -->|In-Distribution| CNN_OUT
        OOD_C -->|OOD / Uncertain| MEM_ROUTE
    end

    subgraph EpisodicMemory["6. Non-Parametric Episodic Memory & RL Governor"]
        KNN["Vectorized k-NN Search<br>(k=30 MNIST, k=10 CIFAR-10)"]
        VOTE["Distance-Weighted Voting<br>y_rescued = argmax(Expected Reward)"]
        RL["Double DQN + PER Policy<br>(src/models/rl_agent.py)"]
        ACT{"Eviction Action<br>0: Ignore | 1: FIFO<br>2: LFU | 3: Redundancy"}
        
        MEM_ROUTE --> KNN --> VOTE
        MEM_ROUTE --> RL --> ACT --> KNN
    end
```

For component specifications, see the [Architecture Documentation](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/docs/architecture/README.md).

---

## Repository Structure

```text
projeto-cnn/
├── LICENSE                                     # GNU General Public License v3.0
├── CITATION.cff                                # Machine-readable citation metadata
├── README.md                                   # Root system overview and entry point
├── requirements.txt                            # Python runtime dependencies
├── evaluate_hybrid_global.py                   # MNIST 5-baseline evaluation script
├── src/
│   ├── config.py                               # Global system configuration and paths
│   ├── data/
│   │   └── loader.py                           # MNIST binary parser and tf.data pipeline
│   ├── models/
│   │   ├── custom_cnn.py                       # [FROZEN] MNIST 4-layer CNN (128D bottleneck)
│   │   ├── knn_bandit_agent.py                 # [SHARED] Episodic memory buffer (128D contract)
│   │   ├── rl_agent.py                         # [SHARED] Double DQN + PER agent (5D state, 4 actions)
│   │   └── reward_manager.py                   # [SHARED] Curriculum Learning reward manager
│   ├── scratch/                                # [FROZEN] Custom SGD, layers, activations, losses
│   └── cifar10/                                # [DEDICATED] CIFAR-10 specialized module
│       ├── __init__.py
│       ├── layers.py                           # BatchNorm2D, Conv2D, ResidualBlock, DenseLayer
│       ├── optimizers.py                       # Adam/AdamW optimizer with decoupled weight decay
│       ├── model.py                            # RawModelCIFAR10 (ResNet-9, 128D latent, 91.18% acc)
│       └── ood_arbiter.py                      # Dual Uncertainty Arbiter (Mahalanobis + Entropy)
├── scripts/
│   ├── train_cnn.py                            # [FROZEN] MNIST CNN training script
│   ├── profile_latent.py                       # [FROZEN] MNIST Mahalanobis profiling
│   ├── simulate_online.py                      # [FROZEN] MNIST online RL simulation
│   └── cifar10/                                # [DEDICATED] CIFAR-10 specialized pipeline
│       ├── train_cnn.py                        # Train ResNet-9 backbone to >= 90% accuracy
│       ├── profile_latent.py                   # Calibrate Dual Uncertainty Arbiter profiles
│       ├── seed_memory.py                      # Seed 5,000 clean exemplars (k=10)
│       ├── train_simulation.py                 # 50,000-step prequential RL streaming simulation
│       └── evaluate_baselines.py               # EAAI 5-baseline evaluation runner (B0 to B4)
├── outputs/
│   ├── mnist/                                  # [FROZEN] Preserved MNIST metrics, checkpoints, dashboard
│   └── cifar10/                                # Dedicated CIFAR-10 checkpoints, metrics, dashboard
└── docs/                                       # Complete technical documentation hierarchy
    ├── README.md                               # Documentation index and navigation map
    ├── architecture/                           # Deep-dive architecture specifications
    ├── getting-started/                        # Installation, quickstart, and configuration
    ├── guides/                                 # Training, evaluation, and visualization guides
    ├── api/                                    # Full API reference across all modules
    ├── results/                                # Benchmark reports and cross-dataset synthesis
    └── Literatura/                             # Curated scientific literature (12 pillars)
```

---

## Quick Start & Reproduction

### Prerequisites
- Python >= 3.10 and < 3.12
- PyTorch >= 2.0, TensorFlow == 2.10.1, NumPy >= 1.24, scikit-learn >= 1.3

```bash
git clone https://github.com/[your-organization]/projeto-cnn.git
cd projeto-cnn
pip install -r requirements.txt
```

### Reproducing MNIST Benchmark
To evaluate the pre-trained MNIST system across all 5 baselines:
```bash
python evaluate_hybrid_global.py --dataset mnist --output-dir outputs/mnist/
```

To execute the full MNIST training and profiling pipeline:
```bash
python scripts/train_cnn.py
python scripts/profile_latent.py
python scripts/simulate_online.py
python evaluate_hybrid_global.py
```

### Reproducing CIFAR-10 Benchmark
To evaluate the pre-trained CIFAR-10 system across all 5 baselines:
```bash
python scripts/cifar10/evaluate_baselines.py --samples-per-level 1000 --noise-levels 0.0 0.2 0.4 0.6 0.8
```

To execute the complete CIFAR-10 training and profiling pipeline:
```bash
# 1. Train ResNet-9 backbone to 90%+ accuracy
python scripts/cifar10/train_cnn.py --epochs 25 --batch-size 128 --lr 0.001

# 2. Profile latent space and calibrate Dual Uncertainty Arbiter
python scripts/cifar10/profile_latent.py --percentile 95.0

# 3. Seed episodic memory buffer with 5,000 prototypes
python scripts/cifar10/seed_memory.py --samples 5000 --k 10

# 4. Train Double DQN active memory agent via streaming simulation
python scripts/cifar10/train_simulation.py --steps 50000 --noise-rate 0.1 --noise-level 0.6

# 5. Run standardized 5-baseline evaluation
python scripts/cifar10/evaluate_baselines.py
```

---

## Documentation Hierarchy

| Section | Path | Description |
|:---|:---|:---|
| [Documentation Index](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/docs/README.md) | [`docs/README.md`](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/docs/README.md) | Master navigation hub for the entire documentation hierarchy |
| [System Architecture](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/docs/architecture/README.md) | [`docs/architecture/`](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/docs/architecture/README.md) | CNN feature extractors, OOD detection regimes, memory buffer, RL agent |
| [Getting Started](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/docs/getting-started/README.md) | [`docs/getting-started/`](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/docs/getting-started/README.md) | Prerequisites, environment verification, quickstart, configuration |
| [Usage Guides](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/docs/guides/README.md) | [`docs/guides/`](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/docs/guides/README.md) | End-to-end training pipelines, evaluation runners, XAI visualization |
| [API Reference](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/docs/api/README.md) | [`docs/api/`](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/docs/api/README.md) | Comprehensive class references for core modules and CIFAR-10 additions |
| [Experimental Results](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/docs/results/README.md) | [`docs/results/`](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/docs/results/README.md) | MNIST and CIFAR-10 baseline benchmarks, and cross-dataset synthesis |
| [Scientific Literature](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/docs/Literatura/README.md) | [`docs/Literatura/`](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/docs/Literatura/README.md) | Curated literature repositories across 12 foundational AI pillars |

---

## Citation

If you utilize this codebase or semiparametric architecture in your research, please cite:

```bibtex
@article{active_episodic_memory_rl_2026,
  title     = {Active Episodic Memory Management via Reinforcement Learning
               for Robust CNN Inference on Out-of-Distribution Data},
  author    = {[Author Name]},
  journal   = {Engineering Applications of Artificial Intelligence},
  year      = {2026},
  publisher = {Elsevier},
  note      = {Under Review}
}
```

Machine-readable citation metadata is available in [`CITATION.cff`](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/CITATION.cff).

---

## License

This project is licensed under the **GNU General Public License v3.0**. See the [`LICENSE`](file:///c:/Users/sanfr/Desktop/projetos-gecad/projeto-cnn/LICENSE) file for the full license text.