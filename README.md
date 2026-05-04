# Hybrid Vision System: Custom CNN & RL Arbitrator

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.10.1](https://img.shields.io/badge/TensorFlow-2.10.1-orange.svg)](https://tensorflow.org/)
[![CUDA 12.6](https://img.shields.io/badge/CUDA-12.6-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Hardware: 2x NVIDIA L40S](https://img.shields.io/badge/Hardware-2x_NVIDIA_L40S-76B900.svg)](https://www.nvidia.com/)


## Overview

This repository implements an enterprise-grade, hybrid computer vision pipeline for the MNIST dataset. Diverging from standard end-to-end Deep Learning approaches, this architecture introduces a Statistical Arbitrator and a Reinforcement Learning (RL) Agent to build a system highly resilient to Out-of-Distribution (OOD) and chaotic data.

The core philosophy is **Dynamic Routing**: clean data is processed by a custom-built Convolutional Neural Network (CNN), while noisy or corrupted inputs trigger an anomaly threshold and are routed to an RL Specialist Agent trained explicitly on ambiguous cases.

The system has been evaluated with **three distinct RL Specialist** implementations to study the trade-offs between parametric and non-parametric approaches:

1. **MLP Q-Network (Contextual Bandit)** — A gradient-trained neural network operating on 128D latent features.
2. **k-NN Bandit (Episodic Memory)** — A training-free, non-parametric agent operating on 10D softmax probabilities.
3. **k-NN Bandit Improved (128D)** — A non-parametric agent operating on the 128D latent space to maximize discriminative signal under noise.

## Core Innovations & Architecture

### 1. From-Scratch Deep Learning (The Visual Cortex)

The CNN bypasses high-level Keras abstractions. Core mathematical operations (Conv2D, Dense, MaxPool2D) are built from raw tf.Module and tf.Variable objects.

* **Initialization:** He Normal for Convolutions (mathematically optimal for ReLU) and Glorot Uniform for Dense layers.
* **Optimization:** Custom Stochastic Gradient Descent (SGD) updating GPU memory directly via atomic assign_sub operations.
* **Latent Space:** Compresses the spatial image into a rich 128D latent feature vector.

### 2. Mahalanobis Triage (The Arbitrator)

Softmax probabilities are often overconfident on noisy data. Instead, this system calculates the Mahalanobis Distance: $$D = \sqrt{(x - \mu)^T \Sigma^{-1} (x - \mu)}$$ of incoming 128D vectors against 10 learned multivariate Gaussian distributions representing the classes.

* If D < 15.0, the CNN's latent representation is trusted.
* If D >= 15.0, the input is classified as an anomaly/OOD and routed.

### 3. RL Fallback (The Specialist)

This system implements two interchangeable RL Specialist agents:

#### 3a. MLP Q-Network (Original)

A strictly off-policy Contextual Bandit agent (Q-Network) that operates on the **128D latent feature vector** from the CNN. Trained via +1/-1 rewards using an epsilon-greedy policy and Bellman-like squared error updates (backpropagation over 10 epochs).

* **Architecture:** Dense(128→64, ReLU) → Dense(64→10)
* **Parameters:** ~8,714 trainable weights
* **State Space:** 128D latent features

#### 3b. k-NN Bandit — Episodic Memory (New)

A non-parametric, training-free agent that replaces gradient-based learning with **episodic memory lookup**. It stores a database of `(state_10D, action, reward)` triplets and uses distance-weighted k-Nearest Neighbors voting to decide actions.

* **Architecture:** Ball Tree index (scikit-learn) over 10D probability vectors
* **Parameters:** 0 (non-parametric — the "model" is the memory bank itself)
* **State Space:** 10D softmax probabilities from the CNN's final layer
* **Scoring:** Distance-weighted `sum()` — closer neighbors and more numerous votes produce stronger signals

#### 3c. k-NN Bandit Improved — 128D (Newest)

An improved version of the k-NN agent that operates on the **128D latent features** instead of the 10D softmax probabilities. This solves the state-space bottleneck observed in the 10D agent, as the latent space preserves much more discriminative signal under heavy noise.

* **Architecture:** Auto/KD-Tree index (scikit-learn) over 128D latent vectors
* **Parameters:** 0
* **State Space:** 128D latent features from the CNN

## Comparative Analysis: Three-Way Benchmark

Both agents were benchmarked side-by-side on the same 10,000 MNIST test images using a fixed random seed (42) for reproducibility. The Mahalanobis routing and CNN are identical in both cases — only the RL Specialist differs.

### Benchmark Results

| Métrica | MLP (Q-Network) | k-NN 10D | k-NN Improved (128D) | Gap (128D vs MLP) |
|---|---|---|---|---|
| **Híbrido (Limpas)** | 97.1% | 97.5% | **97.8%** | +0.6% |
| **Híbrido (Ruído σ=0.6)** | 86.6% | 75.7% | **88.5%** | +1.9% |
| Agente Isolado (Limpas) | 91.4% | 91.5% | **97.6%** | +6.1% |
| Agente Isolado (Ruído) | 86.6% | 75.7% | **88.5%** | +1.9% |
| Inferência (ms) | **3.3** | 10088.5 | 3875.3 | - |
| Treino | 10 épocas + backprop | 0 (memória) | 0 (memória) | - |
| Parâmetros | ~8,714 (128→64→10) | 0 (non-parametric) | 0 (non-parametric) | - |

### Key Findings

1. **Clean Images:** The k-NN Bandit slightly outperforms the MLP (+0.4%) because it achieves 80.4% on the 479 ambiguous images routed by the Arbitrator, vs 72.4% for the MLP.

2. **Noisy Images:** The MLP Q-Network significantly outperforms the k-NN Bandit (+11.0%). This is attributed to:
   * The MLP operates on the **128D latent space**, which preserves richer discriminative signal than the 10D softmax probabilities.
   * The MLP learns a **decision boundary** via gradient descent, generalizing across noise patterns. The k-NN relies on exact neighbor matching, which is less robust when noise creates novel 10D patterns unseen in the memory bank.
   
3. **The 128D Breakthrough:** By moving the k-NN agent from the 10D softmax probabilities to the **128D latent features** and populating the episodic memory with both clean and noisy images (Distribution Shift correction), we closed the 11% accuracy gap and completely surpassed the MLP. The new **k-NN Improved (128D)** agent achieves **88.5%** on noisy images and **97.8%** on clean images, beating the original MLP targets (86.6% and 97.1%) by **+1.9% and +0.6% respectively**, while strictly remaining a non-parametric model with zero backpropagation.
4. **Trade-off:** The 128D k-NN agent requires zero gradient-based training (the memory is populated in a single forward pass) and beats the MLP on accuracy across all scenarios, making it highly attractive. However, inference is slower than the MLP (3.8s vs 3ms for 10k images).

### Per-Class Analysis (Noisy Images, RL Subset)

| Digit | N | MLP Acc | k-NN Acc | Δ |
|---|---|---|---|---|
| 0 | 980 | 96.6% | 95.2% | -1.4% |
| 1 | 1135 | 96.7% | 81.1% | -15.7% |
| 2 | 1030 | 78.3% | 77.0% | -1.4% |
| 3 | 1010 | 88.4% | 84.4% | -4.1% |
| 4 | 982 | 83.6% | 80.1% | -3.5% |
| 5 | 892 | 81.8% | 63.0% | -18.8% |
| 6 | 958 | 90.5% | 87.2% | -3.3% |
| 7 | 1028 | 89.4% | 61.2% | -28.2% |
| 8 | 974 | 81.0% | 59.4% | -21.6% |
| 9 | 1009 | 77.9% | 66.1% | -11.8% |

The k-NN struggles most with digits 7, 8, and 5 under noise — these digits produce similar confusion patterns in the 10D softmax space, causing the k-NN to misroute them to morphologically similar digits.

## Tech Stack & Hardware Optimization

* **Core Framework:** Python 3.10+ and TensorFlow 2.10.1
* **Data & Math:** NumPy 1.26.4, Pandas 2.3.3
* **Machine Learning & XAI:** scikit-learn 1.6.1, Matplotlib 3.9.4, Seaborn 0.13.2
* **Hardware Acceleration:** CUDA Toolkit 12.6 (NVCC V12.6.85) with NVIDIA Driver 591.59.
* **Throughput Optimization:**
  * Explicit tf.data pipelines with .prefetch(tf.data.AUTOTUNE).
  * Multi-GPU categorical loss scaling logic.
  * JIT/PTX compilation forcing and async GPU execution blocking for precise benchmarking.
  * Target Hardware: Tested and optimized for dual NVIDIA L40S execution paths.

## Repository Structure

```text
├── src/
│   ├── models/
│   │   ├── custom_cnn.py          # From-scratch CNN (Conv2D → Latent 128D → Softmax 10D)
│   │   ├── rl_agent.py            # MLP Q-Network Agent (128D → 64 → 10)
│   │   └── knn_bandit_agent.py    # k-NN Bandit Agent (10D episodic memory)
│   ├── data/                      # Binary byte parsers & tf.data.Dataset loaders
│   └── scratch/                   # Raw DL Math: layers, losses, optimizers, activations
├── outputs/                       # Checkpoints, Latent Profiles, and XAI Visualizations
├── train.py                       # CNN Training Engine (Multi-GPU/Single-GPU optimized)
├── train_rl.py                    # k-NN Bandit Memory Population Engine
├── profile_clusters.py            # Latent Space Profiler (Computes Covariance Matrices)
├── evaluate_hybrid_system.py      # End-to-end Inference Pipeline (6-image demo)
├── benchmark_hibrido.py           # k-NN Bandit Benchmark (10k images, clean + noisy)
├── benchmark_comparativo.py       # Side-by-Side MLP vs k-NN Comparison
└── visualize_*.py                 # Explainable AI (XAI) modules
```

## Execution Pipeline

Follow this strict lifecycle to replicate the hybrid model training and evaluation. Ensure your raw binary dataset is placed in data/MNIST/raw/.

### 1. Train the Visual Extractor

Trains the custom from-scratch CNN to establish the 128D latent space.

```bash
python train.py
```

### 2. Profile the Latent Space

Calculates the mu (centroid) and Sigma^-1 (inverse covariance matrix) for the 10 digit distributions.

```bash
python profile_clusters.py
```

### 3. Train the Specialist RL Agent

**Option A — k-NN Bandit (Non-Parametric, Current Default):**
Populates the episodic memory by passing noisy images through the CNN and storing oracle labels.

```bash
python train_rl.py
```

**Option B — MLP Q-Network (Gradient-Based, Original):**
The original MLP training uses backpropagation over 10 epochs. The pre-trained weights are stored at `outputs/rl_agent_weights-1.*`. To retrain from the original script, restore it from git history (`git show 3eecf42:train_rl.py`).

### 4. Evaluate the Full System

Run the end-to-end pipeline on a batch of test data:

```bash
# k-NN Benchmark (10k images)
python benchmark_hibrido.py

# Side-by-side MLP vs k-NN comparison
python benchmark_comparativo.py

# 6-image visual demo
python evaluate_hybrid_system.py
```

## Explainable AI (XAI) & Profiling

Understanding the decision boundaries is a core tenet of this system:

* **Latent Space Collapse (t-SNE):** visualize_hybrid_tsne.py projects the 128D features into 2D, mathematically visualizing how OOD noise collapses standard class clusters.
* **Saliency Maps:** visualize_saliency.py tracks gradients backward from the softmax predictions to the input pixels (dP_c / dX), highlighting the exact morphological features the CNN focuses on.
* **k-NN Decision Profiles:** visualize_rl_decisions.py generates confidence profile comparisons (CNN probabilities vs k-NN expected rewards) and correction flow heatmaps showing how the k-NN re-routes CNN predictions.
* **Hardware Profiling:** benchmark_l40s.py measures raw floating-point matrix multiplication throughput, isolating CPU vs. GPU overhead.