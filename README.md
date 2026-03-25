# Hybrid Vision System: Custom CNN & RL Arbitrator

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.10.1](https://img.shields.io/badge/TensorFlow-2.10.1-orange.svg)](https://tensorflow.org/)
[![CUDA 12.6](https://img.shields.io/badge/CUDA-12.6-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Hardware: 2x NVIDIA L40S](https://img.shields.io/badge/Hardware-2x_NVIDIA_L40S-76B900.svg)](https://www.nvidia.com/)


## Overview

This repository implements an enterprise-grade, hybrid computer vision pipeline for the MNIST dataset. Diverging from standard end-to-end Deep Learning approaches, this architecture introduces a Statistical Arbitrator and a Contextual Bandit Reinforcement Learning (RL) Agent to build a system highly resilient to Out-of-Distribution (OOD) and chaotic data.

The core philosophy is Dynamic Routing: clean data is processed by a custom-built Convolutional Neural Network (CNN), while noisy or corrupted inputs trigger an anomaly threshold and are routed to an RL Specialist Agent trained explicitly on ambiguous cases.

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

A strictly off-policy Contextual Bandit agent (Q-Network) steps in for ambiguous inputs. Trained exclusively via +1/-1 rewards using an epsilon-greedy policy and Bellman-like squared error updates, it learns complex mappings that standard categorical cross-entropy fails to capture.

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
│   ├── models/           # Architectures: custom_cnn.py, rl_agent.py
│   ├── data/             # Binary byte parsers & tf.data.Dataset loaders
│   └── scratch/          # Raw DL Math: layers, losses, optimizers, activations
├── outputs/              # Checkpoints, Latent Profiles, and XAI Visualizations
├── train.py              # CNN Training Engine (Multi-GPU/Single-GPU optimized)
├── train_rl.py           # RL Contextual Bandit Training Engine
├── profile_clusters.py   # Latent Space Profiler (Computes Covariance Matrices)
├── evaluate_hybrid_system.py # End-to-end Inference Pipeline
└── visualize_*.py        # Explainable AI (XAI) modules
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

Trains the Contextual Bandit on corrupted images using an epsilon-greedy exploration policy.

```bash
python train_rl.py
```

### 4. Evaluate the Full System

Run the end-to-end pipeline (Extractor -> Arbitrator -> Decision) on a batch of test data.

```bash
python benchmark_hibrido.py
```

## Explainable AI (XAI) & Profiling

Understanding the decision boundaries is a core tenet of this system:

* **Latent Space Collapse (t-SNE):** visualize_hybrid_tsne.py projects the 128D features into 2D, mathematically visualizing how OOD noise collapses standard class clusters.
* **Saliency Maps:** visualize_saliency.py tracks gradients backward from the softmax predictions to the input pixels (dP_c / dX), highlighting the exact morphological features the CNN focuses on.
* **Hardware Profiling:** benchmark_l40s.py measures raw floating-point matrix multiplication throughput, isolating CPU vs. GPU overhead.