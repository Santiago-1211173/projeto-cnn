# src/config.py — Centralized project configuration
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Random seed for reproducibility
RANDOM_SEED = 42

# Mahalanobis threshold for CNN ↔ RL routing
MAHALANOBIS_THRESHOLD = 12.5

# Noise sweep levels for oracle seeding
NOISE_SWEEP = [0.0, 0.2, 0.4, 0.6, 0.8]

# Threshold sweep range for global evaluation
THRESHOLD_SWEEP_START = 5.0
THRESHOLD_SWEEP_END = 30.0
THRESHOLD_SWEEP_STEP = 2.5

# Data paths
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "MNIST", "raw")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")

# Agent defaults
KNN_K = 30
KNN_N_ACTIONS = 10

# CNN training defaults
CNN_BATCH_SIZE = 256
CNN_EPOCHS = 10
CNN_LEARNING_RATE = 0.01

# Memory bank paths
MEMORY_BANK_10D_PATH = os.path.join(OUTPUT_DIR, "knn_memory_bank.npz")
MEMORY_BANK_128D_PATH = os.path.join(OUTPUT_DIR, "knn_memory_bank_128d.npz")
MAHALANOBIS_PROFILES_PATH = os.path.join(OUTPUT_DIR, "mahalanobis_profiles.npz")
MEMORY_MAPPING_PATH = os.path.join(OUTPUT_DIR, "knn_memory_mapping.json")
