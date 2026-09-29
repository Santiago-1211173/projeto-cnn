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

# --- Fase 1: Memoria Episodica ---
MEMORY_CAPACITY = 5000
LATENT_DIM = 128
KNN_K_NEIGHBORS = 30

# --- Fase 2: Agente RL (pre-declarados para evitar imports circulares) ---
RL_STATE_DIM = 5
RL_N_ACTIONS = 4
CURRICULUM_ALPHA_DECAY = 0.995
SLIDING_VALIDATION_BUFFER_SIZE = 100
REPLAY_BUFFER_CAPACITY = 10000

# CNN training defaults
CNN_BATCH_SIZE = 256
CNN_EPOCHS = 10
CNN_LEARNING_RATE = 0.01

# --- Fase 3: Simulacao Online e Caos ---
SIMULATION_STEPS = 50000
SIMULATION_NOISE_RATE = 0.1
SIMULATION_NOISE_LEVEL = 0.6
SIMULATION_LOG_INTERVAL = 500
SIMULATION_CSV_PATH = os.path.join(OUTPUT_DIR, "train_rl_simulation_log.csv")
RL_AGENT_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "rl_agent_phase3.pt")
MAHALANOBIS_PP_PROFILES_PATH = os.path.join(OUTPUT_DIR, "mahalanobis_pp_profiles.npz")

# Memory bank paths
MEMORY_BANK_10D_PATH = os.path.join(OUTPUT_DIR, "knn_memory_bank.npz")
MEMORY_BANK_128D_PATH = os.path.join(OUTPUT_DIR, "knn_memory_bank_128d.npz")
MAHALANOBIS_PROFILES_PATH = os.path.join(OUTPUT_DIR, "mahalanobis_profiles.npz")
MEMORY_MAPPING_PATH = os.path.join(OUTPUT_DIR, "knn_memory_mapping.json")
