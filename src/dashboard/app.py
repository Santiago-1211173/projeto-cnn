import sys
import os
import io
import time
import base64
import logging
import threading
import json
import numpy as np
import tensorflow as tf
from PIL import Image
from flask import Flask, render_template, request, jsonify
from sklearn.model_selection import train_test_split

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(PROJECT_ROOT)

from src.config import (
    DATA_DIR,
    CHECKPOINT_DIR,
    MAHALANOBIS_PROFILES_PATH,
    MEMORY_BANK_128D_PATH,
    MEMORY_MAPPING_PATH,
    RANDOM_SEED
)
from src.models.custom_cnn import RawModel
from src.models.knn_bandit_agent import KNNBanditAgent
from src.data.loader import load_mnist_raw

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global State
cnn = None
agent = None
mahalanobis_profiles = None

x_train = None
y_train = None
x_test = None
y_test = None

MEMORY_MAPPING = [] # Maps memory index to (x_train_idx, noise_level, is_positive)

TRAINING_STATUS = {
    "is_running": False,
    "progress": 0.0,
    "current_stage": "",
    "logs": [],
    "cnn_acc": 0.0,
    "knn_acc": 0.0,
    "memory_size": 0
}

def log_to_training(msg):
    logger.info(msg)
    if TRAINING_STATUS["is_running"]:
        TRAINING_STATUS["logs"].append(msg)
        # Keep only last 50 lines to avoid overflow
        if len(TRAINING_STATUS["logs"]) > 50:
            TRAINING_STATUS["logs"].pop(0)

def adicionar_ruido_batch(imagens: np.ndarray, intensidade: float = 0.6) -> np.ndarray:
    ruido = np.random.normal(loc=0.0, scale=intensidade, size=imagens.shape)
    return np.clip(imagens + ruido, 0., 1.)

def extrair_features_128d_cnn(imagens: np.ndarray, model, batch_size: int = 500):
    todos_estados = []
    todas_preds = []
    for i in range(0, len(imagens), batch_size):
        batch = imagens[i : i + batch_size]
        batch_tensor = tf.convert_to_tensor(batch, dtype=tf.float32)
        outputs = model(batch_tensor)
        todos_estados.append(outputs["latent_features"].numpy())
        todas_preds.append(np.argmax(outputs["probabilities"].numpy(), axis=1))
    return np.vstack(todos_estados), np.concatenate(todas_preds)

def reconstruct_memory_mapping(model, x_train_data, y_train_data):
    """
    Reconstructs the index mapping dynamically on startup to match the loaded agent memory.
    """
    global MEMORY_MAPPING
    MEMORY_MAPPING = []
    cenarios = [0.0, 0.2, 0.4, 0.6, 0.8]
    logger.info("Reconstructing Episodic Memory mapping (please wait a few seconds)...")
    
    for r, intensidade in enumerate(cenarios):
        logger.info(f"  Mapping scenario {r+1}/{len(cenarios)} (Noise {intensidade})...")
        x_ruido = adicionar_ruido_batch(x_train_data, intensidade) if intensidade > 0 else x_train_data
        _, preds_cnn = extrair_features_128d_cnn(x_ruido, model, batch_size=500)
        
        # Positive
        for idx in range(len(y_train_data)):
            MEMORY_MAPPING.append({
                'x_train_idx': int(idx),
                'noise_level': float(intensidade),
                'is_positive': True
            })
            
        # Negative
        erros = preds_cnn != y_train_data
        error_indices = np.where(erros)[0]
        for idx in error_indices:
            MEMORY_MAPPING.append({
                'x_train_idx': int(idx),
                'noise_level': float(intensidade),
                'is_positive': False
            })
            
    logger.info(f"Mapping completed! {len(MEMORY_MAPPING)} entries mapped.")
    
    # Save the reconstructed mapping so we don't have to guess next time
    try:
        os.makedirs(os.path.dirname(MEMORY_MAPPING_PATH), exist_ok=True)
        with open(MEMORY_MAPPING_PATH, "w") as f:
            json.dump(MEMORY_MAPPING, f)
        logger.info(f"Mapping saved to disk: {MEMORY_MAPPING_PATH}")
    except Exception as e:
        logger.error(f"Error saving mapping JSON: {e}")

def img_to_base64(img_array):
    # img_array is shape (28, 28, 1) in [0, 1]
    img_uint8 = (img_array.squeeze() * 255).astype(np.uint8)
    img = Image.fromarray(img_uint8, mode='L')
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

@app.before_request
def startup():
    global cnn, agent, mahalanobis_profiles, x_train, y_train, x_test, y_test
    if cnn is not None:
        return
        
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    logger.info("Loading CNN model...")
    cnn = RawModel()
    ckpt = tf.train.Checkpoint(model=cnn)
    latest_ckpt = tf.train.latest_checkpoint(CHECKPOINT_DIR)
    if latest_ckpt:
        ckpt.restore(latest_ckpt).expect_partial()
    
    try:
        if os.path.exists(MAHALANOBIS_PROFILES_PATH):
            mahalanobis_profiles = dict(np.load(MAHALANOBIS_PROFILES_PATH, allow_pickle=True))
            logger.info("Mahalanobis profiles loaded.")
        else:
            logger.warning("Mahalanobis profiles not found.")
            mahalanobis_profiles = {}
    except Exception as e:
        logger.error(f"Error loading mahalanobis_profiles: {e}")
        mahalanobis_profiles = {}

    logger.info("Loading k-NN Agent...")
    # Initialize unified KNNBanditAgent in 128D mode
    agent = KNNBanditAgent(k=30, n_actions=10, latent_dim=128, use_pca=False)
    if os.path.exists(MEMORY_BANK_128D_PATH):
        agent.load(MEMORY_BANK_128D_PATH)
    
    logger.info("Loading MNIST dataset and splitting partitions...")
    x_train_full, y_train_full = load_mnist_raw(DATA_DIR, kind='train')
    x_train_full = x_train_full.astype(np.float32) / 255.0

    x_train, x_test, y_train, y_test = train_test_split(
        x_train_full, y_train_full, 
        test_size=0.10, 
        random_state=RANDOM_SEED, 
        shuffle=True, 
        stratify=y_train_full
    )
    
    if os.path.exists(MEMORY_MAPPING_PATH):
        try:
            with open(MEMORY_MAPPING_PATH, "r") as f:
                global MEMORY_MAPPING
                MEMORY_MAPPING = json.load(f)
            logger.info(f"Memory mapping loaded from disk ({len(MEMORY_MAPPING)} entries).")
            # Basic validation
            if len(MEMORY_MAPPING) != agent.memory_size:
                logger.warning("Warning: Mapping JSON size does not match agent memory size. Reconstructing...")
                reconstruct_memory_mapping(cnn, x_train, y_train)
        except Exception as e:
            logger.error(f"Error loading mapping JSON: {e}")
            reconstruct_memory_mapping(cnn, x_train, y_train)
    else:
        reconstruct_memory_mapping(cnn, x_train, y_train)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/images', methods=['GET'])
def get_images():
    indices = np.random.choice(len(x_test), 50, replace=False)
    images_data = []
    for idx in indices:
        images_data.append({
            'index': int(idx),
            'label': int(y_test[idx]),
            'base64': img_to_base64(x_test[idx])
        })
    return jsonify(images_data)

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json
    idx = data.get('index', 0)
    noise_level = float(data.get('noise_level', 0.0))
    threshold = float(data.get('threshold', 15.0))
    k = int(data.get('k', 15))
    
    img = x_test[idx]
    if noise_level > 0:
        img = adicionar_ruido_batch(img[np.newaxis, ...], noise_level)[0]
        
    img_b64 = img_to_base64(img)
    true_label = int(y_test[idx])
    
    tensor = tf.convert_to_tensor(img[np.newaxis, ...], dtype=tf.float32)
    outputs = cnn(tensor)
    
    probs = outputs["probabilities"].numpy()[0]
    latent = outputs["latent_features"].numpy()[0]
    cnn_pred = int(np.argmax(probs))
    cnn_conf = float(np.max(probs))
    
    # Calculate Mahalanobis Distance
    m_dist = 0.0
    cnn_pred_str = str(cnn_pred)
    if cnn_pred_str in mahalanobis_profiles:
        profile_dict = mahalanobis_profiles[cnn_pred_str].item()
        mean = profile_dict["mu"]
        inv_cov = profile_dict["inv_sigma"]
        diff = latent - mean
        m_dist = float(np.sqrt(np.dot(np.dot(diff, inv_cov), diff.T)))
        
    routed_to = "CNN"
    knn_pred = -1
    nearest_neighbors = []
    expected_rewards = []
    
    if m_dist > threshold:
        routed_to = "k-NN"
        agent.k = k
        knn_pred = agent.get_action(latent, epsilon=0.0)
        expected_rewards_arr = agent.get_expected_rewards(latent)
        expected_rewards = expected_rewards_arr.tolist()
        
        # Get nearest neighbors for visualization
        query = latent.reshape(1, -1)
        distances, indices = agent._nn_index.kneighbors(query, n_neighbors=k)
        
        for dist, mem_idx in zip(distances[0], indices[0]):
            map_data = MEMORY_MAPPING[mem_idx]
            original_img = x_train[map_data['x_train_idx']]
            mem_true_label = int(y_train[map_data['x_train_idx']])
            
            if map_data['noise_level'] > 0:
                original_img = adicionar_ruido_batch(original_img[np.newaxis, ...], map_data['noise_level'])[0]
            
            nearest_neighbors.append({
                'memory_idx': int(mem_idx),
                'distance': float(dist),
                'action': int(agent._actions[mem_idx]),
                'reward': float(agent._rewards[mem_idx]),
                'is_positive': map_data['is_positive'],
                'true_label': mem_true_label,
                'base64': img_to_base64(original_img)
            })
    else:
        # Fetch expected rewards to show in the UI for context
        expected_rewards_arr = agent.get_expected_rewards(latent)
        expected_rewards = expected_rewards_arr.tolist()
        
    return jsonify({
        'true_label': true_label,
        'cnn_pred': cnn_pred,
        'cnn_conf': cnn_conf,
        'm_dist': m_dist,
        'threshold': threshold,
        'routed_to': routed_to,
        'final_pred': knn_pred if routed_to == "k-NN" else cnn_pred,
        'knn_pred': knn_pred,
        'expected_rewards': expected_rewards,
        'nearest_neighbors': nearest_neighbors,
        'query_image_base64': img_b64
    })

@app.route('/api/memory_stats', methods=['GET'])
def memory_stats():
    stats = agent.get_memory_stats()
    
    # Sample up to 100 points to avoid heavy payloads and overwhelming the UI
    sample_size = min(100, agent.memory_size)
    if sample_size > 0:
        indices = np.random.choice(agent.memory_size, sample_size, replace=False)
    else:
        indices = []
    
    memory_sample = []
    for i in indices:
        mem_img_b64 = None
        true_label = None
        if i < len(MEMORY_MAPPING):
            map_data = MEMORY_MAPPING[i]
            original_img = x_train[map_data['x_train_idx']]
            true_label = int(y_train[map_data['x_train_idx']])
            if map_data['noise_level'] > 0:
                original_img = adicionar_ruido_batch(original_img[np.newaxis, ...], map_data['noise_level'])[0]
            mem_img_b64 = img_to_base64(original_img)
            
        memory_sample.append({
            'index': int(i),
            'action': int(agent._actions[i]),
            'reward': float(agent._rewards[i]),
            'true_label': true_label,
            'base64': mem_img_b64
        })
        
    stats['memory_sample'] = memory_sample
    return jsonify(stats)

@app.route('/api/train/status', methods=['GET'])
def train_status():
    return jsonify(TRAINING_STATUS)

def calcular_mahalanobis_por_pred(vetores: np.ndarray, preds_cnn: np.ndarray,
                                   perfis: dict) -> np.ndarray:
    """
    Calculates the Mahalanobis distance of each 128D vector to the profile of the class
    PREDICTED by the CNN.
    """
    n = len(vetores)
    dists = np.zeros(n, dtype=np.float64)
    for digito in range(10):
        mask = preds_cnn == digito
        if not np.any(mask):
            continue
        info = perfis[str(digito)].item()
        mu = info["mu"]
        inv_sigma = info["inv_sigma"]
        diff = vetores[mask] - mu
        left = diff @ inv_sigma
        dist_sq = np.sum(left * diff, axis=1)
        dists[mask] = np.sqrt(np.maximum(dist_sq, 0.0))
    return dists

def avaliar_rl_batch(agent_ref, features: np.ndarray, labels: np.ndarray,
                     batch_size: int = 2048) -> float:
    """Evaluates the RL agent in optimized batches. Returns accuracy [0, 1]."""
    correct = 0
    total = len(labels)
    for i in range(0, total, batch_size):
        feats_batch = features[i : i + batch_size]
        labels_batch = labels[i : i + batch_size]
        preds_rl = agent_ref.get_action_batch(feats_batch, epsilon=0.0)
        correct += int(np.sum(preds_rl == labels_batch))
    return correct / total

def construir_dataset_ruido_misto(x: np.ndarray, y: np.ndarray,
                                   niveis: list, seed: int = 42) -> tuple:
    """
    Splits x/y into balanced partitions, applies respective noise intensity, and concatenates.
    """
    n = len(x)
    n_levels = len(niveis)
    slice_size = n // n_levels

    rng = np.random.RandomState(seed)
    indices = rng.permutation(n)

    x_slices, y_slices = [], []
    for i, level in enumerate(niveis):
        start = i * slice_size
        end = start + slice_size if i < n_levels - 1 else n
        idx = indices[start:end]
        x_slices.append(add_noise_batch(x[idx], level))
        y_slices.append(y[idx])

    return np.concatenate(x_slices, axis=0), np.concatenate(y_slices, axis=0)

def add_noise_batch(images: np.ndarray, intensity: float) -> np.ndarray:
    """Adds Gaussian noise to a batch of images."""
    if intensity <= 0.0:
        return images.copy()
    noise = np.random.normal(loc=0.0, scale=intensity, size=images.shape)
    return np.clip(images + noise, 0.0, 1.0).astype(np.float32)

def live_training_worker():
    global agent, MEMORY_MAPPING
    TRAINING_STATUS["is_running"] = True
    TRAINING_STATUS["progress"] = 0.0
    TRAINING_STATUS["logs"] = []
    
    NUM_LOTES = 10
    THRESHOLD_HYBRID = 12.5
    CENARIOS_RUIDO = [0.0, 0.2, 0.4, 0.6, 0.8]
    EVAL_SUBSET_SIZE = 2000
    
    log_to_training("Starting progressive seeding with real-time evaluation...")
    
    # ── 1. Load and Partition Dataset (10% Seed / 90% Eval) ─────────
    log_to_training("Loading full MNIST dataset for hermetic partition split...")
    TRAINING_STATUS["current_stage"] = "Loading data"
    x_full, y_full = load_mnist_raw(DATA_DIR, kind='train')
    x_full = x_full.astype(np.float32) / 255.0
    
    x_seed, x_eval, y_seed, y_eval = train_test_split(
        x_full, y_full,
        test_size=0.90,
        random_state=RANDOM_SEED,
        shuffle=True,
        stratify=y_full
    )
    log_to_training(f"  Seeding Partition (10%): {len(x_seed):,} samples")
    log_to_training(f"  Evaluation Partition (90%): {len(x_eval):,} samples")
    
    # ── 2. Precalculate Evaluation Features (GPU, once) ────────
    TRAINING_STATUS["current_stage"] = "Precalculating GPU features"
    log_to_training(f"Isolating {EVAL_SUBSET_SIZE} evaluation samples (mixed noise)...")
    
    rng = np.random.RandomState(RANDOM_SEED)
    eval_indices = rng.choice(len(x_eval), EVAL_SUBSET_SIZE, replace=False)
    x_eval_sub = x_eval[eval_indices]
    y_eval_sub = y_eval[eval_indices]
    
    # Create balanced mixed noise dataset
    x_eval_noisy, y_eval_sub = construir_dataset_ruido_misto(x_eval_sub, y_eval_sub, CENARIOS_RUIDO, seed=RANDOM_SEED)
    eval_features, eval_preds_cnn = extrair_features_128d_cnn(x_eval_noisy, cnn, batch_size=500)
    
    # Mahalanobis distances (constant, pre-calculated)
    eval_m_dists = calcular_mahalanobis_por_pred(eval_features, eval_preds_cnn, mahalanobis_profiles)
    
    # CNN-only baseline accuracy (constant across all steps)
    cnn_acc_baseline = float(np.mean(eval_preds_cnn == y_eval_sub) * 100)
    log_to_training(f"  Pre-calculated evaluation features shape: {eval_features.shape}")
    log_to_training(f"  CNN-only baseline (mixed noise): {cnn_acc_baseline:.2f}%")
    
    # ── 3. Prepare Incremental Batches ────────────────────────────────────
    TRAINING_STATUS["current_stage"] = "Progressive seeding"
    indices_seed = np.arange(len(x_seed))
    rng.shuffle(indices_seed)
    batches = np.array_split(indices_seed, NUM_LOTES)
    
    log_to_training(f"Split into {NUM_LOTES} batches (~{len(batches[0])} samples/batch)")
    log_to_training("--------------------------------------------------")
    
    # ── 4. Fresh Agent + Mapping ──────────────────────────────────────────
    new_agent = KNNBanditAgent(k=30, n_actions=10, latent_dim=128, use_pca=False)
    new_mapping = []
    
    # ── 5. Progressive Seeding Loop (10 chunks) ─────────────────────────
    for batch_idx, batch_indices in enumerate(batches):
        batch_num = batch_idx + 1
        x_batch = x_seed[batch_indices]
        y_batch = y_seed[batch_indices]
        
        log_to_training(f"── Batch {batch_num}/{NUM_LOTES} ({len(batch_indices)} samples) ──")
        TRAINING_STATUS["current_stage"] = f"Batch {batch_num}/{NUM_LOTES}"
        
        # 5a. Oracle Seeding: generate noise scenarios for current batch
        for intensity in CENARIOS_RUIDO:
            x_noisy = adicionar_ruido_batch(x_batch, intensity) if intensity > 0 else x_batch
            states, preds_cnn_batch = extrair_features_128d_cnn(x_noisy, cnn, batch_size=500)
            
            # Positive experiences (oracle label, reward +1.0)
            new_agent.add_experience_batch(states, y_batch, np.ones(len(y_batch)))
            for idx in batch_indices:
                new_mapping.append({'x_train_idx': int(idx), 'noise_level': float(intensity), 'is_positive': True})
            
            # Negative experiences (CNN error, reward -1.0)
            errors = preds_cnn_batch != y_batch
            num_errors = int(np.sum(errors))
            if num_errors > 0:
                new_agent.add_experience_batch(states[errors], preds_cnn_batch[errors], np.full(num_errors, -1.0))
                for idx in batch_indices[errors]:
                    new_mapping.append({'x_train_idx': int(idx), 'noise_level': float(intensity), 'is_positive': False})
        
        # 5b. Rebuild k-NN index
        log_to_training(f"  Rebuilding k-NN index (memory: {new_agent.memory_size:,})...")
        new_agent.build_index()
        
        # 5c. Instant evaluation on the 2,000 pre-calculated samples
        rl_acc = avaliar_rl_batch(new_agent, eval_features, y_eval_sub) * 100
        
        # Hybrid routing: dist < threshold -> CNN, else -> RL
        preds_rl_eval = new_agent.get_action_batch(eval_features, epsilon=0.0)
        mask_cnn = eval_m_dists < THRESHOLD_HYBRID
        preds_hybrid = np.where(mask_cnn, eval_preds_cnn, preds_rl_eval)
        hybrid_acc = float(np.mean(preds_hybrid == y_eval_sub) * 100)
        
        log_to_training(f"  ✓ Mem: {new_agent.memory_size:,} | "
                        f"Hybrid: {hybrid_acc:.2f}% | RL: {rl_acc:.2f}% | CNN: {cnn_acc_baseline:.2f}%")
        
        TRAINING_STATUS["knn_acc"] = float(rl_acc)
        TRAINING_STATUS["progress"] = (batch_num / NUM_LOTES) * 100
        TRAINING_STATUS["cnn_acc"] = cnn_acc_baseline
        TRAINING_STATUS["memory_size"] = new_agent.memory_size
        time.sleep(0.3)  # Small delay for UI poll sync
    
    # ── 6. Finalize: Hot-reload + Persistence ──────────────────────────
    TRAINING_STATUS["current_stage"] = "Finalizing"
    log_to_training("--------------------------------------------------")
    
    agent = new_agent
    MEMORY_MAPPING = new_mapping
    
    agent.save(MEMORY_BANK_128D_PATH)
    
    try:
        with open(MEMORY_MAPPING_PATH, "w") as f:
            json.dump(MEMORY_MAPPING, f)
    except Exception as e:
        log_to_training(f"Error saving mapping JSON: {e}")
    
    log_to_training(f"Seeding completed! Memory saved to {MEMORY_BANK_128D_PATH}.")
    log_to_training(f"  Final Results: Hybrid {hybrid_acc:.2f}% | RL {rl_acc:.2f}% | CNN {cnn_acc_baseline:.2f}%")
    
    TRAINING_STATUS["is_running"] = False
    TRAINING_STATUS["progress"] = 100.0
    TRAINING_STATUS["current_stage"] = "Completed"

@app.route('/api/train/start', methods=['POST'])
def train_start():
    if TRAINING_STATUS["is_running"]:
        return jsonify({"status": "error", "message": "Training already running."}), 400
        
    thread = threading.Thread(target=live_training_worker)
    thread.daemon = True
    thread.start()
    return jsonify({"status": "ok", "message": "Training started."})

@app.route('/api/evaluate/global', methods=['POST'])
def evaluate_global():
    """
    Runs a full global evaluation audit on unseen data.
    Sweeps across noise levels and returns per-level + global metrics,
    plus a gallery of OOD-routed samples.
    """
    data = request.json or {}
    dataset_choice = data.get('dataset', 't10k')
    THRESHOLD = 12.5
    NOISE_LEVELS = [0.0, 0.2, 0.4, 0.6, 0.8]

    # ── 1. Load dataset according to selection ────────────────────────────
    if dataset_choice == 'split_90':
        x_full, y_full = load_mnist_raw(DATA_DIR, kind='train')
        x_full = x_full.astype(np.float32) / 255.0
        _, x_eval, _, y_eval = train_test_split(
            x_full, y_full,
            test_size=0.90,
            random_state=42,
            shuffle=True,
            stratify=y_full
        )
    else:
        # Default: t10k (native MNIST test set)
        x_eval, y_eval = load_mnist_raw(DATA_DIR, kind='t10k')
        x_eval = x_eval.astype(np.float32) / 255.0

    logger.info(f"[EVAL GLOBAL] Dataset '{dataset_choice}' loaded: {len(x_eval)} samples")

    # ── 2. Build balanced mixed-noise dataset ─────────────────────────────
    x_noisy, y_noisy = construir_dataset_ruido_misto(x_eval, y_eval, NOISE_LEVELS, seed=42)
    n_total = len(y_noisy)
    n_per_level = n_total // len(NOISE_LEVELS)

    # ── 3. Extract features + CNN predictions (GPU batched) ───────────────
    features_128d, preds_cnn = extrair_features_128d_cnn(x_noisy, cnn, batch_size=500)

    # ── 4. Mahalanobis distances ──────────────────────────────────────────
    m_dists = calcular_mahalanobis_por_pred(features_128d, preds_cnn, mahalanobis_profiles)

    # ── 5. RL predictions ─────────────────────────────────────────────────
    preds_rl = agent.get_action_batch(features_128d, epsilon=0.0)

    # ── 6. Hybrid routing ─────────────────────────────────────────────────
    mask_cnn = m_dists < THRESHOLD
    preds_hybrid = np.where(mask_cnn, preds_cnn, preds_rl)

    # ── 7. Per-noise-level accuracy ───────────────────────────────────────
    hybrid_curve = []
    cnn_curve = []
    rl_curve = []

    for i, nivel in enumerate(NOISE_LEVELS):
        start = i * n_per_level
        end = start + n_per_level if i < len(NOISE_LEVELS) - 1 else n_total
        sl = slice(start, end)

        y_sl = y_noisy[sl]
        hybrid_acc_i = float(np.mean(preds_hybrid[sl] == y_sl) * 100)
        cnn_acc_i = float(np.mean(preds_cnn[sl] == y_sl) * 100)
        rl_acc_i = float(np.mean(preds_rl[sl] == y_sl) * 100)

        hybrid_curve.append(round(hybrid_acc_i, 2))
        cnn_curve.append(round(cnn_acc_i, 2))
        rl_curve.append(round(rl_acc_i, 2))

    # ── 8. Global averages ────────────────────────────────────────────────
    global_hybrid = round(float(np.mean(preds_hybrid == y_noisy) * 100), 2)
    global_cnn = round(float(np.mean(preds_cnn == y_noisy) * 100), 2)
    global_rl = round(float(np.mean(preds_rl == y_noisy) * 100), 2)
    global_rl_rate = round(float(np.mean(~mask_cnn) * 100), 2)

    logger.info(f"[EVAL GLOBAL] Hybrid={global_hybrid}% | CNN={global_cnn}% | RL={global_rl}% | RL Rate={global_rl_rate}%")

    # ── 9. Select up to 12 OOD-routed samples ─────────────────────────────
    ood_indices = np.where(m_dists >= THRESHOLD)[0]
    if len(ood_indices) > 12:
        rng = np.random.RandomState(42)
        ood_indices = rng.choice(ood_indices, 12, replace=False)

    routed_samples = []
    for idx in ood_indices:
        routed_samples.append({
            'base64': img_to_base64(x_noisy[idx]),
            'true_label': int(y_noisy[idx]),
            'cnn_pred': int(preds_cnn[idx]),
            'knn_action': int(preds_rl[idx]),
            'distance': round(float(m_dists[idx]), 2)
        })

    return jsonify({
        'global': {
            'hybrid_acc': global_hybrid,
            'cnn_acc': global_cnn,
            'rl_acc': global_rl,
            'rl_rate': global_rl_rate
        },
        'noise_curve': {
            'noises': NOISE_LEVELS,
            'hybrid': hybrid_curve,
            'cnn': cnn_curve,
            'rl': rl_curve
        },
        'routed_samples': routed_samples
    })

if __name__ == '__main__':
    # Initialize components before running
    logger.info("Initializing application data...")
    startup()
    app.run(host='0.0.0.0', port=5000, debug=False)
