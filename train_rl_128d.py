"""
Motor de Treino do Agente RL 128D (k-NN Bandit — Memória Episódica).
"""

import os
import logging
import numpy as np
import tensorflow as tf

from src.models.custom_cnn import RawModel
from src.models.knn_bandit_agent_128d import KNNBanditAgent128D
from src.data.loader import load_mnist_raw

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def adicionar_ruido_batch(imagens: np.ndarray, intensidade: float = 0.6) -> np.ndarray:
    ruido = np.random.normal(loc=0.0, scale=intensidade, size=imagens.shape)
    return np.clip(imagens + ruido, 0., 1.)

def extrair_features_128d_cnn(imagens: np.ndarray, cnn, batch_size: int = 500):
    todos_estados = []
    todas_preds = []
    for i in range(0, len(imagens), batch_size):
        batch = imagens[i : i + batch_size]
        batch_tensor = tf.convert_to_tensor(batch, dtype=tf.float32)
        outputs = cnn(batch_tensor)
        latent = outputs["latent_features"].numpy()
        probs = outputs["probabilities"].numpy()
        preds = np.argmax(probs, axis=1)
        todos_estados.append(latent)
        todas_preds.append(preds)
    return np.vstack(todos_estados), np.concatenate(todas_preds)

def main():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    logger.info("A carregar a CNN (Extrator de Features)...")
    cnn = RawModel()
    ckpt = tf.train.Checkpoint(model=cnn)
    latest_ckpt = tf.train.latest_checkpoint(os.path.join("outputs", "checkpoints"))
    ckpt.restore(latest_ckpt).expect_partial()

    logger.info("A instanciar o Agente Especialista k-NN Bandit 128D...")
    # Using PCA to reduce 128D to 48D for faster retrieval and potential denoising, or 128D directly.
    agent = KNNBanditAgent128D(k=30, n_actions=10, use_pca=False) 

    logger.info("A preparar o Ambiente de Treino...")
    x_train, y_train = load_mnist_raw(os.path.join("data", "MNIST", "raw"), kind='train')
    x_train = x_train.astype(np.float32) / 255.0

    logger.info("\n==================================================")
    logger.info("ORACLE SEEDING (Limpo × 1 + Ruído 0.6 × 3)")
    logger.info("==================================================")

    cenarios = [0.0, 0.6, 0.6, 0.6]

    for r, intensidade in enumerate(cenarios):
        logger.info(f"\n  Realização {r+1}/{len(cenarios)} (Ruído {intensidade})...")
        x_ruido = adicionar_ruido_batch(x_train, intensidade) if intensidade > 0 else x_train
        
        estados, preds_cnn = extrair_features_128d_cnn(x_ruido, cnn, batch_size=500)
        
        # Positivas
        agent.add_experience_batch(estados, y_train, np.ones(len(y_train)))
        
        # Negativas
        erros = preds_cnn != y_train
        n_erros = np.sum(erros)
        if n_erros > 0:
            agent.add_experience_batch(
                estados[erros], preds_cnn[erros], np.full(n_erros, -1.0)
            )
        
        acc_cnn = np.mean(preds_cnn == y_train) * 100
        logger.info(f"    +{len(y_train):,} positivas, +{n_erros:,} negativas | CNN acc: {acc_cnn:.1f}%")

    logger.info("A construir o índice k-NN...")
    agent.build_index()

    logger.info("\n==================================================")
    logger.info("AVALIAÇÃO RÁPIDA")
    logger.info("==================================================")

    sample_idx = np.random.choice(len(x_train), size=3000, replace=False)

    for nivel in [0.0, 0.3, 0.6]:
        if nivel > 0:
            sample_imgs = adicionar_ruido_batch(x_train[sample_idx], nivel)
        else:
            sample_imgs = x_train[sample_idx]

        sample_s, preds_cnn = extrair_features_128d_cnn(sample_imgs, cnn, batch_size=500)
        acc_knn = np.mean(agent.get_action_batch(sample_s, epsilon=0.0) == y_train[sample_idx]) * 100
        acc_cnn = np.mean(preds_cnn == y_train[sample_idx]) * 100
        nome = f"Ruído {nivel:.1f}" if nivel > 0 else "Limpo"
        logger.info(f"  [{nome}] CNN: {acc_cnn:.1f}% | k-NN 128D: {acc_knn:.1f}%")

    stats = agent.get_memory_stats()
    logger.info(f"\n  Memória Total: {stats['size']:,}")
    logger.info(f"  % Positivas: {stats['reward_positive_pct']:.1f}%")

    caminho = os.path.join("outputs", "knn_memory_bank_128d.npz")
    agent.save(caminho)
    logger.info(f"\nConcluído! Memória guardada em: {caminho}")

if __name__ == "__main__":
    main()
