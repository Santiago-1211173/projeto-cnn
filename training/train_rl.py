"""
Motor de Treino do Agente RL (k-NN Bandit — Memória Episódica).

ESTRATÉGIA FINAL: Oracle no nível de ruído alvo.
  - Gera múltiplas realizações de ruído a 0.6 (que é o nível de teste)
  - Para cada imagem: armazena (estado_10D, label_correta, +1)
  - Também armazena (estado_10D, previsão_errada_CNN, -1) para contrastel
  - Sem exploração aleatória, sem mistura de escalas
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import os
import logging
import numpy as np
import tensorflow as tf

from src.models.custom_cnn import RawModel
from src.models.knn_bandit_agent import KNNBanditAgent
from src.data.loader import load_mnist_raw

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def adicionar_ruido_batch(imagens: np.ndarray, intensidade: float = 0.6) -> np.ndarray:
    """Injeta estática num batch inteiro de imagens."""
    ruido = np.random.normal(loc=0.0, scale=intensidade, size=imagens.shape)
    return np.clip(imagens + ruido, 0., 1.)


def extrair_estados_cnn(imagens: np.ndarray, cnn, batch_size: int = 500) -> np.ndarray:
    """Passa as imagens pela CNN em lotes e extrai os arrays de 10D (probabilidades)."""
    todos_estados = []
    for i in range(0, len(imagens), batch_size):
        batch = imagens[i : i + batch_size]
        batch_tensor = tf.convert_to_tensor(batch, dtype=tf.float32)
        outputs = cnn(batch_tensor)
        probs = outputs["probabilities"].numpy()
        todos_estados.append(probs)
    return np.vstack(todos_estados)


def main():
    # 1. Configurar Hardware
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    # 2. Carregar CNN
    logger.info("A carregar a CNN (Extrator de Features)...")
    cnn = RawModel()
    ckpt = tf.train.Checkpoint(model=cnn)
    latest_ckpt = tf.train.latest_checkpoint(os.path.join("outputs", "checkpoints"))
    ckpt.restore(latest_ckpt).expect_partial()

    # 3. Inicializar Agente (k=30 com scoring ponderado por distância)
    logger.info("A instanciar o Agente Especialista k-NN Bandit...")
    agent = KNNBanditAgent(k=30, n_actions=10)

    # 4. Carregar Dados
    logger.info("A preparar o Ambiente de Treino...")
    x_train, y_train = load_mnist_raw(os.path.join("data", "MNIST", "raw"), kind='train')
    x_train = x_train.astype(np.float32) / 255.0

    # ================================================================
    # ORACLE SEEDING: Múltiplas realizações de ruído a 0.6
    # ================================================================
    # A ideia: o teste usa ruído=0.6, então a memória precisa de estados
    # do MESMO nível. Geramos N versões ruidosas diferentes de cada imagem
    # para cobrir densamente o espaço de estados possíveis.
    # ================================================================

    logger.info("\n==================================================")
    logger.info("ORACLE SEEDING (Ruído 0.6 × 3 realizações)")
    logger.info("==================================================")

    n_realizacoes = 3  # 3 versões ruidosas de cada imagem
    intensidade = 0.6

    for r in range(n_realizacoes):
        logger.info(f"\n  Realização {r+1}/{n_realizacoes}...")
        
        # Gerar ruído aleatório diferente a cada realização
        x_ruido = adicionar_ruido_batch(x_train, intensidade)
        
        # Extrair estados 10D
        estados = extrair_estados_cnn(x_ruido, cnn, batch_size=500)
        preds_cnn = np.argmax(estados, axis=1)
        
        # Experiências POSITIVAS: label correta → +1
        agent.add_experience_batch(estados, y_train, np.ones(len(y_train)))
        
        # Experiências NEGATIVAS: onde a CNN erra, marcar a ação errada como -1
        erros = preds_cnn != y_train
        n_erros = np.sum(erros)
        if n_erros > 0:
            agent.add_experience_batch(
                estados[erros], preds_cnn[erros], np.full(n_erros, -1.0)
            )
        
        acc_cnn = np.mean(preds_cnn == y_train) * 100
        logger.info(f"    +{len(y_train):,} positivas, +{n_erros:,} negativas | CNN acc: {acc_cnn:.1f}%")

    # Construir índice
    agent.build_index()

    # ================================================================
    # AVALIAÇÃO RÁPIDA
    # ================================================================
    logger.info("\n==================================================")
    logger.info("AVALIAÇÃO RÁPIDA")
    logger.info("==================================================")

    sample_idx = np.random.choice(len(x_train), size=3000, replace=False)

    for nivel in [0.0, 0.3, 0.6]:
        if nivel > 0:
            sample_imgs = adicionar_ruido_batch(x_train[sample_idx], nivel)
        else:
            sample_imgs = x_train[sample_idx]

        sample_s = extrair_estados_cnn(sample_imgs, cnn, batch_size=500)
        acc_knn = np.mean(agent.get_action_batch(sample_s, epsilon=0.0) == y_train[sample_idx]) * 100
        acc_cnn = np.mean(np.argmax(sample_s, axis=1) == y_train[sample_idx]) * 100
        nome = f"Ruído {nivel:.1f}" if nivel > 0 else "Limpo"
        logger.info(f"  [{nome}] CNN: {acc_cnn:.1f}% | k-NN: {acc_knn:.1f}%")

    # Estatísticas
    stats = agent.get_memory_stats()
    logger.info(f"\n  Memória Total: {stats['size']:,}")
    logger.info(f"  % Positivas: {stats['reward_positive_pct']:.1f}%")

    # Guardar
    caminho = os.path.join("outputs", "knn_memory_bank.npz")
    agent.save(caminho)
    logger.info(f"\nConcluído! Memória guardada em: {caminho}")


if __name__ == "__main__":
    main()