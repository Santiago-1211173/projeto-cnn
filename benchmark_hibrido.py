"""
Script de Avaliação em Massa (Benchmark Final).
Testa as 10.000 imagens do MNIST em Cenário Limpo e Cenário com Ruído.
Usa processamento em Mini-Batches para proteger a VRAM e Álgebra Vetorizada para velocidade.
"""

import os
import time
import logging
import numpy as np
import tensorflow as tf

from src.models.custom_cnn import RawModel
from src.models.rl_agent import QNetworkAgent
from src.data.loader import load_mnist_raw

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

LIMIAR_MAHALANOBIS = 15.0

def calcular_mahalanobis_batch(vetores: np.ndarray, mu: np.ndarray, inv_sigma: np.ndarray) -> np.ndarray:
    """Fórmula de Mahalanobis incrivelmente rápida (Vetorizada para N imagens)."""
    diff = vetores - mu
    left = np.dot(diff, inv_sigma)
    dist_quadrada = np.sum(left * diff, axis=1)
    return np.sqrt(dist_quadrada)

def extrair_dados_em_batch(imagens: np.ndarray, cnn, agent, batch_size=500):
    """Passa as imagens pelas redes em lotes para evitar estourar a memória da GPU."""
    num_batches = len(imagens) // batch_size
    
    todos_vetores_128d = []
    todas_predicoes_cnn = []
    todas_predicoes_rl = []
    
    for i in range(num_batches):
        batch_img = imagens[i * batch_size : (i + 1) * batch_size]
        batch_tensor = tf.convert_to_tensor(batch_img, dtype=tf.float32)
        
        # 1. Extração CNN
        outputs = cnn(batch_tensor)
        vetores = outputs["latent_features"].numpy()
        preds_cnn = tf.argmax(outputs["probabilities"], axis=1).numpy()
        
        # 2. Extração RL (Acesso direto ao cérebro puro, ignorando o Epsilon)
        q_values = agent(tf.convert_to_tensor(vetores, dtype=tf.float32))
        preds_rl = tf.argmax(q_values, axis=1).numpy()
        
        todos_vetores_128d.append(vetores)
        todas_predicoes_cnn.append(preds_cnn)
        todas_predicoes_rl.append(preds_rl)
        
    return np.vstack(todos_vetores_128d), np.concatenate(todas_predicoes_cnn), np.concatenate(todas_predicoes_rl)

def avaliar_cenario(nome_cenario: str, imagens: np.ndarray, labels: np.ndarray, cnn, agent, mapa_ilhas):
    logger.info(f"\n[{nome_cenario}] A iniciar avaliação de 10.000 imagens...")
    start_time = time.time()
    
    # 1. Extração Segura em Batches (Resolve problemas de OOM)
    vetores_128d, predicoes_cnn, predicoes_rl = extrair_dados_em_batch(imagens, cnn, agent)

    # 2. O Árbitro mede distâncias (Vetorizado para todas as imagens instantaneamente)
    todas_distancias = np.zeros((len(imagens), 10))
    for digito in range(10):
        mu = mapa_ilhas[str(digito)].item()["mu"]
        inv_sigma = mapa_ilhas[str(digito)].item()["inv_sigma"]
        todas_distancias[:, digito] = calcular_mahalanobis_batch(vetores_128d, mu, inv_sigma)
    
    # Descobrir a menor distância para cada uma das imagens
    distancias_minimas = np.min(todas_distancias, axis=1)

    # 3. O Roteamento (A Decisão Híbrida)
    mascara_cnn = distancias_minimas < LIMIAR_MAHALANOBIS
    mascara_rl = ~mascara_cnn

    # 4. Recolha de Estatísticas
    total_cnn = np.sum(mascara_cnn)
    total_rl = np.sum(mascara_rl)
    
    acertos_cnn = np.sum((predicoes_cnn == labels) & mascara_cnn)
    acertos_rl = np.sum((predicoes_rl == labels) & mascara_rl)
    
    acc_cnn_isolada = (np.sum(predicoes_cnn == labels) / len(labels)) * 100
    
    total_acertos_hibrido = acertos_cnn + acertos_rl
    acc_hibrida = (total_acertos_hibrido / len(labels)) * 100
    
    tempo_total = time.time() - start_time

    # --- IMPRESSÃO DO RELATÓRIO PARA A TESE ---
    logger.info("="*60)
    logger.info(f" RELATÓRIO FINAL: {nome_cenario.upper()}")
    logger.info("="*60)
    logger.info(f"Tempo de Processamento (10k imgs): {tempo_total:.3f} segundos")
    logger.info("-" * 60)
    logger.info(f"Se a CNN estivesse SOZINHA (Sem Árbitro): Precisão de {acc_cnn_isolada:.1f}%")
    logger.info("-" * 60)
    logger.info("TRABALHO DE EQUIPA (SISTEMA HÍBRIDO):")
    logger.info(f" -> Imagens resolvidas pela CNN: {total_cnn} (Acertou {acertos_cnn})")
    logger.info(f" -> Imagens passadas para o RL : {total_rl} (Acertou {acertos_rl})")
    logger.info("-" * 60)
    logger.info(f"PRECISÃO GLOBAL DO SISTEMA HÍBRIDO: {acc_hibrida:.1f}%")
    logger.info("="*60 + "\n")


def main():
    # 1. Configurar Hardware
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    # 2. Carregar Peças do Sistema
    cnn = RawModel()
    ckpt_cnn = tf.train.Checkpoint(model=cnn)
    ckpt_cnn.restore(tf.train.latest_checkpoint(os.path.join("outputs", "checkpoints"))).expect_partial()

    mapa_ilhas = np.load(os.path.join("outputs", "mahalanobis_profiles.npz"), allow_pickle=True)

    agent = QNetworkAgent()
    ckpt_agent = tf.train.Checkpoint(model=agent)
    # Lembro-me que no teu PC o ficheiro tem o sufixo -1
    ckpt_agent.restore(os.path.join("outputs", "rl_agent_weights-1")).expect_partial()

    # 3. Carregar as 10.000 Imagens Oficiais
    # O loader já devolve com shape (10000, 28, 28, 1) devido à nossa verificação anterior!
    x_test, y_test = load_mnist_raw(os.path.join("data", "MNIST", "raw"), kind='t10k')
    x_test_limpo = x_test.astype(np.float32) / 255.0
    
    # Criar a cópia corrompida das mesmas 10.000 imagens
    ruido = np.random.normal(loc=0.0, scale=0.6, size=x_test_limpo.shape)
    x_test_ruido = np.clip(x_test_limpo + ruido, 0., 1.)

    # 4. Correr a Avaliação Massiva
    avaliar_cenario("Cenário A: Mundo Ideal (Imagens Limpas)", x_test_limpo, y_test, cnn, agent, mapa_ilhas)
    avaliar_cenario("Cenário B: Mundo Caótico (Imagens com Ruído)", x_test_ruido, y_test, cnn, agent, mapa_ilhas)

if __name__ == "__main__":
    main()