"""
Pipeline Final de Inferência Híbrida em Produção.
Testa o sistema completo: 
Imagem -> CNN (10D Probabilidades) -> Árbitro (Mahalanobis) -> Decisão (CNN ou k-NN RL).
"""

import os
import logging
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from src.models.custom_cnn import RawModel
from src.models.knn_bandit_agent import KNNBanditAgent
from src.data.loader import load_mnist_raw

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Limiar de Incerteza (Threshold) descoberto nos nossos testes anteriores
LIMIAR_MAHALANOBIS = 10.0  


def calcular_mahalanobis(vetor_x: np.ndarray, mu: np.ndarray, inv_sigma: np.ndarray) -> float:
    """Fórmula matemática da distância de Mahalanobis."""
    diff = vetor_x - mu
    dist_quadrada = np.dot(np.dot(diff, inv_sigma), diff.T)
    return float(np.sqrt(dist_quadrada))


def adicionar_ruido(imagem: np.ndarray, intensidade: float = 0.6) -> np.ndarray:
    ruido = np.random.normal(loc=0.0, scale=intensidade, size=imagem.shape)
    return np.clip(imagem + ruido, 0., 1.)


def main():
    # 1. Configurar Hardware
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    logger.info("==================================================")
    logger.info("A INICIALIZAR O PIPELINE HÍBRIDO DE PRODUÇÃO...")
    logger.info("==================================================")

    # 2. Carregar o Córtex Visual (CNN)
    cnn = RawModel()
    ckpt_cnn = tf.train.Checkpoint(model=cnn)
    latest_ckpt = tf.train.latest_checkpoint(os.path.join("outputs", "checkpoints"))
    ckpt_cnn.restore(latest_ckpt).expect_partial()
    logger.info("[OK] CNN carregada.")

    # 3. Carregar o Árbitro (Mapa de Mahalanobis)
    caminho_mapa = os.path.join("outputs", "mahalanobis_profiles.npz")
    mapa_ilhas = np.load(caminho_mapa, allow_pickle=True)
    logger.info("[OK] Árbitro de Triagem preparado.")

    # 4. Carregar o Cérebro Especialista (Agente k-NN)
    agent = KNNBanditAgent(k=30, n_actions=10)
    caminho_memoria = os.path.join("outputs", "knn_memory_bank.npz")
    agent.load(caminho_memoria)
    logger.info("[OK] Especialista k-NN ativado.")

    # 5. Preparar Dados de Teste
    x_test, y_test = load_mnist_raw(os.path.join("data", "MNIST", "raw"), kind='t10k')
    x_test = x_test.astype(np.float32) / 255.0

    # Vamos selecionar 6 imagens (3 normais e 3 corrompidas)
    indices = [10, 42, 99] # Três imagens aleatórias
    imagens_teste = []
    labels_reais = []
    tipos = []

    for idx in indices:
        # Adicionar a versão LIMPA
        imagens_teste.append(x_test[idx])
        labels_reais.append(y_test[idx])
        tipos.append("Limpa")
        
        # Adicionar a versão CORROMPIDA da mesma imagem
        imagens_teste.append(adicionar_ruido(x_test[idx], intensidade=0.6))
        labels_reais.append(y_test[idx])
        tipos.append("Ruído")

    # 6. PIPELINE DE INFERÊNCIA
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    logger.info("\n--- INÍCIO DA INFERÊNCIA ---")
    
    for i in range(len(imagens_teste)):
        img = imagens_teste[i]
        label_real = labels_reais[i]
        
        # A. Preparar Tensor
        tensor_img = tf.convert_to_tensor(np.expand_dims(img, axis=0), dtype=tf.float32)
        
        # B. CNN faz a extração (Passo Único Inicial)
        outputs = cnn(tensor_img)
        vetor_128d = outputs["latent_features"]
        prob_cnn = outputs["probabilities"][0]
        estado_10d = prob_cnn.numpy()  # O estado para o k-NN
        predicao_cnn = int(tf.argmax(prob_cnn).numpy())
        
        # C. Árbitro mede a distância (usa o espaço latente 128D)
        vetor_np = vetor_128d.numpy()[0]
        menor_distancia = float('inf')
        
        for digito in range(10):
            mu = mapa_ilhas[str(digito)].item()["mu"]
            inv_sigma = mapa_ilhas[str(digito)].item()["inv_sigma"]
            dist = calcular_mahalanobis(vetor_np, mu, inv_sigma)
            if dist < menor_distancia:
                menor_distancia = dist

        # D. DECISÃO DE ROTEAMENTO (A Magia do Híbrido)
        if menor_distancia < LIMIAR_MAHALANOBIS:
            decisor = "CNN"
            decisao_final = predicao_cnn
        else:
            decisor = "k-NN RL"
            # O Agente usa as probabilidades 10D para decidir
            decisao_final = agent.get_action(estado_10d, epsilon=0.0)

        # E. Registar Resultados
        sucesso = "V" if decisao_final == label_real else "X"
        cor = "green" if sucesso == "V" else "red"
        
        logger.info(f"Img {i+1} ({tipos[i]}): Dist={menor_distancia:.1f} | Decisor={decisor} | Pred={decisao_final} (Real={label_real}) [{sucesso}]")

        # F. Visualização
        axes[i].imshow(img[:, :, 0], cmap='gray')
        axes[i].set_title(
            f"Tipo: {tipos[i]}\nDistância: {menor_distancia:.1f}\nDecisor: {decisor}\nPrevisão: {decisao_final} (Real: {label_real})",
            color=cor, fontweight='bold'
        )
        axes[i].axis('off')

    plt.tight_layout()
    output_path = os.path.join("outputs", "resultado_hibrido.png")
    plt.savefig(output_path, dpi=300)
    logger.info(f"\n[OK] Gráfico final guardado em: {output_path}")


if __name__ == "__main__":
    main()