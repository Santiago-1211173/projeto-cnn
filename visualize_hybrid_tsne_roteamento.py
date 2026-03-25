"""
Visualização Avançada do Roteamento Híbrido (t-SNE).
Compara o espaço latente de 2000 imagens limpas vs. 2000 com ruído, 
usando HUE para o Dígito Real e STYLE para a Decisão do Árbitro (CNN vs. RL).
Prova visual da Triagem Híbrida em ação.
"""

import os
import logging
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

from src.models.custom_cnn import RawModel
from src.data.loader import load_mnist_raw

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Limiar de Incerteza (Threshold) descoberto nos nossos testes anteriores
LIMIAR_MAHALANOBIS = 15.0  

def calcular_mahalanobis_batch(vetores: np.ndarray, mu: np.ndarray, inv_sigma: np.ndarray) -> np.ndarray:
    """Fórmula de Mahalanobis vetorizada para N imagens."""
    diff = vetores - mu
    left = np.dot(diff, inv_sigma)
    dist_quadrada = np.sum(left * diff, axis=1)
    return np.sqrt(dist_quadrada)

def get_embeddings_and_routing(imagens, cnn, mapa_ilhas):
    """Extrai os vetores 128D e calcula a decisão do Árbitro para cada um."""
    # 1. CNN extrai vetores (Batch)
    outputs = cnn(tf.convert_to_tensor(imagens, dtype=tf.float32))
    vetores_128d = outputs["latent_features"].numpy()

    # 2. Árbitro calcula distâncias (Batch)
    todas_distancias = np.zeros((len(imagens), 10))
    for digito in range(10):
        mu = mapa_ilhas[str(digito)].item()["mu"]
        inv_sigma = mapa_ilhas[str(digito)].item()["inv_sigma"]
        todas_distancias[:, digito] = calcular_mahalanobis_batch(vetores_128d, mu, inv_sigma)
    
    # Descobrir a menor distância para cada imagem
    distancias_minimas = np.min(todas_distancias, axis=1)

    # 3. Decidir o Roteamento
    # True (Circle) se CNN domina, False (X) se RL é acionado
    roteamento = np.where(distancias_minimas < LIMIAR_MAHALANOBIS, "CNN Dominante", "Roteado para RL")
    
    return vetores_128d, roteamento

def main():
    # 1. Configurar Hardware
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    logger.info("==================================================")
    logger.info("A INICIAR A DISSECAÇÃO VISUAL DO ROTEAMENTO HÍBRIDO")
    logger.info("==================================================")

    # 2. Carregar Peças (CNN e Mahalanobis)
    cnn = RawModel()
    ckpt_cnn = tf.train.Checkpoint(model=cnn)
    ckpt_cnn.restore(tf.train.latest_checkpoint(os.path.join("outputs", "checkpoints"))).expect_partial()
    
    caminho_mapa = os.path.join("outputs", "mahalanobis_profiles.npz")
    mapa_ilhas = np.load(caminho_mapa, allow_pickle=True)
    logger.info("[OK] CNN e Mapa de Mahalanobis carregados.")

    # 3. Preparar Dados (2000 imagens)
    x_test, y_test = load_mnist_raw(os.path.join("data", "MNIST", "raw"), kind='t10k')
    x_test_limpo = x_test[:2000].astype(np.float32) / 255.0
    labels = y_test[:2000]
    
    # Criar a versão caótica
    ruido = np.random.normal(loc=0.0, scale=0.6, size=x_test_limpo.shape)
    x_test_ruido = np.clip(x_test_limpo + ruido, 0., 1.)
    logger.info("[OK] Dados preparados.")

    # 4. Extrair Features e Roteamento
    logger.info("A extrair features e a calcular roteamento híbrido...")
    vetores_limpos, rot_limpos = get_embeddings_and_routing(x_test_limpo, cnn, mapa_ilhas)
    vetores_ruido, rot_ruido = get_embeddings_and_routing(x_test_ruido, cnn, mapa_ilhas)

    # 5. Aplicar o t-SNE
    logger.info("A calcular o t-SNE para o Cenário Limpo (Aguarde...)")
    tsne_limpo = TSNE(n_components=2, random_state=42, perplexity=30)
    features_2d_limpo = tsne_limpo.fit_transform(vetores_limpos)

    logger.info("A calcular o t-SNE para o Cenário de Ruído (Aguarde...)")
    tsne_ruido = TSNE(n_components=2, random_state=42, perplexity=30)
    features_2d_ruido = tsne_ruido.fit_transform(vetores_ruido)

    # 6. Desenhar a Obra de Arte Final
    logger.info("A desenhar o painel comparativo...")
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    sns.set_theme(style="whitegrid")
    
    cores = sns.color_palette("tab10", 10)
    # Definir Círculo para CNN, X para RL
    marcadores = {"CNN Dominante": "o", "Roteado para RL": "X"}

    # Painel Esquerdo: Mundo Ideal
    # Mostra quem o Árbitro aceitou nas imagens limpas
    sns.scatterplot(
        ax=axes[0],
        x=features_2d_limpo[:, 0], y=features_2d_limpo[:, 1], 
        hue=labels, palette=cores, style=rot_limpos, markers=marcadores, 
        alpha=0.8, s=60, legend=False
    )
    axes[0].set_title("Cenário A: CNN em Mundo Ideal (Imagens Limpas)\nGeometria Organizada -> Roteamento para CNN Dominante (Círculos)", fontsize=14, fontweight='bold')
    axes[0].set_xlabel("t-SNE 1")
    axes[0].set_ylabel("t-SNE 2")

    # Painel Direito: Mundo Hostil
    # Mostra quem o Árbitro aceitou nas imagens com ruído
    sns.scatterplot(
        ax=axes[1],
        x=features_2d_ruido[:, 0], y=features_2d_ruido[:, 1], 
        hue=labels, palette=cores, style=rot_ruido, markers=marcadores, 
        alpha=0.8, s=60, legend="full"
    )
    axes[1].set_title("Cenário B: CNN sob Ruído Hostil\nGeometria Colapsada -> Roteamento para Especialista RL (X's)", fontsize=14, fontweight='bold')
    axes[1].set_xlabel("t-SNE 1")
    axes[1].set_ylabel("t-SNE 2")
    
    # Organizar Legendas
    leg = axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    leg.get_frame().set_linewidth(1.0)
    
    plt.tight_layout()
    
    output_path = os.path.join("outputs", "painel_roteamento_hibrido_tsne.png")
    plt.savefig(output_path, dpi=300)
    logger.info(f"==================================================")
    logger.info(f"[SUCESSO] Painel final guardado em: {output_path}")
    logger.info("==================================================")

if __name__ == "__main__":
    main()