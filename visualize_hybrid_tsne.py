"""
Visualização do Colapso do Espaço Latente (t-SNE).
Compara visualmente como a CNN agrupa as imagens limpas (Mundo Ideal) 
vs. como a CNN espalha as imagens com ruído (Mundo Caótico).
Isto prova visualmente a necessidade do Árbitro e do Agente RL.
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

def main():
    # 1. Configurar Hardware
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    logger.info("==================================================")
    logger.info("A INICIAR A DISSECAÇÃO VISUAL (t-SNE COMPARATIVO)")
    logger.info("==================================================")

    # 2. Carregar o Córtex Visual (CNN)
    cnn = RawModel()
    ckpt_cnn = tf.train.Checkpoint(model=cnn)
    ckpt_cnn.restore(tf.train.latest_checkpoint(os.path.join("outputs", "checkpoints"))).expect_partial()
    logger.info("[OK] CNN carregada.")

    # 3. Preparar Dados (Apenas 2000 imagens para o t-SNE não demorar uma eternidade)
    x_test, y_test = load_mnist_raw(os.path.join("data", "MNIST", "raw"), kind='t10k')
    
    # O loader já traz (N, 28, 28, 1), por isso não precisamos de expand_dims!
    x_test_limpo = x_test[:2000].astype(np.float32) / 255.0
    labels = y_test[:2000]
    
    # Criar a versão caótica
    ruido = np.random.normal(loc=0.0, scale=0.6, size=x_test_limpo.shape)
    x_test_ruido = np.clip(x_test_limpo + ruido, 0., 1.)
    logger.info("[OK] Dados de teste preparados (Limpos e Corrompidos).")

    # 4. Extrair os Vetores de 128 Dimensões
    logger.info("A extrair features latentes através da CNN...")
    outputs_limpos = cnn(tf.convert_to_tensor(x_test_limpo, dtype=tf.float32))
    vetores_limpos = outputs_limpos["latent_features"].numpy()

    outputs_ruido = cnn(tf.convert_to_tensor(x_test_ruido, dtype=tf.float32))
    vetores_ruido = outputs_ruido["latent_features"].numpy()

    # 5. Aplicar o t-SNE
    # Nota: Executamos duas reduções independentes para ver como a geometria muda
    logger.info("A calcular o t-SNE para o Cenário Limpo (Aguarde...)")
    tsne_limpo = TSNE(n_components=2, random_state=42, perplexity=30)
    features_2d_limpo = tsne_limpo.fit_transform(vetores_limpos)

    logger.info("A calcular o t-SNE para o Cenário de Ruído (Aguarde...)")
    tsne_ruido = TSNE(n_components=2, random_state=42, perplexity=30)
    features_2d_ruido = tsne_ruido.fit_transform(vetores_ruido)

    # 6. Desenhar a Obra de Arte
    logger.info("A desenhar os gráficos...")
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    sns.set_theme(style="whitegrid")
    
    cores = sns.color_palette("tab10", 10)

    # Gráfico A: O Mundo Ideal
    sns.scatterplot(
        ax=axes[0],
        x=features_2d_limpo[:, 0], y=features_2d_limpo[:, 1], 
        hue=labels, palette=cores, legend=False, alpha=0.7, s=40
    )
    axes[0].set_title("Cenário A: CNN em Mundo Ideal (Imagens Limpas)\nIlhas Perfeitas -> Decisão fácil para a CNN", fontsize=14, fontweight='bold')
    axes[0].set_xlabel("t-SNE 1")
    axes[0].set_ylabel("t-SNE 2")

    # Gráfico B: O Mundo Caótico
    sns.scatterplot(
        ax=axes[1],
        x=features_2d_ruido[:, 0], y=features_2d_ruido[:, 1], 
        hue=labels, palette=cores, legend="full", alpha=0.7, s=40
    )
    axes[1].set_title("Cenário B: CNN sob Efeito de Ruído\nColapso das Ilhas -> Intervenção do Árbitro e Agente RL", fontsize=14, fontweight='bold')
    axes[1].set_xlabel("t-SNE 1")
    axes[1].set_ylabel("t-SNE 2")
    axes[1].legend(title="Dígito Original", bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    
    output_path = os.path.join("outputs", "colapso_latente_tsne.png")
    plt.savefig(output_path, dpi=300)
    logger.info(f"==================================================")
    logger.info(f"[SUCESSO] Gráfico guardado em: {output_path}")
    logger.info("==================================================")

if __name__ == "__main__":
    main()