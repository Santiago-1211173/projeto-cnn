"""
Script de Visualização do Espaço Latente (Fase 2 e 3).
Carrega o modelo treinado, extrai as features latentes (128D) e projeta-as em 2D.
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
    # 1. Configurar GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    # 2. Carregar o Modelo e Restaurar Pesos
    logger.info("A instanciar a rede e a carregar pesos do disco...")
    model = RawModel()
    ckpt = tf.train.Checkpoint(model=model)
    
    # Procura o último checkpoint guardado na pasta
    checkpoint_dir = os.path.join("outputs", "checkpoints")
    latest_ckpt = tf.train.latest_checkpoint(checkpoint_dir)
    if not latest_ckpt:
        logger.error("Nenhum checkpoint encontrado! Corre o train.py primeiro.")
        return
        
    ckpt.restore(latest_ckpt).expect_partial()
    logger.info(f"Pesos restaurados com sucesso de: {latest_ckpt}")

    # 3. Carregar Dados de TESTE (Imagens que a rede nunca viu)
    logger.info("A carregar dados de teste...")
    data_dir = os.path.join("data", "MNIST", "raw")
    
    # Carregamos o dataset de teste (kind='t10k') - apenas 2000 imagens para o gráfico não ficar confuso
    x_test, y_test = load_mnist_raw(data_dir, kind='t10k')
    x_test = x_test[:2000].astype(np.float32) / 255.0
    y_test = y_test[:2000]

    # 4. Extração do Espaço Latente (O Objetivo da Tarefa)
    logger.info("A dissecar a rede: extraindo as features latentes (128 dimensões)...")
    
    # Passamos as imagens pela rede. Em vez de usar as probabilidades, 
    # acedemos à chave "latent_features" do nosso dicionário.
    x_tensor = tf.convert_to_tensor(x_test)
    outputs = model(x_tensor)
    
    features_128d = outputs["latent_features"].numpy()

    # 5. Redução de Dimensionalidade Não-Linear (t-SNE)
    logger.info("A aplicar t-SNE para desdobrar de 128D para 2D (pode demorar uns 30 segundos)...")
    # O t-SNE é intensivo, mas revela os clusters reais
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    features_2d = tsne.fit_transform(features_128d)

    # 6. Gerar e Guardar o Gráfico
    logger.info("A desenhar o mapa de pensamento da CNN...")
    plt.figure(figsize=(12, 10))
    sns.set_theme(style="whitegrid")

    sns.scatterplot(
        x=features_2d[:, 0], 
        y=features_2d[:, 1], 
        hue=y_test, 
        palette=sns.color_palette("tab10", 10),
        legend="full",
        alpha=0.7,
        s=30
    )

    plt.title("Dissecação da CNN: Espaço Latente em 2D (PCA)", fontsize=16, fontweight='bold')
    plt.xlabel("Componente Principal 1", fontsize=12)
    plt.ylabel("Componente Principal 2", fontsize=12)
    plt.legend(title="Dígito Original", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    # Guardar na pasta outputs
    output_path = os.path.join("outputs", "espaco_latente_t-SNE.png")
    plt.savefig(output_path, dpi=300)
    logger.info(f"==================================================")
    logger.info(f"SUCESSO! Abre o ficheiro gerado: {output_path}")
    logger.info(f"==================================================")

if __name__ == "__main__":
    main()