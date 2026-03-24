"""
Script de Perfilagem Espacial (Mahalanobis).
Passa o dataset de treino pela CNN para calcular o centro geométrico (Média) 
e a dispersão (Matriz de Covariância) das 10 "ilhas" no espaço latente de 128D.
"""

import os
import logging
import numpy as np
import tensorflow as tf

from src.models.custom_cnn import RawModel
from src.data.loader import load_mnist_raw

# Configuração de Logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def main():
    # 1. Configurar Hardware
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    # 2. Carregar o Modelo e Restaurar Pesos
    logger.info("A carregar o cérebro da CNN...")
    model = RawModel()
    ckpt = tf.train.Checkpoint(model=model)
    
    latest_ckpt = tf.train.latest_checkpoint(os.path.join("outputs", "checkpoints"))
    if not latest_ckpt:
        logger.error("Checkpoint não encontrado. Treina o modelo primeiro!")
        return
    ckpt.restore(latest_ckpt).expect_partial()
    logger.info("Pesos restaurados com sucesso.")

    # 3. Carregar Dados de TREINO (É com eles que desenhamos o mapa oficial)
    logger.info("A carregar dataset de treino para extração...")
    data_dir = os.path.join("data", "MNIST", "raw")
    x_train, y_train = load_mnist_raw(data_dir, kind='train')
    
    # Usar uma amostra representativa (ex: 20.000 imagens) para não estourar a RAM
    # ao calcular matrizes gigantes de covariância.
    amostras = 20000
    x_train = x_train[:amostras].astype(np.float32) / 255.0
    y_train = y_train[:amostras]

    # 4. Extração em Lote (Batching) para a GPU não sufocar
    logger.info("A extrair as features de 128D (O Espaço Latente)...")
    batch_size = 500
    features_list = []
    
    for i in range(0, len(x_train), batch_size):
        batch_x = x_train[i : i + batch_size]
        batch_tensor = tf.convert_to_tensor(batch_x)
        
        # Extrair dicionário e isolar as características latentes
        outputs = model(batch_tensor)
        features_list.append(outputs["latent_features"].numpy())
        
    all_features = np.vstack(features_list)

    # 5. Calcular a Matemática de cada "Ilha"
    logger.info("A calcular o Centroide e a Covariância para cada dígito...")
    
    perfis_mahalanobis = {}
    
    for digito in range(10):
        # 5.1. Isolar apenas os vetores que pertencem a este dígito
        indices = np.where(y_train == digito)[0]
        features_do_digito = all_features[indices]
        
        # 5.2. O Centro da Ilha (Média de cada uma das 128 dimensões)
        mu = np.mean(features_do_digito, axis=0)
        
        # 5.3. O Formato da Ilha (Matriz de Covariância 128x128)
        # rowvar=False indica que cada coluna é uma variável (as 128 features)
        sigma = np.cov(features_do_digito, rowvar=False)
        
        # 5.4. Estabilidade Numérica: Adicionar um epsilon minúsculo à diagonal
        # Isto garante que a matriz é invertível na matemática do Mahalanobis
        epsilon = 1e-6
        sigma += np.eye(sigma.shape[0]) * epsilon
        
        # 5.5. A Inversa da Covariância (É isto que usamos na fórmula final)
        inv_sigma = np.linalg.inv(sigma)
        
        perfis_mahalanobis[str(digito)] = {
            "mu": mu,
            "inv_sigma": inv_sigma
        }
        logger.info(f"  -> Perfil do dígito {digito} mapeado com sucesso.")

    # 6. Guardar o Mapa no disco
    output_path = os.path.join("outputs", "mahalanobis_profiles.npz")
    np.savez_compressed(output_path, **perfis_mahalanobis)
    
    logger.info("==================================================")
    logger.info(f"SUCESSO! Mapa das Ilhas guardado em: {output_path}")
    logger.info("==================================================")

if __name__ == "__main__":
    main()