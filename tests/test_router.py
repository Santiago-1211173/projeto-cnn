"""
Script de Teste do Árbitro de Triagem (CNN vs RL).
Usa as distâncias de Mahalanobis para decidir se a CNN tem 
certeza absoluta ou se a imagem deve ser reencaminhada para o Agente de RL.
"""

import os
import logging
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from src.models.custom_cnn import RawModel
from src.data.loader import load_mnist_raw

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def calcular_mahalanobis(vetor_x: np.ndarray, mu: np.ndarray, inv_sigma: np.ndarray) -> float:
    """Fórmula matemática da distância de Mahalanobis."""
    diff = vetor_x - mu
    # D = sqrt( (x - mu)^T * Sigma^-1 * (x - mu) )
    dist_quadrada = np.dot(np.dot(diff, inv_sigma), diff.T)
    return float(np.sqrt(dist_quadrada))

def adicionar_ruido(imagem: np.ndarray, intensidade: float = 0.5) -> np.ndarray:
    """Injeta estática na imagem para simular um caso muito difícil/ambíguo."""
    ruido = np.random.normal(loc=0.0, scale=intensidade, size=imagem.shape)
    imagem_corrompida = imagem + ruido
    return np.clip(imagem_corrompida, 0., 1.)

def main():
    # 1. Carregar CNN
    model = RawModel()
    ckpt = tf.train.Checkpoint(model=model)
    latest_ckpt = tf.train.latest_checkpoint(os.path.join("outputs", "checkpoints"))
    ckpt.restore(latest_ckpt).expect_partial()

    # 2. Carregar o Mapa de Mahalanobis (O cérebro do Árbitro)
    caminho_mapa = os.path.join("outputs", "mahalanobis_profiles.npz")
    # FIX APLICADO: allow_pickle=True para autorizar o NumPy a ler dicionários
    mapa_ilhas = np.load(caminho_mapa, allow_pickle=True)
    logger.info("Mapa de Mahalanobis carregado!")

    # 3. Carregar Imagens de Teste
    x_test, y_test = load_mnist_raw(os.path.join("data", "MNIST", "raw"), kind='t10k')
    x_test = x_test.astype(np.float32) / 255.0

    # 4. Selecionar uma imagem (ex: a primeira imagem de teste, que é um "7")
    idx = 0
    img_limpa = x_test[idx]
    label_real = y_test[idx]
    
    # Criar a versão corrompida dessa mesma imagem
    img_corrompida = adicionar_ruido(img_limpa, intensidade=0.6)

    # Preparar para o TensorFlow (adicionar dimensão do batch [1, 28, 28, 1])
    # FIX APLICADO: dtype=tf.float32 explicitamente forçado em ambos os tensores
    batch_limpo = tf.convert_to_tensor(np.expand_dims(img_limpa, axis=0), dtype=tf.float32)
    batch_corrompido = tf.convert_to_tensor(np.expand_dims(img_corrompida, axis=0), dtype=tf.float32)

    # 5. Passar pela CNN (Extrair Vetores 128D)
    vetor_limpo = model(batch_limpo)["latent_features"].numpy()[0]
    vetor_corrompido = model(batch_corrompido)["latent_features"].numpy()[0]

    # 6. O TRABALHO DO ÁRBITRO: Medir a distância mínima a uma ilha
    def avaliar_vetor(vetor: np.ndarray, tipo: str):
        menor_distancia = float('inf')
        digito_mais_proximo = -1

        for digito in range(10):
            mu = mapa_ilhas[str(digito)].item()["mu"]
            inv_sigma = mapa_ilhas[str(digito)].item()["inv_sigma"]
            
            dist = calcular_mahalanobis(vetor, mu, inv_sigma)
            
            if dist < menor_distancia:
                menor_distancia = dist
                digito_mais_proximo = digito
                
        logger.info(f"--- Imagem {tipo} ---")
        logger.info(f"A ilha mais próxima é o '{digito_mais_proximo}'.")
        logger.info(f"Distância de Mahalanobis: {menor_distancia:.2f}")
        return menor_distancia, digito_mais_proximo

    dist_limpa, _ = avaliar_vetor(vetor_limpo, "LIMPA")
    dist_corr, _ = avaliar_vetor(vetor_corrompido, "CORROMPIDA (Com Ruído)")

    # 7. Visualizar para confirmar
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    
    axes[0].imshow(img_limpa[:,:,0], cmap='gray')
    axes[0].set_title(f"Limpa\nDistância à Ilha: {dist_limpa:.1f}")
    axes[0].axis('off')
    
    axes[1].imshow(img_corrompida[:,:,0], cmap='gray')
    axes[1].set_title(f"Outlier\nDistância à Ilha: {dist_corr:.1f}")
    axes[1].axis('off')
    
    output_path = os.path.join("outputs", "teste_triagem.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    logger.info(f"\nImagem guardada em: {output_path}")

if __name__ == "__main__":
    main()