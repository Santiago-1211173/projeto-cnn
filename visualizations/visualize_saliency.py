"""
Script de Visualização de Saliência (Fase 4 - XAI).
Calcula os gradientes da previsão em relação aos píxeis da entrada para 
descobrir exatamente onde a CNN está a focar a sua atenção.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import os
import logging
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from src.models.custom_cnn import RawModel
from src.data.loader import load_mnist_raw

# Configuração de Logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def compute_saliency_map(model: tf.Module, image: tf.Tensor) -> np.ndarray:
    """
    Calcula o mapa de saliência usando a derivada da previsão em relação à imagem.
    """
    # Converter a imagem para um Tensor monitorizado pelo GradientTape
    image_tensor = tf.convert_to_tensor(image, dtype=tf.float32)
    
    with tf.GradientTape() as tape:
        # Dizer explicitamente ao TensorFlow para observar os píxeis da imagem
        tape.watch(image_tensor)
        
        # Forward pass (A rede devolve o nosso dicionário dissecado)
        outputs = model(image_tensor)
        probabilities = outputs["probabilities"]
        
        # Descobrir qual foi a classe com maior probabilidade
        predicted_class_idx = tf.argmax(probabilities[0])
        # Isolar a probabilidade exata dessa classe vencedora
        winning_score = probabilities[0, predicted_class_idx]

    # MAGIA XAI: Calcular o gradiente da probabilidade da classe vencedora
    # em relação aos píxeis da imagem de entrada.
    gradients = tape.gradient(winning_score, image_tensor)
    
    # Obter os valores absolutos dos gradientes
    saliency = tf.abs(gradients)
    
    # Normalizar o mapa de calor entre 0 e 1 para podermos desenhar
    saliency_max = tf.reduce_max(saliency)
    saliency_min = tf.reduce_min(saliency)
    saliency_normalized = (saliency - saliency_min) / (saliency_max - saliency_min + 1e-10)
    
    return saliency_normalized[0].numpy(), int(predicted_class_idx.numpy())

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

    # 3. Carregar algumas imagens de teste
    logger.info("A carregar imagens de teste...")
    data_dir = os.path.join("data", "MNIST", "raw")
    x_test, y_test = load_mnist_raw(data_dir, kind='t10k')
    
    # Normalizar imagens
    x_test = x_test.astype(np.float32) / 255.0

    # 4. Selecionar imagens interessantes (Vamos escolher as primeiras 5 imagens)
    num_images_to_show = 5
    
    plt.figure(figsize=(15, 3 * num_images_to_show))
    
    logger.info("A calcular derivadas parciais (Saliência) para as imagens...")
    for i in range(num_images_to_show):
        # Preparar a imagem com shape [1, 28, 28, 1]
        img_input = np.expand_dims(x_test[i], axis=0)
        real_label = y_test[i]
        
        # Calcular a Saliência
        saliency_map, predicted_label = compute_saliency_map(model, img_input)
        
        # Remover as dimensões extra para desenhar [28, 28]
        img_2d = img_input[0, :, :, 0]
        saliency_2d = saliency_map[:, :, 0]

        # --- DESENHAR ---
        # A) Imagem Original
        ax1 = plt.subplot(num_images_to_show, 3, i * 3 + 1)
        ax1.imshow(img_2d, cmap='gray')
        ax1.set_title(f"Original (Real: {real_label})")
        ax1.axis('off')

        # B) Mapa de Calor Puro (Saliência)
        ax2 = plt.subplot(num_images_to_show, 3, i * 3 + 2)
        ax2.imshow(saliency_2d, cmap='hot')
        ax2.set_title(f"Foco da CNN (Previsto: {predicted_label})")
        ax2.axis('off')

        # C) Sobreposição (Onde a rede olhou na imagem)
        ax3 = plt.subplot(num_images_to_show, 3, i * 3 + 3)
        ax3.imshow(img_2d, cmap='gray')
        # Colocar o mapa de calor por cima com transparência (alpha=0.6)
        ax3.imshow(saliency_2d, cmap='hot', alpha=0.6)
        ax3.set_title("Sobreposição XAI")
        ax3.axis('off')

    output_path = os.path.join("outputs", "mapa_saliencia.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    
    logger.info("==================================================")
    logger.info(f"Radiografia gerada com sucesso: {output_path}")
    logger.info("==================================================")

if __name__ == "__main__":
    main()