"""
Motor de Treino do Agente RL (Contextual Bandit).
Treina o Agente Especialista a resolver imagens corrompidas 
usando apenas recompensas (+1/-1) e a equação de Bellman simplificada.
"""

import os
import logging
import numpy as np
import tensorflow as tf

from src.models.custom_cnn import RawModel
from src.models.rl_agent import QNetworkAgent
from src.data.loader import load_mnist_raw

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def adicionar_ruido_batch(imagens: np.ndarray, intensidade: float = 0.6) -> np.ndarray:
    """Injeta estática num batch inteiro de imagens para forçar a triagem do Árbitro."""
    ruido = np.random.normal(loc=0.0, scale=intensidade, size=imagens.shape)
    return np.clip(imagens + ruido, 0., 1.)

def main():
    # 1. Configurar Hardware
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    # 2. Carregar o Extrator (CNN) - O Cérebro visual (Congelado, não treina mais)
    logger.info("A carregar a CNN (Extrator de Features)...")
    cnn = RawModel()
    ckpt = tf.train.Checkpoint(model=cnn)
    latest_ckpt = tf.train.latest_checkpoint(os.path.join("outputs", "checkpoints"))
    ckpt.restore(latest_ckpt).expect_partial()

    # 3. Inicializar o Agente RL e o Otimizador
    logger.info("A instanciar o Agente Especialista RL...")
    agent = QNetworkAgent()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    # 4. Carregar Dados de Treino
    logger.info("A preparar o Ambiente de Treino...")
    x_train, y_train = load_mnist_raw(os.path.join("data", "MNIST", "raw"), kind='train')
    x_train = x_train.astype(np.float32) / 255.0

    # Hiperparâmetros do RL
    epochs = 10
    batch_size = 64
    num_batches = len(x_train) // batch_size
    
    # Epsilon (Exploração vs Explotação)
    epsilon_inicial = 1.0  # Começa 100% à sorte
    epsilon_final = 0.05   # Acaba a explorar apenas 5% das vezes
    epsilon = epsilon_inicial
    decaimento_epsilon = (epsilon_inicial - epsilon_final) / (epochs * num_batches)

    logger.info("\n==================================================")
    logger.info("INÍCIO DO TREINO POR REINFORCEMENT LEARNING")
    logger.info("==================================================")

    for epoch in range(epochs):
        recompensa_media_epoca = 0
        
        # Baralhar dados no início de cada época
        indices = np.random.permutation(len(x_train))
        x_train_shuffled = x_train[indices]
        y_train_shuffled = y_train[indices]

        for b in range(num_batches):
            # A. Preparar o Batch "Difícil"
            batch_x_limpo = x_train_shuffled[b * batch_size : (b + 1) * batch_size]
            batch_y_real = y_train_shuffled[b * batch_size : (b + 1) * batch_size]
            batch_x_ruido = adicionar_ruido_batch(batch_x_limpo)
            
            # FIX APLICADO: O loader original já devolve (Batch, 28, 28, 1). 
            # Basta converter diretamente para Tensor sem expandir as dimensões!
            batch_tensor = tf.convert_to_tensor(batch_x_ruido, dtype=tf.float32)

            # B. Extrair o Estado (128D) usando a CNN (A CNN NÃO é atualizada aqui)
            estados = cnn(batch_tensor)["latent_features"] # Shape: [64, 128]

            # C. Política Epsilon-Greedy Vectorizada (Para ser rápido na GPU)
            # Lançamos os dados para as 64 imagens ao mesmo tempo
            mascara_exploracao = np.random.rand(batch_size) < epsilon
            acoes_aleatorias = np.random.randint(0, 10, size=batch_size)
            acoes_gananciosas = tf.argmax(agent(estados), axis=1).numpy()
            
            # O agente decide a sua ação (mistura de sorte e ganância)
            acoes = np.where(mascara_exploracao, acoes_aleatorias, acoes_gananciosas)

            # D. O Ambiente avalia as ações e distribui recompensas
            recompensas = np.where(acoes == batch_y_real, 1.0, -1.0)
            recompensa_media_epoca += np.mean(recompensas)

            # E. A MAGIA DO RL: Atualizar o Cérebro do Agente (Loss = (Q - R)^2)
            with tf.GradientTape() as tape:
                # 1. O agente prevê os lucros para todos os 10 botões
                q_values_todos = agent(estados)
                
                # 2. Nós SÓ queremos saber o Q-Value da ação que ele efetivamente tomou!
                # Usamos tf.gather_nd para pescar apenas os valores dos botões carregados.
                indices_gather = tf.stack([tf.range(batch_size), tf.convert_to_tensor(acoes, dtype=tf.int32)], axis=1)
                q_values_tomados = tf.gather_nd(q_values_todos, indices_gather)
                
                # 3. Calcular a Função de Custo (O Erro de Surpresa)
                recompensas_tensor = tf.convert_to_tensor(recompensas, dtype=tf.float32)
                loss = tf.reduce_mean(tf.square(q_values_tomados - recompensas_tensor))

            # F. Aplicar Gradientes (Backpropagation apenas no Agente)
            gradientes = tape.gradient(loss, agent.trainable_variables)
            optimizer.apply_gradients(zip(gradientes, agent.trainable_variables))

            # G. Decair Epsilon (O agente fica mais confiante à medida que aprende)
            epsilon = max(epsilon_final, epsilon - decaimento_epsilon)

            # H. Logs a cada 200 batches
            if b % 200 == 0:
                acc_simulada = np.mean(recompensas == 1.0) * 100
                logger.info(f"Época {epoch+1} | Batch {b:03d} | Epsilon: {epsilon:.2f} | Loss RL: {loss:.3f} | Acc Batch: {acc_simulada:.1f}%")

        # Fim da época
        logger.info(f"-> RESUMO DA ÉPOCA {epoch+1}: Recompensa Média: {recompensa_media_epoca/num_batches:.3f}\n")

    # Guardar os pesos do Agente
    caminho_agente = os.path.join("outputs", "rl_agent_weights")
    ckpt_agent = tf.train.Checkpoint(model=agent)
    ckpt_agent.save(caminho_agente)
    logger.info(f"Treino concluído! Cérebro do Especialista guardado em: {caminho_agente}")

if __name__ == "__main__":
    main()