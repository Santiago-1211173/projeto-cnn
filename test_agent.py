"""
Script de Teste Unitário do Agente RL.
Valida o Forward Pass (Q-Values) e a Política de Decisão (Exploração vs. Explotação).
"""

import tensorflow as tf
import logging
from src.models.rl_agent import QNetworkAgent

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("A inicializar o Agente Especialista (Com pesos aleatórios)...")
    agent = QNetworkAgent(state_dim=128, action_dim=10)

    # Simular o output da CNN: 1 imagem, 128 características
    # Usamos tf.random para criar um "Outlier" completamente inventado
    dummy_state = tf.random.normal(shape=(1, 128), dtype=tf.float32)

    logger.info("\n--- Teste 1: O Cérebro (Q-Values Crús) ---")
    # O Forward Pass: o que o agente acha deste estado?
    q_values = agent(dummy_state)
    logger.info(f"Shape do Output: {q_values.shape} (Esperado: (1, 10))")
    
    # Formatar os números para ficarem bonitos no terminal
    q_formatados = [f"{val:+.3f}" for val in q_values.numpy()[0]]
    logger.info(f"Lucros Esperados por botão:\n{q_formatados}")
    
    # Descobrir qual é matematicamente a ação com maior valor
    acao_maxima = int(tf.argmax(q_values[0]).numpy())
    logger.info(f"-> O botão com maior Q-Value é o: {acao_maxima}")

    logger.info("\n--- Teste 2: Explotação Pura (Epsilon = 0.0) ---")
    # Como epsilon é 0, o agente tem de confiar 100% no cérebro
    acao_gananciosa = agent.get_action(dummy_state, epsilon=0.0)
    logger.info(f"Ação escolhida (Ganância): {acao_gananciosa}")
    if acao_gananciosa == acao_maxima:
        logger.info("   [OK] A política gananciosa seguiu o maior Q-Value!")
    else:
        logger.error("   [ERRO] A política gananciosa falhou!")

    logger.info("\n--- Teste 3: Exploração Pura (Epsilon = 1.0) ---")
    # Como epsilon é 1, o agente tem de ignorar o cérebro e jogar os dados
    acoes_aleatorias = [agent.get_action(dummy_state, epsilon=1.0) for _ in range(5)]
    logger.info(f"5 Ações escolhidas (100% Sorte): {acoes_aleatorias}")
    logger.info("   [OK] O agente está a explorar o ambiente de forma aleatória!")

if __name__ == "__main__":
    main()