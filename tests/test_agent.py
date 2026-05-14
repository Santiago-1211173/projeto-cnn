"""
Script de Teste Unitário do Agente k-NN Bandit.
Valida a memória episódica, a busca k-NN, e a política de decisão.
"""

import numpy as np
import logging
from src.models.knn_bandit_agent import KNNBanditAgent

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def main():
    logger.info("A inicializar o Agente k-NN Bandit (Memória vazia)...")
    agent = KNNBanditAgent(k=5, n_actions=10)

    # --- Teste 1: Popular a Memória ---
    logger.info("\n--- Teste 1: Popular a Memória Episódica ---")
    
    # Simular 100 experiências onde o dígito "3" tem um padrão reconhecível
    np.random.seed(42)
    for _ in range(100):
        # Criar um estado 10D que parece com o dígito 3 (prob alta na posição 3)
        state = np.random.dirichlet(np.ones(10))
        true_digit = np.argmax(state)
        action = true_digit  # Ação correta
        reward = 1.0
        agent.add_experience(state, action, reward)
        
        # Também adicionar experiências erradas
        wrong_action = (true_digit + np.random.randint(1, 10)) % 10
        agent.add_experience(state, wrong_action, -1.0)
    
    logger.info(f"Memória populada com {agent.memory_size} experiências")
    assert agent.memory_size == 200, f"Esperado 200, obtido {agent.memory_size}"
    logger.info("   [OK] Tamanho da memória correto!")

    # --- Teste 2: Construir o Índice ---
    logger.info("\n--- Teste 2: Construir o Índice k-NN ---")
    agent.build_index()
    logger.info("   [OK] Índice construído com sucesso!")

    # --- Teste 3: Recompensas Esperadas ---
    logger.info("\n--- Teste 3: Consultar Recompensas Esperadas ---")
    
    # Criar um estado que se parece com o dígito 7 (prob alta na posição 7)
    test_state = np.zeros(10, dtype=np.float32)
    test_state[7] = 0.8
    test_state[1] = 0.1
    test_state[9] = 0.1
    
    expected_rewards = agent.get_expected_rewards(test_state)
    logger.info(f"Recompensas esperadas para cada ação:")
    for i, r in enumerate(expected_rewards):
        marker = " <-- MELHOR" if i == np.argmax(expected_rewards) else ""
        logger.info(f"  Ação {i}: {r:+.3f}{marker}")
    
    logger.info(f"-> O agente escolheria a ação: {np.argmax(expected_rewards)}")
    logger.info("   [OK] Recompensas esperadas calculadas!")

    # --- Teste 4: Explotação Pura ---
    logger.info("\n--- Teste 4: Explotação Pura (Epsilon = 0.0) ---")
    action = agent.get_action(test_state, epsilon=0.0)
    logger.info(f"Ação escolhida (Ganância): {action}")
    assert action == np.argmax(expected_rewards), "A ação deveria ser o argmax!"
    logger.info("   [OK] A política gananciosa seguiu a maior recompensa esperada!")

    # --- Teste 5: Exploração Pura ---
    logger.info("\n--- Teste 5: Exploração Pura (Epsilon = 1.0) ---")
    acoes = [agent.get_action(test_state, epsilon=1.0) for _ in range(20)]
    logger.info(f"20 Ações aleatórias: {acoes}")
    assert len(set(acoes)) > 1, "Com exploração pura deveria haver diversidade!"
    logger.info("   [OK] O agente explora de forma aleatória!")

    # --- Teste 6: Guardar e Carregar ---
    logger.info("\n--- Teste 6: Persistência (Save/Load) ---")
    import os
    test_path = os.path.join("outputs", "test_knn_memory.npz")
    agent.save(test_path)
    
    agent2 = KNNBanditAgent(k=5, n_actions=10)
    agent2.load(test_path)
    
    assert agent2.memory_size == agent.memory_size, "Memórias deveriam ter o mesmo tamanho!"
    action2 = agent2.get_action(test_state, epsilon=0.0)
    assert action2 == action, "Ações deveriam ser iguais após carregar!"
    logger.info(f"   [OK] Memória guardada e restaurada ({agent2.memory_size} experiências)")
    
    # Limpar ficheiro de teste
    os.remove(test_path)

    # --- Teste 7: Estatísticas ---
    logger.info("\n--- Teste 7: Estatísticas da Memória ---")
    stats = agent.get_memory_stats()
    logger.info(f"  Tamanho: {stats['size']}")
    logger.info(f"  Recomp. Média: {stats['reward_mean']:.3f}")
    logger.info(f"  % Positivas: {stats['reward_positive_pct']:.1f}%")
    logger.info("   [OK] Estatísticas calculadas!")

    logger.info("\n==================================================")
    logger.info("TODOS OS TESTES PASSARAM COM SUCESSO!")
    logger.info("==================================================")


if __name__ == "__main__":
    main()