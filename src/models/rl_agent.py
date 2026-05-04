"""
[DEPRECADO] Módulo do Agente de Reinforcement Learning (Contextual Bandit).
Este agente atua como o "Especialista" que resolve os casos ambíguos rejeitados pela CNN.

AVISO: Este módulo está DEPRECADO. Foi substituído pelo KNNBanditAgent em:
    src/models/knn_bandit_agent.py
    
O novo agente usa k-Nearest Neighbors (Memória Episódica) em vez de uma MLP,
eliminando a necessidade de backpropagation e épocas de treino pesadas.

Este ficheiro é mantido apenas para referência e compatibilidade retroativa.
"""

import tensorflow as tf
import numpy as np
from src.scratch.layers import DenseLayer
from src.scratch.activations import relu

class QNetworkAgent(tf.Module):
    def __init__(self, state_dim: int = 128, action_dim: int = 10, name: str = "q_network"):
        super().__init__(name=name)
        
        # --- O CÉREBRO DO AGENTE ---
        # Não precisa de ser tão complexo como a CNN. Ele só precisa de mapear 
        # as 128 características latentes para as 10 ações possíveis.
        self.dense1 = DenseLayer(in_features=state_dim, out_features=64, name=f"{name}_fc1")
        self.dense2 = DenseLayer(in_features=64, out_features=action_dim, name=f"{name}_fc2")

    def __call__(self, state: tf.Tensor) -> tf.Tensor:
        """
        O Forward Pass do Agente.
        Recebe o estado (vetor de 128D) e devolve os Q-Values (10D).
        ATENÇÃO: Não usamos Softmax no final! Os outputs são números reais (lucro esperado).
        """
        x = self.dense1(state)
        x = relu(x)
        q_values = self.dense2(x) # Output: [batch_size, 10]
        
        return q_values

    def get_action(self, state: tf.Tensor, epsilon: float) -> int:
        """
        Política de Decisão Epsilon-Greedy.
        É aqui que o agente decide se joga pelo seguro ou se tenta a sorte.
        """
        # 1. Lançar os dados para ver se vamos Explorar ou Explotar
        if np.random.rand() < epsilon:
            # EXPLORAÇÃO (Exploration): Ignoramos o cérebro e escolhemos uma ação à sorte
            acao_escolhida = np.random.randint(0, 10)
        else:
            # EXPLOTAÇÃO (Exploitation): Perguntamos ao cérebro qual é a ação mais lucrativa
            q_values = self(state)
            acao_escolhida = int(tf.argmax(q_values[0]).numpy())
            
        return acao_escolhida