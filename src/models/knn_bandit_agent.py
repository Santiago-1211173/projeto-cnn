"""
Módulo do Agente k-NN Bandit (Memória Episódica).
Substitui o antigo QNetworkAgent (MLP) por uma abordagem não-paramétrica.

Em vez de treinar uma rede neuronal, este agente armazena triplos
(estado_10D, ação, recompensa) e usa a busca por vizinhos mais próximos
para estimar a recompensa esperada de cada ação possível.

NÃO usa Deep Learning. O "cérebro" é uma base de dados + busca k-NN.
"""

import os
import logging
import numpy as np
from sklearn.neighbors import NearestNeighbors

logger = logging.getLogger(__name__)


class KNNBanditAgent:
    """
    Agente de Reinforcement Learning baseado em k-Nearest Neighbors.
    
    Arquitetura:
        - Memória Episódica: Base de dados de experiências passadas.
        - Busca k-NN: Para cada novo estado, encontra os k estados mais 
          parecidos que já viu no passado.
        - Decisão: Calcula o score ponderado por distância para cada uma das 
          10 ações possíveis, com base no histórico dos vizinhos.
          Usa SUM (voto ponderado) em vez de MEAN para que mais vizinhos
          a votar numa ação = sinal mais forte.
    
    Parâmetros:
        k (int): Número de vizinhos a consultar (default: 30).
        n_actions (int): Número de ações possíveis (10 dígitos).
    """

    def __init__(self, k: int = 30, n_actions: int = 10):
        self.k = k
        self.n_actions = n_actions

        # --- A MEMÓRIA EPISÓDICA (O "Cérebro" do Agente) ---
        # Listas que crescem à medida que o agente acumula experiência.
        self._states: list = []    # Cada entrada: array de 10D (probabilidades CNN)
        self._actions: list = []   # Cada entrada: int (0-9, a ação tomada)
        self._rewards: list = []   # Cada entrada: float (+1.0 ou -1.0)

        # O Índice de Busca (Construído após popular a memória)
        self._nn_index: NearestNeighbors | None = None
        self._states_array: np.ndarray | None = None  # Cache numpy para performance

    @property
    def memory_size(self) -> int:
        """Retorna o número total de experiências armazenadas."""
        return len(self._states)

    # =========================================================================
    # FASE 1: POPULAR A MEMÓRIA
    # =========================================================================

    def add_experience(self, state_10d: np.ndarray, action: int, reward: float) -> None:
        """
        Armazena uma nova experiência na memória episódica.
        
        Args:
            state_10d: Array 10D com as probabilidades/logits da CNN.
            action: A ação tomada (0-9).
            reward: A recompensa recebida (+1.0 ou -1.0).
        """
        self._states.append(np.asarray(state_10d, dtype=np.float32).flatten())
        self._actions.append(int(action))
        self._rewards.append(float(reward))

    def add_experience_batch(self, states: np.ndarray, actions: np.ndarray, rewards: np.ndarray) -> None:
        """
        Armazena um batch inteiro de experiências de uma só vez (mais eficiente).
        
        Args:
            states: Array (N, 10) com os estados.
            actions: Array (N,) com as ações.
            rewards: Array (N,) com as recompensas.
        """
        for i in range(len(states)):
            self._states.append(np.asarray(states[i], dtype=np.float32).flatten())
            self._actions.append(int(actions[i]))
            self._rewards.append(float(rewards[i]))

    # =========================================================================
    # FASE 2: CONSTRUIR O ÍNDICE DE BUSCA
    # =========================================================================

    def build_index(self) -> None:
        """
        Constrói o índice k-NN sobre toda a memória acumulada.
        Deve ser chamado DEPOIS de popular a memória e ANTES de fazer inferência.
        
        Usa Ball Tree como algoritmo de busca — eficiente para espaços de 10D.
        """
        if self.memory_size == 0:
            raise ValueError("A memória está vazia! Popule-a antes de construir o índice.")

        self._states_array = np.array(self._states, dtype=np.float32)
        
        # O k efetivo nunca pode exceder o tamanho da memória
        k_efetivo = min(self.k, self.memory_size)

        self._nn_index = NearestNeighbors(
            n_neighbors=k_efetivo,
            algorithm='ball_tree',
            metric='euclidean'
        )
        self._nn_index.fit(self._states_array)
        
        logger.info(f"Índice k-NN construído com {self.memory_size} experiências (k={k_efetivo}).")

    # =========================================================================
    # FASE 3: INFERÊNCIA (CONSULTAR A MEMÓRIA)
    # =========================================================================

    def get_expected_rewards(self, state_10d: np.ndarray) -> np.ndarray:
        """
        Para um dado estado 10D, consulta os k vizinhos mais próximos e calcula
        o score ponderado por distância para cada uma das 10 ações possíveis.
        
        Usa SUM ponderado (não MEAN) para que:
        - Mais vizinhos a votar numa ação = score mais alto
        - Vizinhos mais próximos têm mais peso (1/distância)
        
        Retorna:
            Array de 10 valores: score ponderado por ação.
        """
        if self._nn_index is None:
            raise RuntimeError("O índice k-NN não foi construído! Chame build_index() primeiro.")

        query = np.asarray(state_10d, dtype=np.float32).reshape(1, -1)
        distances, indices = self._nn_index.kneighbors(query)
        neighbor_indices = indices[0]
        neighbor_distances = distances[0]

        # Pesos inversamente proporcionais à distância (vizinhos mais perto = mais peso)
        # Adicionar epsilon para evitar divisão por zero
        weights = 1.0 / (neighbor_distances + 1e-8)

        # Recolher as ações e recompensas dos vizinhos
        neighbor_actions = np.array([self._actions[i] for i in neighbor_indices])
        neighbor_rewards = np.array([self._rewards[i] for i in neighbor_indices])

        # Calcular o score ponderado para cada ação (SUM, não MEAN)
        expected_rewards = np.zeros(self.n_actions, dtype=np.float32)
        for action in range(self.n_actions):
            mask = neighbor_actions == action
            if np.any(mask):
                # Score = soma dos (reward * peso) dos vizinhos que tomaram esta ação
                expected_rewards[action] = np.sum(neighbor_rewards[mask] * weights[mask])

        return expected_rewards

    def get_expected_rewards_batch(self, states: np.ndarray) -> np.ndarray:
        """
        Versão vetorizada para processar um batch inteiro de estados.
        Usa SUM ponderado por distância (consistente com get_expected_rewards).
        """
        if self._nn_index is None:
            raise RuntimeError("O índice k-NN não foi construído! Chame build_index() primeiro.")

        states = np.asarray(states, dtype=np.float32)
        if states.ndim == 1:
            states = states.reshape(1, -1)

        all_distances, all_indices = self._nn_index.kneighbors(states)

        actions_array = np.array(self._actions)
        rewards_array = np.array(self._rewards)

        result = np.zeros((len(states), self.n_actions), dtype=np.float32)

        for i, (neighbor_indices, neighbor_distances) in enumerate(zip(all_indices, all_distances)):
            weights = 1.0 / (neighbor_distances + 1e-8)
            neighbor_actions = actions_array[neighbor_indices]
            neighbor_rewards = rewards_array[neighbor_indices]

            for action in range(self.n_actions):
                mask = neighbor_actions == action
                if np.any(mask):
                    result[i, action] = np.sum(neighbor_rewards[mask] * weights[mask])

        return result

    def get_action(self, state_10d: np.ndarray, epsilon: float = 0.0) -> int:
        """
        Política Epsilon-Greedy.
        
        Args:
            state_10d: Array 10D com as probabilidades da CNN.
            epsilon: Probabilidade de exploração aleatória (0.0 = ganância pura).
            
        Retorna:
            A ação escolhida (0-9).
        """
        # Exploração: Ação aleatória
        if np.random.rand() < epsilon:
            return int(np.random.randint(0, self.n_actions))

        # Explotação: Escolher a ação com maior recompensa esperada
        expected = self.get_expected_rewards(state_10d)
        return int(np.argmax(expected))

    def get_action_batch(self, states: np.ndarray, epsilon: float = 0.0) -> np.ndarray:
        """
        Política Epsilon-Greedy vetorizada para um batch de estados.
        
        Args:
            states: Array (N, 10) com os estados.
            epsilon: Probabilidade de exploração.
            
        Retorna:
            Array (N,) com as ações escolhidas.
        """
        n = len(states)
        
        # Calcular as recompensas esperadas para todo o batch
        expected = self.get_expected_rewards_batch(states)
        greedy_actions = np.argmax(expected, axis=1)

        # Aplicar a máscara de exploração
        explore_mask = np.random.rand(n) < epsilon
        random_actions = np.random.randint(0, self.n_actions, size=n)

        return np.where(explore_mask, random_actions, greedy_actions)

    # =========================================================================
    # PERSISTÊNCIA (GUARDAR/CARREGAR A MEMÓRIA)
    # =========================================================================

    def save(self, path: str) -> None:
        """
        Guarda a memória episódica no disco como ficheiro .npz.
        
        O ficheiro contém 3 arrays:
            - states: (N, 10) float32
            - actions: (N,) int
            - rewards: (N,) float32
        """
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        np.savez_compressed(
            path,
            states=np.array(self._states, dtype=np.float32),
            actions=np.array(self._actions, dtype=np.int32),
            rewards=np.array(self._rewards, dtype=np.float32)
        )
        logger.info(f"Memória episódica guardada em: {path} ({self.memory_size} experiências)")

    def load(self, path: str) -> None:
        """
        Carrega a memória episódica do disco e reconstrói o índice k-NN.
        """
        data = np.load(path)
        self._states = list(data["states"])
        self._actions = list(data["actions"].astype(int))
        self._rewards = list(data["rewards"].astype(float))
        
        logger.info(f"Memória carregada de: {path} ({self.memory_size} experiências)")
        
        # Reconstruir o índice automaticamente
        self.build_index()

    # =========================================================================
    # DIAGNÓSTICO
    # =========================================================================

    def get_memory_stats(self) -> dict:
        """Retorna estatísticas sobre a memória para diagnóstico."""
        if self.memory_size == 0:
            return {"size": 0}
        
        rewards = np.array(self._rewards)
        actions = np.array(self._actions)
        
        return {
            "size": self.memory_size,
            "reward_mean": float(np.mean(rewards)),
            "reward_positive_pct": float(np.mean(rewards > 0) * 100),
            "actions_distribution": {
                int(a): int(np.sum(actions == a)) for a in range(self.n_actions)
            }
        }
