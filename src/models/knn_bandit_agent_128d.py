import os
import logging
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)

class KNNBanditAgent128D:
    """
    Agente de Reinforcement Learning baseado em k-Nearest Neighbors para 128D.
    
    Arquitetura:
        - Recebe latent_features (128D).
        - Aplica PCA para reduzir a dimensionalidade (ex: 48D) para melhor performance do k-NN,
          ou usa 128D diretamente se n_components=None.
        - Memória Episódica: Base de dados de experiências passadas.
        - Busca k-NN: Encontra k estados mais parecidos.
        - Decisão: Score ponderado por distância (SUM).
    """

    def __init__(self, k: int = 30, n_actions: int = 10, use_pca: bool = True, pca_components: int = 48):
        self.k = k
        self.n_actions = n_actions
        self.use_pca = use_pca
        self.pca_components = pca_components

        self._states: list = []    
        self._actions: list = []   
        self._rewards: list = []   

        self._nn_index = None
        self._pca = None
        self._states_array = None

    @property
    def memory_size(self) -> int:
        return len(self._states)

    def add_experience(self, state_128d: np.ndarray, action: int, reward: float) -> None:
        self._states.append(np.asarray(state_128d, dtype=np.float32).flatten())
        self._actions.append(int(action))
        self._rewards.append(float(reward))

    def add_experience_batch(self, states: np.ndarray, actions: np.ndarray, rewards: np.ndarray) -> None:
        for i in range(len(states)):
            self._states.append(np.asarray(states[i], dtype=np.float32).flatten())
            self._actions.append(int(actions[i]))
            self._rewards.append(float(rewards[i]))

    def build_index(self) -> None:
        if self.memory_size == 0:
            raise ValueError("A memória está vazia!")

        self._states_array = np.array(self._states, dtype=np.float32)

        if self.use_pca:
            self._pca = PCA(n_components=self.pca_components)
            logger.info(f"A ajustar PCA para {self.pca_components} componentes...")
            self._states_array = self._pca.fit_transform(self._states_array)
        
        k_efetivo = min(self.k, self.memory_size)

        # Em dimensões moderadas (30-50), KDTree ou auto funciona bem.
        self._nn_index = NearestNeighbors(
            n_neighbors=k_efetivo,
            algorithm='auto',
            metric='euclidean',
            n_jobs=-1  # usar múltiplos cores
        )
        self._nn_index.fit(self._states_array)
        
        logger.info(f"Índice k-NN (128D) construído com {self.memory_size} experiências (k={k_efetivo}). PCA={self.use_pca}")

    def get_expected_rewards(self, state_128d: np.ndarray) -> np.ndarray:
        if self._nn_index is None:
            raise RuntimeError("O índice k-NN não foi construído!")

        query = np.asarray(state_128d, dtype=np.float32).reshape(1, -1)
        if self.use_pca and self._pca is not None:
            query = self._pca.transform(query)

        distances, indices = self._nn_index.kneighbors(query)
        neighbor_indices = indices[0]
        neighbor_distances = distances[0]

        weights = 1.0 / (neighbor_distances + 1e-8)

        neighbor_actions = np.array([self._actions[i] for i in neighbor_indices])
        neighbor_rewards = np.array([self._rewards[i] for i in neighbor_indices])

        expected_rewards = np.zeros(self.n_actions, dtype=np.float32)
        for action in range(self.n_actions):
            mask = neighbor_actions == action
            if np.any(mask):
                expected_rewards[action] = np.sum(neighbor_rewards[mask] * weights[mask])

        return expected_rewards

    def get_expected_rewards_batch(self, states: np.ndarray) -> np.ndarray:
        if self._nn_index is None:
            raise RuntimeError("O índice k-NN não foi construído!")

        states = np.asarray(states, dtype=np.float32)
        if states.ndim == 1:
            states = states.reshape(1, -1)

        if self.use_pca and self._pca is not None:
            states = self._pca.transform(states)

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

    def get_action(self, state_128d: np.ndarray, epsilon: float = 0.0) -> int:
        if np.random.rand() < epsilon:
            return int(np.random.randint(0, self.n_actions))

        expected = self.get_expected_rewards(state_128d)
        return int(np.argmax(expected))

    def get_action_batch(self, states: np.ndarray, epsilon: float = 0.0) -> np.ndarray:
        n = len(states)
        expected = self.get_expected_rewards_batch(states)
        greedy_actions = np.argmax(expected, axis=1)

        explore_mask = np.random.rand(n) < epsilon
        random_actions = np.random.randint(0, self.n_actions, size=n)

        return np.where(explore_mask, random_actions, greedy_actions)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        # Guardar também os componentes do PCA se for usado
        save_dict = {
            "states": np.array(self._states, dtype=np.float32),
            "actions": np.array(self._actions, dtype=np.int32),
            "rewards": np.array(self._rewards, dtype=np.float32),
            "use_pca": np.array([self.use_pca]),
            "pca_components": np.array([self.pca_components])
        }
        if self.use_pca and self._pca is not None:
            save_dict["pca_mean"] = self._pca.mean_
            save_dict["pca_components_"] = self._pca.components_
            
        np.savez_compressed(path, **save_dict)
        logger.info(f"Memória episódica (128D) guardada em: {path} ({self.memory_size} experiências)")

    def load(self, path: str) -> None:
        data = np.load(path)
        self._states = list(data["states"])
        self._actions = list(data["actions"].astype(int))
        self._rewards = list(data["rewards"].astype(float))
        
        if "use_pca" in data:
            self.use_pca = bool(data["use_pca"][0])
            self.pca_components = int(data["pca_components"][0])
        else:
            self.use_pca = False
            
        if self.use_pca and "pca_mean" in data:
            self._pca = PCA(n_components=self.pca_components)
            self._pca.mean_ = data["pca_mean"]
            self._pca.components_ = data["pca_components_"]
            # To allow transform without fit
            self._pca.explained_variance_ = np.ones(self.pca_components) # dummy
        
        logger.info(f"Memória (128D) carregada de: {path} ({self.memory_size} experiências)")
        
        self.build_index()

    def get_memory_stats(self) -> dict:
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
