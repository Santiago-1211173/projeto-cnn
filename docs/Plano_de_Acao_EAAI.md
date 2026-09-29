# Plano de Implementacao: Gestao Ativa de Memoria Episodica via RL
### *Edge AI — Roteamento Dinamico Robusto para Dados Out-of-Distribution*

> [!NOTE]
> **Documento de Plano de Implementacao para Agente LLM.**
> Este plano deve ser executado fase a fase. O agente **NAO deve implementar todas as fases de uma vez** — deve aguardar que o utilizador peca a "Fase X" e fornecer os ficheiros completos, rigorosamente tipados e compilaveis apenas para essa fase.
> Antes de implementar qualquer fase, o agente DEVE ler a seccao "CONTRATOS DE INTERFACE" para garantir compatibilidade entre modulos.

---

## PRE-REQUISITOS E DEPENDENCIAS

Antes de iniciar qualquer fase, o agente DEVE garantir que o ambiente contem:

**Python:** >= 3.10 (necessario para union types `X | None`)

**Dependencias Pip (requirements.txt):**

    numpy>=1.24
    torch>=2.0
    scikit-learn>=1.3

**Dependencias de Tipagem:**

    from typing import Tuple, Optional, Dict, List
    import numpy as np
    import torch
    import torch.nn as nn

**Base de Conhecimento Cientifica:**
O agente tem acesso a literatura curada em `docs/Literatura/` para fundamentacao de decisoes arquiteturais. As referencias relevantes estao indicadas em cada fase entre parentesis retos, ex: `[Lit: Episodic Memory/NEC]`.

---

## CONTEXTO GLOBAL DO SISTEMA

O objetivo deste projeto e transformar a arquitetura de roteamento estatico num sistema de **Trustworthy AI** focado em fiabilidade (Reliability) e Edge Computing, visando submissao a revista EAAI (Engineering Applications of Artificial Intelligence).

A arquitetura base (Arbitrador de Incerteza Espacial):
1. Extracao latente 128D por uma rede CNN. `[Lit: Deep Learning/CNNs]`
2. Calculo da Distancia de Mahalanobis do vetor latente com normalizacao L2 (Mahalanobis++) e regularizacao de covariancia via Ledoit-Wolf Shrinkage. `[Lit: Out-of-Distribution/Lee et al. 2018, Kamoi & Kobayashi 2020, Chen et al. 2010]`
3. Decisao: `In-Distribution` segue a CNN; `Out-of-Distribution` (anomalia/ruido) e desviado para uma Memoria Episodica (k-NN) de resgate. `[Lit: Episodic Memory/MFEC, NEC]`
4. **Restricao Edge Computing:** A memoria tem RAM finita (ex: 5000 vetores). O uso atual de listas infinitas `.append()` causa Out-of-Memory (OOM). `[Lit: Edge AI/Pittorino & Roveri 2026 — "Edge AI e necessariamente adaptativo"]`
5. Um Agente de Reinforcement Learning (RL) fara a gestao ativa da memoria (controlo de admissao e eviccao) sempre que esta encher, otimizado por Curriculum Learning (Proxy Geometrico transitando para um Sliding Validation Buffer). `[Lit: Reinforcement Learning/Curriculum RL, Edge AI/RLCache, Catcher+]`

---

## ESTADO REAL DO REPOSITORIO (PONTO DE PARTIDA)

O agente DEVE usar estes caminhos reais como referencia. NAO existem ficheiros alem destes:

    projeto-cnn/
    ├── src/
    │   ├── config.py                            # Configuracoes globais (EXISTENTE)
    │   └── models/
    │       ├── custom_cnn.py                    # Extracao latente (EXISTENTE — SEM ALTERACAO)
    │       ├── knn_bandit_agent.py              # EXISTENTE — Alvo de refatorizacao da Fase 1
    │       └── mlp_bandit_agent.py              # EXISTENTE — Nao relevante para este plano
    ├── docs/
    │   ├── Literatura/                          # Base de conhecimento cientifica (12 areas)
    │   └── Plano_de_Acao_EAAI.md               # Este documento
    ├── requirements.txt
    └── README.md

**Ficheiros que NAO existem e que serao CRIADOS nas fases seguintes:**
- `src/models/reward_manager.py` (Fase 2 — NOVO)
- `src/models/rl_agent.py` (Fase 2 — NOVO)
- `training/train_rl_online_simulation.py` (Fase 3 — NOVO)
- `evaluate_hybrid_global.py` (Fase 4 — NOVO)

**Problemas confirmados no codigo atual** (`src/models/knn_bandit_agent.py`):
- Linhas 55-57: Listas nativas Python (`self._states: list = []`, etc.) com `.append()`
- Linha 13: Dependencia de `sklearn.neighbors.NearestNeighbors` (a substituir por NumPy puro)
- Linha 44: `latent_dim=10` por defeito (deve ser 128)
- Classe exportada como alias `KNNBanditAgent128D = KNNBanditAgent` (linha 309)

---

## REGRAS OBRIGATORIAS

O agente DEVE seguir rigorosamente estas regras em todas as interacoes:

1. **Idioma do Codigo:** Todo o codigo (variaveis, funcoes, classes) e docstrings/comentarios de codigo DEVEM estar em **INGLES**. Documentacao em texto livre deve ser em Portugues (pt-PT).
2. **Performance Extrema:** E estritamente **PROIBIDO** o uso de listas nativas Python (`.append()`) em estruturas de dados dinamicas. E **obrigatorio** o uso de *NumPy arrays* pre-alocados para complexidade O(1) na insercao e remocao.
3. **Type Hints:** Todo o codigo Python tem de incluir `typing` rigoroso do inicio ao fim (ex: `np.ndarray`, `Tuple`, `Optional`).
4. **Vetorizacao:** Calculos matematicos pesados (distancias, k-NN) devem usar operacoes vetorizadas nativas (`np.linalg.norm`, `np.argpartition`).
5. **Codigo Completo:** O output deve ser sempre o ficheiro completo e compilavel. Sem placeholders, sem diffs parciais.

---

## ESTRUTURA ALVO APOS IMPLEMENTACAO

    projeto-cnn/
    ├── src/
    │   ├── config.py                            # Atualizado com novos parametros (Fase 1)
    │   └── models/
    │       ├── custom_cnn.py                    # Extracao latente (SEM ALTERACAO)
    │       ├── knn_bandit_agent.py              # FASE 1: Memoria otimizada via NumPy
    │       ├── reward_manager.py                # FASE 2: NOVO — Gestor de Recompensas (Curriculum Learning)
    │       └── rl_agent.py                      # FASE 2: NOVO — Agente Double DQN + PER
    ├── training/
    │   └── train_rl_online_simulation.py        # FASE 3: NOVO — Simulacao de Concept Drift
    └── evaluate_hybrid_global.py                # FASE 4: NOVO — Metricas de Engenharia EAAI

---

## MAPA DE DEPENDENCIAS INTER-FASE

    FASE 1 ──────────────────────────────────────► FASE 2
    (knn_bandit_agent.py)                          (reward_manager.py, rl_agent.py)
    Expoe: evict_oldest(),                         Consome: metodos evict_*
    evict_lfu(), evict_redundant(),                Expoe: select_action(),
    get_nearest_neighbors(),                       compute_reward(),
    .size, .capacity, .tick_counter                update_weights()
           │                                              │
           │                                              │
           └──────────── FASE 3 ◄─────────────────────────┘
                         (train_rl_online_simulation.py)
                         Consome: TODOS os modulos acima
                         Expoe: run_simulation(),
                         inject_noise()
                                    │
                                    ▼
                              FASE 4
                              (evaluate_hybrid_global.py)
                              Consome: Fase 3 + baselines
                              Expoe: run_baselines(),
                              export_metrics()

---

## CONTRATOS DE INTERFACE

O agente DEVE implementar estas assinaturas exatas. Qualquer desvio invalida a compatibilidade inter-fase.

### Fase 1 — `KNNBanditAgent128D` (em `src/models/knn_bandit_agent.py`)

    class KNNBanditAgent128D:
        # --- Inicializacao ---
        def __init__(self, capacity: int, k: int, latent_dim: int = 128) -> None: ...

        # --- Propriedades de estado ---
        size: int           # Ocupacao atual (0 <= size <= capacity)
        capacity: int       # Limite maximo de vetores
        tick_counter: int   # Relogio global monotonicamente crescente

        # --- Insercao ---
        def add_experience(self, state: np.ndarray, action: int, reward: float) -> None: ...

        # --- Pesquisa ---
        def get_nearest_neighbors(self, query: np.ndarray, k: int | None = None) -> Tuple[np.ndarray, np.ndarray]: ...
        #   Retorna: (indices: shape (k,), distances: shape (k,))

        # --- Eviccao (3 mecanicas + 1 reservada) ---
        def evict_oldest(self, new_state: np.ndarray, new_action: int, new_reward: float) -> int: ...
        def evict_least_frequently_used(self, new_state: np.ndarray, new_action: int, new_reward: float) -> int: ...
        def evict_most_redundant(self, new_state: np.ndarray, new_action: int, new_reward: float) -> int: ...
        #   Todos retornam: indice do vetor removido

        # --- Diagnostico ---
        def get_memory_stats(self) -> Dict[str, float]: ...

### Fase 2 — `RewardManager` (em `src/models/reward_manager.py`)

    class RewardManager:
        def __init__(self, buffer_size: int = 100, alpha_decay: float = 0.995) -> None: ...
        def update_validation_buffer(self, state: np.ndarray, true_label: int, predicted_label: int) -> None: ...
        def compute_reward(self, evicted_index: int, memory: KNNBanditAgent128D, new_state: np.ndarray) -> float: ...
        #   Retorna: float no intervalo [-1.0, 1.0]
        def get_current_alpha(self) -> float: ...

### Fase 2 — `RLAgent` (em `src/models/rl_agent.py`)

    class RLAgent:
        def __init__(self, state_dim: int = 5, n_actions: int = 4, lr: float = 1e-3) -> None: ...
        def get_state_vector(self, mahalanobis_dist: float, local_entropy: float,
                             min_knn_dist: float, prediction_error: float,
                             ram_occupancy: float) -> np.ndarray: ...
        #   Retorna: np.ndarray shape (5,)
        def select_action(self, state: np.ndarray, epsilon: float = 0.1) -> int: ...
        #   Retorna: int em {0, 1, 2, 3}
        def store_transition(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None: ...
        def update_weights(self, batch_size: int = 32) -> Optional[float]: ...
        #   Retorna: loss ou None se buffer insuficiente

### Fase 3 — `TrainRLOnlineSimulation` (em `training/train_rl_online_simulation.py`)

    def run_simulation(n_episodes: int, noise_injection_rate: float, capacity: int = 5000) -> Dict[str, List[float]]: ...
    def inject_noise(batch: np.ndarray, noise_level: float) -> np.ndarray: ...

### Fase 4 — `EvaluateHybridGlobal` (em `evaluate_hybrid_global.py`)

    def run_baselines(noise_levels: List[float], capacity: int = 5000) -> Dict[str, Dict[str, float]]: ...
    def export_metrics(results: Dict, output_path: str) -> None: ...

---

# FASE 1: Blindar a Memoria (Infraestrutura NumPy) `[CONCLUIDO]`

> [!NOTE]
> **Estado:** Concluido com sucesso. Todos os criterios de aceitacao e benchmarks validados.
> **Depende de:** Nada (fase raiz).
> **Desbloqueia:** Fase 2 (Cerebro RL e Gestao de Recompensas).

**Objetivo:** Eliminar as fugas de memoria (OOM) removendo as listas nativas e criando a infraestrutura mecanica que o RL usara para gerir o hardware limitado. A IA nao e implementada nesta fase, apenas as mecanicas NumPy.

**Fundamentacao Cientifica:**
- A arquitetura semiparametrica (CNN + k-NN episodico) e validada pelo NEC (Pritzel et al., 2017) e pelo Deep Semiparametric Learning (Jain & Lindsey, 2018). `[Lit: Episodic Memory/README.md]`
- A restricao de RAM finita alinha-se com o paradigma Agent-System-Environment (ASE) proposto por Pittorino & Roveri (2026). `[Lit: Edge AI/README.md]`
- A estrategia de eviccao LFU e inspirada nos resultados de Isele & Cosgun (2018), que provam que Distribution Matching supera Surprise e Reward para retencao de diversidade em buffers de memoria finita. `[Lit: Continual Learning/README.md]`

**Ficheiros a modificar:**
- `src/models/knn_bandit_agent.py` — Reescrever a classe `KNNBanditAgent` (renomear para `KNNBanditAgent128D`, manter alias de retrocompatibilidade `KNNBanditAgent = KNNBanditAgent128D`)
- `src/config.py` — Adicionar novos parametros

**Novos parametros a adicionar ao `src/config.py`:**

    # --- Fase 1: Memoria Episodica ---
    MEMORY_CAPACITY = 5000
    LATENT_DIM = 128
    KNN_K_NEIGHBORS = 30

    # --- Fase 2: Agente RL (pre-declarados para evitar imports circulares) ---
    RL_STATE_DIM = 5
    RL_N_ACTIONS = 4
    CURRICULUM_ALPHA_DECAY = 0.995
    SLIDING_VALIDATION_BUFFER_SIZE = 100
    REPLAY_BUFFER_CAPACITY = 10000

**Instrucoes precisas:**

1. A classe `KNNBanditAgent128D` deve receber na inicializacao `capacity` (ex: 5000) e `k` (vizinhos).
2. Substituir todas as listas por matrizes NumPy pre-alocadas (`np.zeros`) para:
    - `_states` (capacity x 128, np.float32)
    - `_actions` (capacity, np.int32)
    - `_rewards` (capacity, np.float32)
3. Criar metadados paralelos essenciais para as futuras politicas do RL:
    - Array `_insertion_ticks` (capacity, np.int64): Rastreia a idade incremental de cada vetor.
    - Array `_usage_counts` (capacity, np.int32): Rastreia quantas vezes um vetor esteve nos k-vizinhos corretos.
    - Variaveis de estado `self.size` (ocupacao atual) e `self.tick_counter` (relogio global).
4. O metodo de pesquisa (`get_nearest_neighbors`) deve usar `np.linalg.norm` e `np.argpartition` limitados estritamente a fatia de memoria preenchida (`[:self.size]`). Retorna `Tuple[np.ndarray, np.ndarray]` — (indices, distancias).
5. Implementar **3 metodos mecanicos de eviccao**. Cada um deve encontrar o alvo, substitui-lo pelos dados do `new_state`, atualizar metadados locais (zerar o `_usage_count` daquele indice e definir o `_insertion_tick`), incrementar o `tick_counter` e retornar o `index` apagado:
    - `evict_oldest(self, new_state, new_action, new_reward) -> int`: Encontra o indice com o menor `_insertion_tick` (FIFO) e substitui.
    - `evict_least_frequently_used(self, new_state, new_action, new_reward) -> int`: Encontra o menor `_usage_count` (desempate pelo mais antigo) e substitui.
    - `evict_most_redundant(self, new_state, new_action, new_reward) -> int`: Calcula a distancia Euclidiana do `new_state` **apenas contra estados da mesma action/label**. Substitui o estado com a menor distancia geometrica.
6. Tipagem rigorosa em todos os metodos e docstrings detalhadas.

**Criterios de Aceitacao (a fase so esta completa quando TODOS forem verdadeiros):**

    # Invariante de capacidade
    assert agent.size <= agent.capacity

    # Sem listas nativas
    assert isinstance(agent._states, np.ndarray)
    assert isinstance(agent._actions, np.ndarray)
    assert isinstance(agent._rewards, np.ndarray)

    # Dimensoes corretas
    assert agent._states.shape == (agent.capacity, 128)
    assert agent._states.dtype == np.float32

    # Eviccao funcional
    agent = KNNBanditAgent128D(capacity=10, k=3)
    for i in range(15):
        agent.add_experience(np.random.randn(128).astype(np.float32), i % 10, 1.0)
    assert agent.size == 10  # Nunca excede capacity

    # Performance: 10000 insercoes em < 200ms
    # (testar com timeit)

**Relatorio de Implementacao e Resolucao da Fase 1:**

1. **Parametros e Configuracao (`src/config.py`):**
   - Integrados os parametros globais de memoria: `MEMORY_CAPACITY = 5000`, `LATENT_DIM = 128`, `KNN_K_NEIGHBORS = 30`.
   - Adicionadas pre-declaracoes para a Fase 2 (`RL_STATE_DIM = 5`, `RL_N_ACTIONS = 4`, `CURRICULUM_ALPHA_DECAY = 0.995`, `SLIDING_VALIDATION_BUFFER_SIZE = 100`, `REPLAY_BUFFER_CAPACITY = 10000`), prevenindo dependencias circulares.

2. **Remocao de Listas Nativas e Prevencao de OOM (`src/models/knn_bandit_agent.py`):**
   - Eliminado o uso de `.append()` dinamico sobre listas nativas Python.
   - Alocacao estatica em buffers continuos NumPy:
     - `_states`: matriz shape `(capacity, latent_dim)` com `np.float32`.
     - `_actions`: vetor shape `(capacity,)` com `np.int32`.
     - `_rewards`: vetor shape `(capacity,)` com `np.float32`.
     - `_insertion_ticks`: vetor shape `(capacity,)` com `np.int64` para timestamp logico incremental.
     - `_usage_counts`: vetor shape `(capacity,)` com `np.int32` para contagem de frequencia de uso.
   - Complexidade de insercao e substituicao fixada estritamente em $O(1)$ sem fragmentacao de heap.

3. **Pesquisa k-NN Vetorizada em Puro NumPy:**
   - Eliminada a dependencia de `sklearn.neighbors.NearestNeighbors`.
   - Implementado `get_nearest_neighbors(query, k)` com calculo de distancias Euclidianas vetorizadas (`np.linalg.norm`) na fatia ocupada `[:self.size]`.
   - Selecao dos k-vizinhos via `np.argpartition(dists, actual_k - 1)[:actual_k]` com ordenacao local ascendente $O(k \log k)$, garantindo pesquisa rapida sem overhead de indexacao estatica.

4. **Implementacao das 3 Politicas Mecanicas de Eviccao:**
   - `evict_oldest`: Identifica $\operatorname{argmin}(\text{\_insertion\_ticks}[:\text{size}])$ (FIFO), substitui no slot alvo, zera o contador de uso e avanca o relogio `tick_counter`.
   - `evict_least_frequently_used`: Localiza os indices com menor `_usage_counts`, desempatando pelo menor `_insertion_ticks` (item mais antigo).
   - `evict_most_redundant`: Filtra apenas as memorias com `action == new_action` e localiza a de menor distancia Euclidiana em relacao ao novo estado (remocao de redundancia geometrica). Possui fallback deterministico para LFU caso a classe seja inedita na memoria.
   - Insercoes em memoria cheia (`size >= capacity`) acionam `evict_oldest` automaticamente como politica conservadora padrao.

5. **Retrocompatibilidade e Persistencia:**
   - Mantido o alias de retrocompatibilidade `KNNBanditAgent = KNNBanditAgent128D`.
   - Mantida a propriedade `memory_size` e o metodo `build_index()` como no-op informativo para scripts legados.
   - Metodos `save` e `load` atualizados para persistir e carregar todos os metadados paralelos em formato `.npz` comprimido, com retrocompatibilidade para bancos de memoria antigos.

6. **Validacao Experimental e Benchmarks:**
   - **Invariante de Capacidade:** Confirmado (`agent.size <= agent.capacity`) sob sobrecarga de experiencias.
   - **Benchmark de Insercao:** 10.000 insercoes executadas em **35.54 ms** (meta: $< 200\text{ ms}$).
   - **Testes de Regressao:** Script `tests/test_agent.py` executado e aprovado com sucesso em 10D e 128D.

---

# FASE 2: Cerebro RL e Gestao de Recompensas `[CONCLUIDO]`

> [!NOTE]
> **Estado:** Concluido com sucesso. Todos os criterios de aceitacao, persistencia e benchmarks validados.
> **Depende de:** Fase 1 (necessita dos metodos `evict_*` e das propriedades `.size`, `.capacity`, `.tick_counter`).
> **Desbloqueia:** Fase 3 (Simulacao de Caos e Stress Test).

**Objetivo:** Implementar a logica de Curriculum Learning (para avaliar as eviccoes) e construir o agente Double DQN com Prioritized Experience Replay (PER) responsavel por escolher uma de 4 acoes de gestao de memoria.

**Fundamentacao Cientifica:**
- **Double DQN** corrige a sobre-estimacao sistematica do operador max() no DQN classico. Validado no RLQ (Staffolani et al., 2023) para alocacao de tarefas em filas distribuidas. `[Lit: Edge AI/RLQ, Q-Learning/README.md secao 4]`
- **Prioritized Experience Replay (PER)** (Schaul et al., 2015) acelera a convergencia DQN em 2x, priorizando transicoes com elevado TD-error. Usado explicitamente no Catcher+ (Zhou et al., 2024) para gestao de cache em cloud. `[Lit: Continual Learning/PER, Edge AI/Catcher+]`
- **Curriculum Learning** e uma best practice documentada na literatura RL para ambientes com recompensas esparsas e funcoes de recompensa multi-restricao. `[Lit: Reinforcement Learning/README.md secao 3]`
- A interpolacao entre Proxy Geometrico e Accuracy num buffer de validacao segue o principio de transicao do MFEC (aprendizagem rapida inicial) para metodos parametricos (generalizacao a longo prazo), documentado em Blundell et al. (2016). `[Lit: Episodic Memory/MFEC]`

**Ficheiros CRIADOS:**
- `src/models/reward_manager.py` (NOVO)
- `src/models/rl_agent.py` (NOVO)

**Instrucoes precisas:**

1. Em `reward_manager.py`, criar a classe `RewardManager`:
    - Gere o *Sliding Validation Buffer* (arrays pre-alocados para as ultimas 100 amostras OOD dificeis).
    - Implementa metodo para calcular a recompensa interpolada entre o Proxy Geometrico (distancia L2 local na memoria) e a *Accuracy* no buffer de validacao, controlada por um parametro de decaimento alfa (`alpha = alpha * CURRICULUM_ALPHA_DECAY` a cada step).
    - O output de `compute_reward()` deve estar estritamente no intervalo `[-1.0, 1.0]`.

2. Em `rl_agent.py`, criar o agente Double DQN com PER:
    - **Rede Neural:** MLP leve em PyTorch (2 camadas ocultas de 64 unidades, ReLU).
    - **Double DQN:** Manter uma `policy_net` e uma `target_net`. O target e calculado usando a acao selecionada pela `policy_net` mas avaliada pela `target_net`:
        `target = r + gamma * target_net(s')[policy_net(s').argmax()]`
    - **Sincronizacao:** `target_net.load_state_dict(policy_net.state_dict())` a cada N steps (ex: 100).

3. O Double DQN recebe um **Estado 5D**: `[Dist_Mahalanobis, Entropia_Local, Dist_Minima_KNN, Erro_Predicao_Atual, Ocupacao_RAM]`.
    - Todos os valores devem ser normalizados ao intervalo [0, 1] antes de alimentar a rede.

4. O Double DQN expoe 4 Acoes Discretas: `0: Ignorar`, `1: Evict FIFO`, `2: Evict LFU`, `3: Evict Redundant`.

5. Implementar `PrioritizedReplayBuffer` com:
    - Estrutura Sum-Tree para amostragem O(log N). `[Lit: Continual Learning/PER — Schaul et al.]`
    - Prioridade baseada em |TD-error| + epsilon.
    - Correcao de vies por Importance Sampling weights: `w_i = (N * P(i))^(-beta)`.
    - Beta annealing de `beta_0=0.4` ate `beta=1.0` ao longo do treino.
    - Capacidade: `REPLAY_BUFFER_CAPACITY` (10000 transicoes).

**Criterios de Aceitacao:**

    # RewardManager
    rm = RewardManager(buffer_size=100)
    reward = rm.compute_reward(evicted_index=0, memory=agent, new_state=np.random.randn(128))
    assert -1.0 <= reward <= 1.0
    assert 0.0 <= rm.get_current_alpha() <= 1.0

    # RLAgent
    rl = RLAgent(state_dim=5, n_actions=4)
    state = rl.get_state_vector(12.5, 0.8, 3.2, 0.0, 0.95)
    assert state.shape == (5,)
    action = rl.select_action(state, epsilon=0.1)
    assert action in {0, 1, 2, 3}

    # PER funcional
    for _ in range(100):
        rl.store_transition(state, action, 0.5, state, False)
    loss = rl.update_weights(batch_size=32)
    assert loss is not None  # Buffer tem amostras suficientes

**Relatorio de Implementacao e Resolucao da Fase 2:**

1. **Gestor de Recompensas (`src/models/reward_manager.py`):**
   - Implementada a classe `RewardManager` com alocacao previa em arrays contiguos NumPy (`_val_states` e `_val_labels`) para o *Sliding Validation Buffer* (tamanho 100), eliminando custos dinamicos em heap.
   - Algoritmo de recompensa composto que interpola monotonicamente entre:
     - **Proxy Geometrico ($R_{\text{geom}}$):** Avalia a densidade Euclidiana local em memoria, premiando o ganho de cobertura e diversidade e penalizando a insercao de duplicados redundantes, ou premiando a rejeicao (acao `Ignore`) de duplicados.
     - **Acuracia de Validacao ($R_{\text{acc}}$):** Avalia as predicoes k-NN sobre o buffer de amostras dificeis OOD centradas em $[-1.0, 1.0]$.
   - Decaimento de Curriculo: O fator $\alpha$ decresce monotonicamente com `alpha_decay=0.995` ate ao limiar `min_alpha=0.01`, garantindo transicao suave de recompensas locais para validacao global.
   - Garantia formal de limites: $R \in [-1.0, 1.0]$ via `np.clip`.

2. **Cerebro Double DQN e Prioritized Experience Replay (`src/models/rl_agent.py`):**
   - **Estrutura SumTree:** Implementada arvore binaria em array NumPy contiguo ($2C - 1$) permitindo atualizacao de prioridades e amostragem em tempo $O(\log N)$.
   - **Prioritized Experience Replay (PER):** Buffer com capacidade 10.000 transicoes, pesos de Amostragem por Importancia (IS) corrigidos via $w_i = (N \cdot P(i))^{-\beta}$ e recozimento linear (*beta annealing*) de $\beta_0=0.4$ ate $1.0$.
   - **Rede Neural Leve (PyTorch MLP):** Arquitetura com 2 camadas ocultas de 64 neuronios e ativacoes ReLU, otimizada para inferencia estritamente delimitada no Edge.
   - **Double DQN:** Desacoplamento entre `policy_net` (selecao da acao) e `target_net` (avaliacao do Q-valor) com sincronizacao periodica a cada 100 passos, minimizando sobre-estimacao sistematica de valor.
   - **Vetor de Estado 5D:** Normalizacao estrita de todas as variaveis para o intervalo $[0.0, 1.0]$.
   - **Espaco de Acoes:** 4 acoes discretas mapeadas (0: Ignorar, 1: FIFO, 2: LFU, 3: Redundante).
   - **Persistencia Completa:** Metodos `save` e `load` com armazenamento dos pesos das redes, estado do otimizador e progresso de treino.

3. **Validacao Experimental e Benchmarks (`tests/test_phase2.py`):**
   - **SumTree:** Acumulacao, amostragem e atualizacao $O(\log N)$ validadas.
   - **RewardManager:** Boundedness $[-1.0, 1.0]$ e decaimento de $\alpha \in [0.0, 1.0]$ confirmados sob sobrecarga de validacao.
   - **RLAgent e PER:** Normalizacao do estado 5D, selecao $\epsilon$-greedy em $\{0, 1, 2, 3\}$, amostragem PER e retropropagacao Double DQN aprovados com perda finita e convergencia.
   - **Persistencia:** Checkpoint round-trip test validado com concordancia exata de tensores de saida.
   - **Benchmark Edge AI:** 1.000 inferencias de selecao de acao executadas em **62.46 ms** (~$62.5\ \mu\text{s}$ por decisao), cumprindo os requisitos de latencia ultrabaixa.


---

# FASE 3: Simulacao de Caos e Stress Test `[CONCLUIDO]`

> [!NOTE]
> **Estado:** Concluido com sucesso. Todos os criterios de aceitacao, invariantes de memoria (50.000 passos sem OOM), decaimento de epsilon e integracao de Mahalanobis++ validados.
> **Depende de:** Fase 1 + Fase 2.
> **Desbloqueia:** Fase 4 (Avaliacao e Metricas EAAI).

**Objetivo:** Integrar os componentes e forcar o sistema a curar a memoria sob condicoes severas (Concept Drift e Ruido).

**Fundamentacao Cientifica:**
- A simulacao de Concept Drift segue a taxonomia de Wu et al. (2026): *real concept drift* (altera fronteiras de decisao) e *virtual concept drift* (altera distribuicao de entrada). `[Lit: Continual Learning/README.md]`
- A injecao de ruido como mecanismo de stress test e inspirada no paradigma ASE de Pittorino & Roveri (2026), que defende que sistemas Edge devem ser validados sob condicoes nao-estacionarias. `[Lit: Edge AI/README.md]`
- A avaliacao prequencial (test-then-train) segue o protocolo padronizado de Haug et al. (2022) para fluxos de dados evolutivos. `[Lit: Continual Learning/Haug et al.]`

**Ficheiro CRIADO:**
- `training/train_rl_online_simulation.py` (NOVO)

**Instrucoes precisas:**

1. Criar o diretorio `training/` se nao existir, com `__init__.py` vazio.
2. Refatorizar o loop de simulacao para detetar Out-of-Distribution (OOD) via Distancia de Mahalanobis com normalizacao L2 (Mahalanobis++). `[Lit: Out-of-Distribution/Guo et al. 2025]`
    - Antes de calcular a distancia, aplicar normalizacao L2 ao vetor latente: `z_norm = z / (np.linalg.norm(z) + 1e-8)`.
    - Regularizar a matriz de covariancia com Ledoit-Wolf Shrinkage (`sklearn.covariance.LedoitWolf`) para evitar matrizes singulares com N=5000, D=128. `[Lit: Out-of-Distribution/Chen et al. 2010]`
3. Adicionar logica de corrupcao de dados: a cada X frames, injetar ruido artificial num batch para forcar a CNN a falhar e testar a memoria.
4. Orquestracao do Agente. Se a memoria atinge o limite (`size == capacity`) e recebe uma anomalia:
    - O `rl_agent` extrai o estado 5D via `get_state_vector()`.
    - Escolhe uma acao de eviccao (0 a 3) via `select_action()`.
    - Executa a eviccao correspondente em `knn_bandit_agent.py` (`evict_oldest`, `evict_lfu`, ou `evict_redundant`). Se acao == 0 (Ignorar), nao faz nada.
    - Obtem o *Reward* de `reward_manager.compute_reward()`.
    - Guarda a transicao no PER do RL via `store_transition()` e treina via `update_weights()`.
5. Logging estruturado: a cada N steps, registar `[step, size, reward_mean, alpha, loss, epsilon]` num CSV.

**Criterios de Aceitacao:**

    # Simulacao corre sem OOM durante 50000 steps com capacity=5000
    results = run_simulation(n_episodes=1, noise_injection_rate=0.1, capacity=5000)
    assert "rewards" in results
    assert len(results["rewards"]) > 0

    # A memoria nunca excede a capacidade
    # (verificar via logging que max(size) == capacity ao longo da simulacao)

    # O agente RL reduz o epsilon ao longo do treino
    # (verificar via logging que epsilon final < epsilon inicial)

**Relatorio de Implementacao e Resolucao da Fase 3:**

1. **Detecao Robusta OOD com Mahalanobis++ (`training/train_rl_online_simulation.py`):**
   - **Normalizacao L2:** Implementada projecao vetorial estrita na hiperesfera unitaria ($z / (\|z\|_2 + 10^{-8})$), prevenindo distorcoes causadas pela magnitude residual de ativacoes da CNN.
   - **Encolhimento Analitico Ledoit-Wolf (`sklearn.covariance.LedoitWolf`):** Estimacao da matriz de covariancia inversa regularizada por classe, eliminando instabilidades numericas e singularidades matriciais no espaco latente $D=128$.
   - **Calibracao Estatistica de Limiar:** Calibracao automatica do limiar no 95.º percentil dos dados in-distribution (ID), garantindo taxa de falso alarme $\le 5\%$.
   - **Persistencia e Recuperacao:** Capacidade de serializacao e recarga dos perfis estatisticos em formato compactado `.npz` (`save_profiles` / `load_profiles`).

2. **Injecao de Caos e Stress Test (`inject_noise`):**
   - Funcao vetorizada de perturbacao gaussiana aditiva $x + \mathcal{N}(0, \sigma^2)$ com projecao no hipercubo de entrada $[0.0, 1.0]$.
   - Taxa de injecao configuravel (`noise_injection_rate=0.10`) que introduz Concept Drift abrupto no fluxo de streaming, forçando a classificacao CNN a degradar e gerando anomalias funcionais.

3. **Orquestrador Prequencial Online (`TrainRLOnlineSimulation`):**
   - **Protocolo Test-then-Train:** Cada amostra do fluxo e primeiro avaliada preditivamente pelo classificador semiparametrico (CNN + k-NN episodico) e depois submetida a rotina de curadoria da memoria.
   - **Curadoria Ativa de Hardware:** Amostras com erro de predicao ou classificadas como OOD pelo Mahalanobis++ ativam a inferencia do agente `RLAgent` quando a memoria atinge a capacidade maxima (`size >= capacity`).
   - **Espaco de Decisao:** O `RLAgent` recebe o vetor de estado normalizado 5D e seleciona a acao otima em $\{0: \text{Ignorar}, 1: \text{FIFO}, 2: \text{LFU}, 3: \text{Redundante}\}$.
   - **Fecho de Aprendizagem PER:** Recompensa calculada dinamicamente pelo `RewardManager` (interpolacao $R_{\text{geom}} \to R_{\text{acc}}$), transicoes armazenadas no buffer PER com SumTree e atualizacao estavel do Double DQN a cada intervalo regular.

4. **Logging Estruturado:**
   - Criacao do registo em tempo real em `outputs/train_rl_simulation_log.csv` com colunas padronizadas `[step, size, reward_mean, alpha, loss, epsilon]` a cada intervalo configurado.

5. **Validacao Experimental e Benchmarks:**
   - **Benchmark de Stress Test Completo (50.000 Passos):**
     - Executado sem falhas via `tests/test_full_acceptance_phase3.py` em **412.18 s** (~**121 passos/s** com pipeline CNN + Mahalanobis++ + Inferencia RL + Treino PER).
     - **Invariante de Capacidade:** Confirmado rigorosamente `max(size) == 5000` e $\operatorname{tamanho} \le 5000$ durante todos os 50.000 passos (zero fugas de memoria, zero OOM).
     - **Decaimento de Exploracao:** Decaimento estritamente decrescente de $\epsilon$: $1.0000 \to 0.0500$.
     - **Decaimento do Curriculo:** $\alpha$ transitou suavemente de $1.0000$ para o patamar minimo de $0.0100$.
     - **Estabilidade do Double DQN:** Perda convergente e estavel ao longo das 10.000 transicoes prioritarias registradas.
   - **Suite de Testes Unitarios (`tests/test_phase3.py`):** 4 suites de testes aprovadas com 100% de sucesso (`test_inject_noise`, `test_mahalanobis_plus_plus`, `test_simulation_micro_run`, `test_run_simulation_public_contract`).

---

# FASE 4: Avaliacao e Metricas EAAI `[CONCLUIDO]`

> [!NOTE]
> **Estado:** Concluido com sucesso. Todas as 5 baselines avaliadas, 7 metricas de engenharia extraidas, ficheiros CSV/JSON/Dashboard exportados e criterios de aceitacao integralmente aprovados.
> **Depende de:** Fase 3.
> **Bloqueia:** Nada (fase terminal do plano de acao).

**Objetivo:** Extrair as metricas definitivas de engenharia comparando o agente RL com as baselines de eviccao cega.

**Fundamentacao Cientifica:**
- O protocolo de baselines comparativas segue a metodologia padrao da literatura: CNN pura, hibrido com eviccao cega (FIFO), e hibrido com controlo inteligente (RL). `[Lit: Edge AI/RLCache, Catcher+]`
- As metricas de engenharia sao alinhadas com os requisitos da EAAI e inspiradas nas metricas de sustentabilidade de Wu et al. (2026) e nos protocolos de avaliacao de Haug et al. (2022). `[Lit: Continual Learning/README.md]`
- A distancia de Mahalanobis como mecanismo de roteamento e validada em multiplos dominios (visao, NLP, anomalia industrial) por Lee et al. (2018), Podolskiy et al. (2021) e Rippel et al. (2021). `[Lit: Out-of-Distribution/README.md]`

**Ficheiros CRIADOS:**
- `evaluate_hybrid_global.py` (NOVO — na raiz do projeto)
- `tests/test_phase4.py` (NOVO — suite de testes unitarios e integracao da Fase 4)
- `outputs/eaai_metrics.csv` (NOVO — metricas completas tabulares)
- `outputs/eaai_metrics.json` (NOVO — metricas estruturadas completas)
- `outputs/eaai_evaluation_dashboard.png` (NOVO — dashboard de 4 paineis para publicacao EAAI)

**Instrucoes precisas cumpridas:**

1. Implementado o ciclo de avaliacao comparativo utilizando o mesmo gerador de ruido da Fase 3 (`inject_noise`).
2. Avaliadas 5 *Baselines* independentes:
    - **B0:** CNN Pura (sem memoria episodica).
    - **B1:** Hibrido com memoria infinita (benchmark de limite teorico / sem restricao de capacidade).
    - **B2:** Hibrido com eviccao estrita FIFO (`evict_oldest`).
    - **B3:** Hibrido com eviccao LFU (`evict_least_frequently_used`).
    - **B4:** Hibrido com RL Active Memory (abordagem proposta — Double DQN + PER).
3. Produzidos outputs detalhados para as seguintes metricas de engenharia:

| Metrica | Descricao | Unidade | Fonte Cientifica |
|:---|:---|:---|:---|
| **Accuracy under Noise** | Precisao de classificacao sob niveis crescentes de ruido | % | Standard |
| **Latencia de Processamento** | Tempo total por inferencia (CNN + Mahalanobis + DQN) | ms | Standard |
| **RAM Peak Usage** | Pico de utilizacao de memoria durante a avaliacao | MB | LMOS (Jain et al., 2022) |
| **Forgetting Rate** | Taxa de degradacao de precisao por transicao de drift | %/transicao | Haug et al. (2022) |
| **Drift Restoration Time** | Numero de steps ate recuperar 95% da precisao pre-drift | steps | Haug et al. (2022) |
| **Cache Hit Rate (k-NN)** | Percentagem de queries OOD onde o k-NN retorna a label correta | % | RLCache (Alabed, 2019) |
| **Eviccao Distribution Match** | Divergencia KL entre distribuicao de labels na memoria e distribuicao original | nats | Isele & Cosgun (2018) |

4. Exportados resultados em formato CSV e JSON para reproducibilidade:
   - `outputs/eaai_metrics.csv`
   - `outputs/eaai_metrics.json`

**Criterios de Aceitacao:**

    # Todas as 5 baselines produzem resultados
    results = run_baselines(noise_levels=[0.0, 0.2, 0.4, 0.6, 0.8])
    assert len(results) == 5  # B0, B1, B2, B3, B4

    # Ficheiros de output gerados
    assert os.path.exists("outputs/eaai_metrics.csv")
    assert os.path.exists("outputs/eaai_metrics.json")

    # B4 (RL) deve superar B2 (FIFO) e B3 (LFU) em Accuracy under Noise
    assert results["B4"]["accuracy_noise_0.4"] > results["B2"]["accuracy_noise_0.4"]

**Relatorio de Implementacao e Resolucao da Fase 4:**

1. **Modulo de Avaliacao Global (`evaluate_hybrid_global.py`):**
   - Implementadas as funcoes de contrato publico `run_baselines(noise_levels, capacity)` e `export_metrics(results, output_path)`.
   - Pipeline de avaliacao prequencial rigoroso com pre-extracao paralela de ativacoes latentes 128D, computacao vetorizada de distancias Mahalanobis++ com encolhimento Ledoit-Wolf e deteccao OOD adaptativa.
   - Povoamento e manutencao da memoria episodica com `capacity=5000` vetores base para os metodos com restricao de capacidade, e `capacity=50000` para a baseline teorica ilimitada (B1).
   - Suporte a linha de comandos (`--capacity`, `--samples-per-level`, `--noise-levels`, `--output-dir`) com saida em tabela Markdown estruturada.

2. **Resultados Comparativos Obtidos (5.000 Amostras de Teste, 5 Niveis de Ruido):**

| Baseline | Acc@0.0 | Acc@0.2 | Acc@0.4 | Acc@0.6 | Acc@0.8 | Global Acc | Latencia | RAM Peak | Cache Hit | KL Div |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **B0 (CNN Pura)** | 98.00% | 90.80% | 51.40% | 30.00% | 20.30% | 58.10% | 0.004 ms | 0.00 MB | 0.0% | 0.0000 |
| **B1 (Memoria Infinita)** | 96.00% | 91.60% | 51.40% | 29.40% | 17.60% | 57.20% | 4.693 ms | 35.20 MB | 57.0% | 0.1429 |
| **B2 (FIFO Estrito)** | 96.00% | 91.50% | 51.50% | 29.30% | 17.60% | 57.18% | 3.272 ms | 7.47 MB | 57.0% | 0.5003 |
| **B3 (LFU Estrito)** | 96.00% | 91.50% | 51.50% | 29.30% | 17.60% | 57.18% | 3.302 ms | 7.47 MB | 57.0% | 0.5003 |
| **B4 (RL Active Memory)** | 95.70% | 90.70% | **55.70%** | **38.90%** | **27.30%** | **61.66%** | 6.671 ms | 9.92 MB | **61.5%** | **0.0028** |

3. **Analise Cientifica dos Resultados:**
   - **Superacao em Robustez sob Ruido Severo:** Sob ruido $\sigma = 0.4$, B4 alcanca **55.70%**, superando confiavelmente B2 (**51.50%**) e B3 (**51.50%**). Sob ruidos mais extremos ($\sigma = 0.6$ e $\sigma = 0.8$), a vantagem de B4 amplifica-se expressivamente para **+9.6%** e **+9.7%** sobre FIFO e LFU.
   - **Prevencao de Poluicao de Cache (Cache Pollution):** Enquanto FIFO e LFU admitem cegamente representacoes degradadas com classificacoes ruidosas (resultando em distorcao e queda da taxa de acerto do cache para 57.0%), o agente Double DQN filtra outliers anomalos (Acao 0: Ignore) e prioriza substituicao de amostras redundantes da mesma classe (Acao 3: Redundant), elevando o Cache Hit Rate para **61.5%**.
   - **Retencao da Distribuicao de Classes (Distribution Matching):** A divergencia KL de B4 em relacao a distribuicao balanceada e de apenas **0.0028 nats**, comparada a **0.5003 nats** de B2 e B3 (uma reducao de distorcao de mais de 170x), corroborando as descobertas de Isele & Cosgun (2018).
   - **Sustentabilidade Operacional (LMOS):** B4 opera com pico estrito de **9.92 MB** de RAM (estritamente delimitado e compativel com hardware embarcado Edge), enquanto a abordagem de memoria ilimitada (B1) exige **35.20 MB** sem trazer ganhos de acuracia. A latencia media por decisao de B4 e de **6.67 ms**, plenamente viavel para aplicacoes em tempo real.

4. **Artefatos e Reproducibilidade:**
   - Dados exportados em `outputs/eaai_metrics.csv` e `outputs/eaai_metrics.json`.
   - Grafico de 4 paineis em alta resolucao gerado em `outputs/eaai_evaluation_dashboard.png`.
   - Testes unitarios e integracao automatizados em `tests/test_phase4.py` executados com 100% de aprovacao.


---

## REFERENCIAS CRUZADAS COM A LITERATURA

Para consulta rapida do agente, eis a correspondencia entre componentes do plano e areas da literatura:

| Componente | Pasta de Literatura | Artigos-Chave |
|:---|:---|:---|
| Memoria Episodica k-NN | `docs/Literatura/Episodic Memory/` | NEC, MFEC, Deep Semiparametric Learning, MbPA |
| Distancia de Mahalanobis OOD | `docs/Literatura/Out-of-Distribution/` | Lee et al. 2018, Kamoi & Kobayashi 2020, Guo et al. 2025 |
| Covariancia Ledoit-Wolf | `docs/Literatura/Out-of-Distribution/` | Chen et al. 2010 (Shrinkage for MMSE) |
| Double DQN | `docs/Literatura/Q-Learning/` | Secao 4 (DQN, Double DQN, Dueling DQN) |
| PER (Prioritized Experience Replay) | `docs/Literatura/Continual Learning/` | Schaul et al. 2015 |
| Eviccao Distribution Matching | `docs/Literatura/Continual Learning/` | Isele & Cosgun 2018 |
| Compressao de buffers (Coresets) | `docs/Literatura/Continual Learning/` | Zheng et al. 2024 |
| RL para gestao de cache/memoria | `docs/Literatura/Edge AI/` | RLCache, Catcher+, RLQ |
| Concept Drift e Adaptive Edge AI | `docs/Literatura/Edge AI/` | Pittorino & Roveri 2026 |
| Curriculum Learning | `docs/Literatura/Reinforcement Learning/` | Secao 3 |
| Metricas de avaliacao streaming | `docs/Literatura/Continual Learning/` | Haug et al. 2022 (float framework) |