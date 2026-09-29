# Documentação Técnica: KNN Bandit Agent (128D)

Este documento descreve detalhadamente o funcionamento e a arquitetura do agente de Reinforcement Learning (RL) implementado no ficheiro `src/models/knn_bandit_agent_128d.py`. 

Este script implementa a classe `KNNBanditAgent128D`, que adota uma abordagem de **Contextual Bandits** com recurso a **k-Nearest Neighbors (k-NN)** e **Memória Episódica**, em vez de utilizar uma rede neuronal paramétrica tradicional (como DQN ou PPO).

---

## 1. Abordagem Teórica: Porquê k-NN em vez de Deep RL?

Em sistemas híbridos de Visão Computacional + RL, o uso de redes neuronais (Deep RL) para a componente de decisão pode trazer problemas:
- **Instabilidade e Esquecimento Catastrófico:** Redes neuronais precisam de muito tempo de treino, afinação de hiperparâmetros (learning rate, discount factor, etc.) e podem "esquecer" aprendizagens anteriores se a distribuição de dados mudar.
- **Opacidade:** É difícil interpretar por que motivo uma rede neuronal tomou uma determinada decisão (Black Box).

Para resolver isto, o `KNNBanditAgent128D` utiliza **Aprendizagem Baseada em Instâncias (Instance-Based Learning)**:
1. **Memória Episódica:** Em vez de aprender "pesos", o agente memoriza todas as experiências passadas sob a forma de tuplos `(Estado, Ação, Recompensa)`.
2. **Decisão por Semelhança (k-NN):** Perante uma nova situação, o agente procura no seu "caderno" (a memória episódica) as `k` situações passadas mais semelhantes. A ação recomendada será aquela que deu mais lucro (recompensa) no passado nessas situações específicas.

Isto proporciona **transparência total**: podemos sempre inspecionar a memória e ver exatamente quais os `k` vizinhos que justificaram a decisão do agente.

---

## 2. Arquitetura e Pipeline de Decisão

O pipeline deste agente divide-se nas seguintes fases cruciais:

### 2.1. O Estado (State)
O agente recebe como estado um vetor de características (*latent features*) extraído pela camada penúltima de uma CNN, correspondendo a **128 dimensões** (128D). Este vetor encapsula a representação abstrata da imagem.

```python
def add_experience(self, state_128d: np.ndarray, action: int, reward: float) -> None:
    # O estado recebido (state_128d) é "achatado" num array unidimensional
    self._states.append(np.asarray(state_128d, dtype=np.float32).flatten())
```

### 2.2. Redução de Dimensionalidade (PCA)
Trabalhar diretamente com distâncias Euclidianas em 128 dimensões sofre da **"Maldição da Dimensionalidade"** (em dimensões muito altas, todas as distâncias tendem a ser parecidas, degradando a eficácia do k-NN).
- O agente utiliza **PCA (Principal Component Analysis)** para reduzir opcionalmente as 128 dimensões para um número mais tratável (por omissão, `48D`).
- Isto condensa a informação mais relevante, ignora o ruído e melhora brutalmente o desempenho do algoritmo k-NN.

```python
if self.use_pca:
    self._pca = PCA(n_components=self.pca_components)
    logger.info(f"A ajustar PCA para {self.pca_components} componentes...")
    self._states_array = self._pca.fit_transform(self._states_array)
```

> [!NOTE]
> **Preservação dos Dados Originais (128D vs 48D):** A memória episódica (armazenada no disco e na lista de Python `_states`) **guarda sempre o vetor original de 128 dimensões**. O PCA não apaga a informação! A redução para 48D acontece apenas em tempo de execução dentro da função `build_index()`, que cria uma representação temporária (`_states_array`) puramente para o motor de procura espacial k-NN. Isto é uma excelente prática arquitetural que garante que possa ajustar o número de componentes do PCA a qualquer altura no futuro sem perder a experiência acumulada do agente.

### 2.3. Memória Episódica e Índice (KDTree)
As experiências passadas são mantidas em três listas sincronizadas: `_states`, `_actions` e `_rewards`. 

**Representação conceptual da memória:**

| Índice na Memória | `_states` (Vetor 128D) | `_actions` (Ação tomada) | `_rewards` (Recompensa) |
| :--- | :--- | :---: | :---: |
| **0** | `[0.15, -0.42, 0.88, ..., 0.11]` | `3` | `1.0` |
| **1** | `[-0.22, 0.91, -0.05, ..., -0.73]` | `3` | `-1.0` |
| **2** | `[0.44, -0.10, 0.33, ..., 0.02]` | `7` | `1.0` |
| **...** | ... | ... | ... |
| **N** | `[0.81, 0.22, 0.55, ..., -0.14]` | `2` | `1.0` |

Quando a "fase de treino/memorização" está concluída, o método `build_index()` utiliza o algoritmo `NearestNeighbors` do `scikit-learn` para estruturar estes dados (normalmente construindo internamente uma *KDTree* ou *BallTree*), permitindo procuras espaciais ultrarrápidas.

**KDTree vs BallTree: Como funcionam e como o algoritmo escolhe?**
- **A Necessidade:** Procurar os `k` vizinhos exaustivamente numa memória episódica com 50.000 experiências seria lento demais (força-bruta, $O(N)$). Estas estruturas em árvore organizam o espaço matemático permitindo a pesquisa ignorar blocos inteiros de pontos distantes, acelerando o processo para complexidades aproximadas de $O(\log N)$.
- **KDTree (k-Dimensional Tree):** Particiona o espaço através de cortes retilíneos (hiperplanos perpendiculares aos eixos). É incrivelmente rápida para baixas dimensionalidades (normalmente abaixo das 20 dimensões). No entanto, à medida que a dimensionalidade sobe, a sua estrutura de "caixas" começa a perder muita eficiência (um fenómeno derivado da maldição da dimensionalidade).
- **BallTree:** Em vez de cortes retos, particiona o espaço em "esferas" hiperespaçais agrupadas (bolas dentro de bolas). Esta abordagem geométrica é desenhada especificamente para gerir e mitigar ineficiências em espaços de alta dimensionalidade.
- **A Escolha no Código (`algorithm='auto'`):** Ao passar `algorithm='auto'` para o `NearestNeighbors`, deixamos o `scikit-learn` analisar a estrutura dos dados no `fit()`. Devido à nossa elevada dimensionalidade (seja com os 128D brutos ou mesmo com os 48D gerados pelo PCA), o motor do `scikit-learn` optará nativamente por construir e indexar a memória usando uma **BallTree**, por ser matematicamente muito mais otimizada para o nosso cenário do que a KDTree. Se nós tivéssemos ajustado o PCA para, digamos, 10 componentes, a biblioteca construiria autonomamente uma KDTree.

```python
# Em dimensões moderadas (30-50), KDTree ou auto funciona bem.
self._nn_index = NearestNeighbors(
    n_neighbors=k_efetivo,
    algorithm='auto',
    metric='euclidean',
    n_jobs=-1  # usar múltiplos cores
)
self._nn_index.fit(self._states_array)
```

### 2.4. Avaliação e Ponderação (Inverse Distance Weighting)
Quando confrontado com um novo estado, o agente:
1. Calcula as distâncias para os `k` vizinhos mais próximos.
2. Atribui um "peso" a cada vizinho usando a fórmula: `weight = 1.0 / (distancia + 1e-8)`. Vizinhos mais próximos (menor distância) têm muito mais influência na decisão do que vizinhos mais afastados.
3. Soma as recompensas passadas (multiplicadas pelo peso) agrupadas por cada ação possível (0-9).

```python
# 1. Obter distâncias e índices dos k vizinhos
distances, indices = self._nn_index.kneighbors(query)

# 2. Ponderação pelo inverso da distância
weights = 1.0 / (neighbor_distances + 1e-8)

# 3. Somatório das recompensas passadas (ponderado)
expected_rewards = np.zeros(self.n_actions, dtype=np.float32)
for action in range(self.n_actions):
    mask = neighbor_actions == action
    if np.any(mask):
        expected_rewards[action] = np.sum(neighbor_rewards[mask] * weights[mask])
```

---

## 3. Análise Detalhada do Código

De seguida, dissecamos as partes mais cruciais do script:

### `__init__` (Construtor)
Define os parâmetros primários:
- `k`: Número de vizinhos a consultar.
- `n_actions`: Número de ações possíveis (ex: 10 dígitos).
- `use_pca` e `pca_components`: Controlam a aplicação da redução de dimensionalidade antes do k-NN.
Inicializa as listas vazias que servirão como Memória Episódica.

### `add_experience` e `add_experience_batch`
Aqui reside o "treino" deste agente, que contrasta fortemente com o treino tradicional em Deep Learning.
- **Sem Backpropagation:** Ao contrário das redes neuronais, não existe um ciclo matemático de recálculo de pesos. O "treino" do agente consiste pura e simplesmente em anotar e guardar as ocorrências nas suas listas sincronizadas (Memória Episódica), sob a forma de Estado, Ação e Recompensa.
- **Processamento Individual:** A função `add_experience` assegura que as características da imagem (o array Numpy) são planificadas (`flatten()`) para garantir consistência estrutural, e adiciona-as ao final da lista.

```python
def add_experience(self, state_128d: np.ndarray, action: int, reward: float) -> None:
    self._states.append(np.asarray(state_128d, dtype=np.float32).flatten())
    self._actions.append(int(action))
    self._rewards.append(float(reward))
```

- **Processamento em Lote (_Batch_):** Quando o sistema necessita de ingerir milhares de imagens de uma só vez (por exemplo, ao "digerir" um dataset de treino), chamar a função individual num loop do programa principal causaria estrangulamentos de performance. A função `add_experience_batch` recebe imediatamente matrizes Numpy completas e injeta-as rapidamente em memória contínua.

```python
def add_experience_batch(self, states: np.ndarray, actions: np.ndarray, rewards: np.ndarray) -> None:
    for i in range(len(states)):
        self._states.append(np.asarray(states[i], dtype=np.float32).flatten())
        self._actions.append(int(actions[i]))
        self._rewards.append(float(rewards[i]))
```

> [!WARNING]
> **O Ciclo de Execução em Tempo Real (Real-Time Execution):** É vital compreender que chamar o `add_experience` **não atualiza o motor de busca k-NN imediatamente**. Em tempo real, quando a CNN vê uma nova imagem e o agente toma uma decisão que gera uma recompensa, esse tuplo é atirado para o final das listas. No entanto, a árvore espacial (BallTree) e o PCA **mantêm-se inalterados**. Isto é feito de propósito para evitar quebras de performance astronómicas (*lag* de recálculo por cada imagem). Para que a nova experiência que acabou de entrar passe a "existir" para o motor k-NN e influencie as previsões seguintes, a arquitetura do programa deve invocar explicitamente a função `build_index()` mais tarde (por exemplo, no fim de um *batch* ou no fim de um episódio) para reconstruir a árvore com os novos dados incluídos.

### `build_index`
Este método converte a "base de dados crua" (as listas Python) num formato otimizado para procura espacial. É aqui que ocorre a ponte entre as estruturas dinâmicas e o motor algébrico de alta performance.

1. **Conversão para Numpy Array:** A lista nativa do Python `self._states` (que é excelente e flexível para irmos adicionando itens um a um durante a fase de memorização com a função `append()`) é convertida numa matriz `np.array` do tipo `np.float32`. Este passo é uma exigência técnica vital: as listas do Python guardam referências para objetos espalhados aleatoriamente pela RAM. Um array Numpy pega em todos esses vetores 128D e agrupa-os num bloco de memória C estruturado e contíguo. Isto tira partido da arquitetura do CPU, permitindo operações vetoriais matemáticas à velocidade da luz.

```python
self._states_array = np.array(self._states, dtype=np.float32)
```

2. **Transformação PCA:** Faz o `fit_transform` do PCA (se ativado). Com a matriz Numpy estruturada, o algoritmo analisa os dados, descobre os eixos de maior variância e devolve uma nova matriz Numpy matematicamente limpa onde cada linha tem agora as dimensões comprimidas (ex: 48D).

```python
if self.use_pca:
    self._pca = PCA(n_components=self.pca_components)
    logger.info(f"A ajustar PCA para {self.pca_components} componentes...")
    self._states_array = self._pca.fit_transform(self._states_array)
```

3. **Construção da Árvore Espacial:** Inicia o objeto `NearestNeighbors(algorithm='auto', metric='euclidean')` e ajusta-o (`fit`) à matriz final. É neste momento que a KDTree ou a BallTree é esculpida fisicamente em memória.

```python
k_efetivo = min(self.k, self.memory_size)
self._nn_index = NearestNeighbors(
    n_neighbors=k_efetivo,
    algorithm='auto',
    metric='euclidean',
    n_jobs=-1
)
self._nn_index.fit(self._states_array)
```

> **Importante:** Este método deve ser chamado obrigatoriamente antes de o agente conseguir fazer qualquer tipo de previsão, e também depois de a memória ter recebido novas experiências.

### `get_expected_rewards` e `get_expected_rewards_batch`
Aqui reside o "cérebro" das previsões. O objetivo é calcular qual a estimativa de retorno (lucro/recompensa) que cada uma das 10 ações possíveis vai gerar, baseando-se nas experiências passadas mais similares.

1. **Preparo e Procura Espacial:** O vetor de características recebido pela CNN (128D) é compactado pelo modelo PCA (ficando com as mesmas 48 dimensões da árvore). De seguida, é atirado para a função `kneighbors` da árvore espacial.
```python
query = np.asarray(state_128d, dtype=np.float32).reshape(1, -1)
if self.use_pca and self._pca is not None:
    query = self._pca.transform(query)

# A magia algébrica: Encontra a distância e o ID na memória dos 'k' vizinhos mais próximos
distances, indices = self._nn_index.kneighbors(query)
```

2. **Inverse Distance Weighting (Ponderação pelo Inverso da Distância):** O algoritmo precisa de dar mais "voz" aos vizinhos mais parecidos. Por isso, a fórmula calcula um peso que é inversamente proporcional à distância. A constante `1e-8` garante que, se houver um vizinho a uma distância exata de `0.0` (uma imagem virtualmente idêntica), o programa não crashe com um erro fatal de "Divisão por Zero".
```python
weights = 1.0 / (neighbor_distances + 1e-8)
```

3. **Agregação e Cálculo de Estimativas:** Usando máscaras booleanas rápidas do Numpy, o código faz o somatório matemático. Por exemplo, percorre todos os vizinhos que representam a ação "3" e soma as suas recompensas (já multiplicadas pelo seu "peso" / "voz"). O output final é um array onde o índice com maior valor é o favorito do agente.
```python
expected_rewards = np.zeros(self.n_actions, dtype=np.float32)
for action in range(self.n_actions):
    mask = neighbor_actions == action # Filtra os vizinhos que tomaram esta ação específica
    if np.any(mask):
        # Soma as recompensas ponderadas apenas dessa ação
        expected_rewards[action] = np.sum(neighbor_rewards[mask] * weights[mask])
```

Na variante `get_expected_rewards_batch`, a fundação algorítmica é idêntica. Contudo, em vez de processar `query` a `query`, processa uma grelha multidimensional `[N, 128]`. O `kneighbors` devolve de imediato todas as distâncias num bloco gigante, e o ciclo itera através da grelha Numpy final (`zip(all_indices, all_distances)`), conseguindo processar milhares de predições cruzadas com a memória num par de milissegundos.

### `get_action` e `get_action_batch`
Implementam a política de exploração **Epsilon-Greedy**. 
- **Exploração (Aleatório):** Se um número aleatório gerado pelo computador for menor que o valor de `epsilon`, a função devolve uma ação completamente à sorte. Isto é essencial no início da vida do agente para o forçar a tentar ações diferentes e ver o que acontece (amostragem tentativa-erro para construir a Memória Episódica).
- **Previsão/Ganância (Inferencia k-NN):** Caso contrário (o comportamento normal em "produção" com epsilon = 0.0), a função vai pelo caminho ótimo: pede as recompensas estimadas baseadas nas experiências dos vizinhos e devolve o índice da ação com o maior valor esperado (`np.argmax`).

```python
def get_action(self, state_128d: np.ndarray, epsilon: float = 0.0) -> int:
    # Exploração: Toma uma decisão puramente aleatória
    if np.random.rand() < epsilon:
        return int(np.random.randint(0, self.n_actions))

    # Ganância: Confia na sabedoria da Memória Episódica (k-NN)
    expected = self.get_expected_rewards(state_128d)
    return int(np.argmax(expected))
```

Para acelerar drasticamente o processamento de grandes lotes de imagens (*batch*), o código do `get_action_batch` implementa esta mesma lógica, mas através de **máscaras vetoriais** (uma técnica de altíssima performance no Numpy). 

```python
def get_action_batch(self, states: np.ndarray, epsilon: float = 0.0) -> np.ndarray:
    n = len(states)
    expected = self.get_expected_rewards_batch(states)
    greedy_actions = np.argmax(expected, axis=1)

    # Cria uma máscara binária onde True = "Força-te a explorar"
    explore_mask = np.random.rand(n) < epsilon
    random_actions = np.random.randint(0, self.n_actions, size=n)

    # np.where: Mistura as respostas. Se a máscara for True usa random, senão usa greedy
    return np.where(explore_mask, random_actions, greedy_actions)
```

### A Metamorfose da Memória: `save` e `load`
A forma como as experiências (Estados, Ações e Recompensas) estão armazenadas muda consoante o ciclo de vida do programa. A memória passa por 3 formatos distintos:

1. **Na Fase de Recolha (Listas Python):** Em tempo real, as memórias não estão numa matriz tabular, mas sim em **três listas Python separadas e sincronizadas** (`_states`, `_actions`, `_rewards`). Elas "comunicam" puramente através do índice partilhado: o estado no índice `5` corresponde à ação e recompensa também no índice `5`. O uso de listas nativas do Python na fase de exploração é totalmente intencional porque o método `.append()` do Python é imensamente mais rápido e flexível para alocar memória dinamicamente do que re-alocar arrays Numpy a cada nova imagem inserida.
2. **Na Fase de Previsão/Busca (Numpy Arrays em RAM):** Como explicado no `build_index`, a lista `_states` sofre uma metamorfose, sendo aglomerada num bloco de memória C contíguo (`_states_array` - Numpy) estritamente para alimentar a BallTree de forma rápida.
3. **No Disco Rígido (Ficheiro `.npz`):** Quando o utilizador pretende desligar o script, as listas em RAM desaparecem. Os métodos `save` e `load` servem para exportar a "mente" (memória episódica) para o disco de forma permanente.
   - A função `save` não exporta as listas nativas. Força a conversão das 3 listas em vetores Numpy individuais e comprime-os todos dentro de um único ficheiro `.npz` (que é na verdade um arquivo em formato dicionário ZIP, altamente otimizado pela livraria Numpy).
   - Quando faz o `load()`, o ficheiro é descompactado para a RAM e os dados voltam a ser instanciados de volta para as 3 listas base de Python para que a recolha (`append()`) possa prosseguir onde parou!

- **Nota Especial no `save`:** Se o PCA foi ativado, exporta juntamente a média (`pca_mean`) e os matrizes de componentes principais (`pca_components_`), para garantir que, ao carregar o modelo meses mais tarde, um estado 128D novo possa ser matematicamente esmagado para 48D usando *exatamente a mesma projeção espacial* em que a memória original do agente vive.

## Conclusão

O script `knn_bandit_agent_128d.py` providencia um subsistema de decisão avançado que une os conceitos de Inteligência Artificial (*Contextual Bandits*) com as melhores práticas de Engenharia de Software e otimização de memória. 

A arquitetura contorna a clássica e morosa afinação (*fine-tuning*) das redes neuronais do Deep RL, apostando num paradigma de **Aprendizagem Baseada em Instâncias** sustentado por:
- **Transparência Absoluta:** Ao usar a inferência k-NN com *Inverse Distance Weighting*, cada decisão do agente pode ser justificada matematicamente inspecionando os vizinhos na Memória Episódica, abolindo o efeito "caixa negra".
- **Gestão de Memória Otimizada:** A dicotomia entre as listas dinâmicas Python (para garantir *appends* ultrarrápidos em execução *real-time*) e os Numpy Arrays *C-contiguous* (para alavancar a performance algébrica do processador nas procuras espaciais) revela um sistema profundamente focado em performance.
- **Robustez Algorítmica:** O uso transversal de máscaras lógicas para execução *batch* veloz, a escolha automática de estruturas espaciais para alta dimensionalidade (*BallTree*), e a compactação tática das *features* de 128D via PCA sem nunca corromper ou apagar o dataset original na memória.

O resultado final é um agente leve, extremamente veloz em predições massivas, interpretável e capaz de estabilizar um sistema híbrido de Visão Computacional.
