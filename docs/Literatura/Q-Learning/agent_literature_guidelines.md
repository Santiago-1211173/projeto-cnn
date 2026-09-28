# Protocolo de Curadoria e Navegação na Literatura: Q-Learning em Redes Neuronais e Visão Computacional

> **Ficheiro Central Associado:** `README.md` (no mesmo diretório)  
> **Natureza do Documento:** Guia metodológico de recolha, análise crítica e referenciação científica para agentes autónomos e investigadores.  
> **Domínio Científico:** Inteligência Artificial / Aprendizagem Automática / Aprendizagem por Reforço Sem Modelo (*Model-Free Reinforcement Learning*), Otimização de Políticas de Decisão e Aplicações Híbridas em Visão Computacional.

---

## 1. Enquadramento Teórico & Âmbito do Tópico

### 1.1 Definição Canónica na Literatura Internacional de IA
O **Q-Learning** (introduzido formalmente por Watkins em 1989 e consolidado por Watkins & Dayan em 1992) é um algoritmo fundamental de **Aprendizagem por Reforço Sem Modelo (*Model-Free Reinforcement Learning*)** baseado em Diferenças Temporais (*Temporal Difference Learning* - TD), desenhado para aprender a função ação-valor ótima $Q^*(s, a)$ diretamente a partir da experiência de interação com o ambiente, sem exigir conhecimento prévio da dinâmica de transição ou do modelo de recompensas.

O problema de decisão sequencial é formalizado no âmbito de um **Processo de Decisão de Markov (MDP)** quíntuplo definido por $\mathcal{M} = (\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma)$, onde:
- $\mathcal{S}$ denota o conjunto de estados admissíveis do ambiente;
- $\mathcal{A}$ representa o conjunto de ações que o agente pode executar;
- $\mathcal{P}(s' \mid s, a) = \mathbb{P}(S_{t+1} = s' \mid S_t = s, A_t = a)$ define a probabilidade de transição de estado;
- $\mathcal{R}(s, a) = \mathbb{E}[R_{t+1} \mid S_t = s, A_t = a]$ expressa o valor esperado da recompensa imediata escalar;
- $\gamma \in [0, 1)$ é o fator de desconto temporal aplicado a recompensas futuras.

A função ação-valor $Q^{\pi}(s, a)$ quantifica o retorno cumulativo esperado ao tomar a ação $a$ no estado $s$ sob uma política arbitrária $\pi(a \mid s)$:
$$Q^{\pi}(s, a) = \mathbb{E}_{\pi} \left[ \sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \;\middle|\; S_t = s, A_t = a \right]$$

O princípio de otimalidade de Bellman postula que a função ação-valor ótima $Q^*(s, a) = \max_{\pi} Q^{\pi}(s, a)$ satisfaz a equação recursiva:
$$Q^*(s, a) = \mathcal{R}(s, a) + \gamma \sum_{s' \in \mathcal{S}} \mathcal{P}(s' \mid s, a) \max_{a' \in \mathcal{A}} Q^*(s', a')$$

A regra de atualização do Q-Learning aproxima esta solução de forma iterativa e estocástica (*off-policy*):
$$Q(S_t, A_t) \leftarrow (1 - \alpha) Q(S_t, A_t) + \alpha \left[ R_{t+1} + \gamma \max_{a \in \mathcal{A}} Q(S_{t+1}, a) \right]$$
ou, equivalentemente, expressa através do termo de erro de diferença temporal (*TD Error* $\delta_t$):
$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \, \delta_t, \quad \text{onde } \delta_t = R_{t+1} + \gamma \max_{a \in \mathcal{A}} Q(S_{t+1}, a) - Q(S_t, A_t)$$
em que $\alpha \in (0, 1]$ representa a taxa de aprendizagem (*learning rate*).

```
+---------------------------------------------------------------------------------------------------+
|                                 Ciclo Interativo do Q-Learning                                    |
+---------------------------------------------------------------------------------------------------+

            +-------------------------------------------------------------+
            |                        AMBIENTE                             |
            |       (MDP: Estado S_t, Recompensa R_t, Transições P)       |
            +-------------------------------------------------------------+
                        ^                                    |
                        | Ação A_t                           | Estado S_{t+1}
                        | (ex.: Camada CNN / Rotação)        | Recompensa R_{t+1}
                        |                                    v
            +-------------------------------------------------------------+
            |                         AGENTE                              |
            |                                                             |
            |   1. Observa S_t e seleciona A_t via Política eps-greedy:   |
            |      A_t = argmax_a Q(S_t, a)  [com prob. 1 - eps]          |
            |      A_t ~ Uniforme(A)         [com prob. eps]              |
            |                                                             |
            |   2. Recebe transição (S_t, A_t, R_{t+1}, S_{t+1})          |
            |                                                             |
            |   3. Computa TD Target: y_t = R_{t+1} + gamma * max_a Q     |
            |      Calcula TD Error:  delta_t = y_t - Q(S_t, A_t)         |
            |                                                             |
            |   4. Atualiza Tabela Q ou Aproximador de Parâmetros:        |
            |      Q(S_t, A_t) <- Q(S_t, A_t) + alpha * delta_t           |
            +-------------------------------------------------------------+
```

---

### 1.2 Problema Fundamental que Aborda
O Q-Learning aborda e resolve quatro limitações fundamentais dos sistemas tradicionais de aprendizagem profunda e processamento de sinal visual:

1. **Dependência Crítica de Modelos do Ambiente (*Model-Free Independence*):**
   - Na programação dinâmica clássica (como a Iteração de Valor ou de Política), o cálculo de políticas ótimas exige o conhecimento estrito da matriz de probabilidades de transição $\mathcal{P}(s' \mid s, a)$ e da função de recompensa média $\mathcal{R}(s, a)$.
   - Em problemas práticos complexos — como o espaço combinatório de hiperparâmetros de uma rede convolucional ou as interações ótico-geométricas de alinhamento visual —, estas matrizes são analiticamente intratáveis. O Q-Learning opera por amostragem pura de trajetórias empíricas, dispensando qualquer aproximação explícita da dinâmica do ambiente.

2. **Desacoplamento entre Comportamento e Otimização (*Off-Policy Separation*):**
   - Algoritmos *on-policy* (como o SARSA) avaliam e aprimoram a política que está a ser executada no momento, o que torna a convergência vulnerável à estocasticidade da exploração.
   - O Q-Learning é intrinsecamente *off-policy*: a atualização da função $Q$ assume sempre a ação maximizadora futura $\max_a Q(S_{t+1}, a)$, garantindo que o agente converge assintoticamente para a política ótima $\pi^*(s) = \operatorname{argmax}_a Q^*(s, a)$ mesmo que as trajetórias de treino sejam recolhidas através de políticas exploratórias altamente ruidosas (como $\epsilon$-greedy ou políticas aleatórias). Isso viabiliza o uso de memórias de repetição de experiências (*Experience Replay buffers*).

3. **Superação do Viés Humano e da Tentativa-e-Erro em Arquiteturas Neuronais (NAS):**
   - O desenho convencional de Redes Neuronais Convolucionais (CNNs) depende de heurísticas empíricas manuais formuladas por especialistas humanos, frequentemente enviesadas para padrões simétricos e intuitivos (ex.: dobrar canais a cada redução de resolução).
   - O Q-Learning atua como um meta-otimizador autónomo (*Neural Architecture Search* - NAS): ao formular a escolha sequencial de camadas (convoluções, ativações, *kernel sizes*, *pooling*, *softmax*) como passos de decisão num MDP, o agente descobre topologias não convencionais e contra-intuitivas (ex.: blocos com *upsampling* inesperado de canais ou passos assimétricos de *pooling*) que superam as arquiteturas desenhadas manualmente em termos de acurácia e eficiência.

4. **Mitigação da Explosão Combinatória do Espaço de Estados em Visão (*State Explosion Bottleneck*):**
   - A incorporação direta de tensores de características visuais de alta dimensão em agentes de reforço resulta frequentemente na maldição da dimensionalidade, instabilidade nos gradientes e custos computacionais proibitivos.
   - A literatura recente contorna este estrangulamento reformulando o MDP em espaços de estados minimalistas (ex.: *Two-State Q-Learning*), onde métricas escalares simples de dispersão estatística ou desvio padrão de *scores* sintetizam a incerteza do sistema, permitindo que matrizes $Q$ de dimensões minúsculas ($2 \times 2$ ou $2 \times 3$) controlem transformações geométricas ativas (como rotações angulares) e alcancem convergência ultrarrápida.

---

### 1.3 Taxonomia e Principais Famílias de Abordagens no Estado da Arte

As aplicações e variantes de Q-Learning intersetadas com redes neuronais organizam-se em quatro famílias metodológicas complementares:

```
                                Taxonomia de Q-Learning & Deep Learning
                                                  |
         +--------------------+-------------------+-------------------+--------------------+
         |                    |                                       |                    |
  1. Representação do  2. Meta-Otimização e    3. Controlo Ativo de    4. Mecanismos de
     Espaço Q             Pesquisa Estrutural     Entrada & Alinhamento   Estabilidade
         |                    |                   (Active Vision)             |
   - Q-Learning         - NAS Sequencial            |                   - Experience Replay
     Tabular Canónico     (MetaQNN)           - Two-State Q-Learning    - Target Networks
   - Minimalista        - Seleção de              (Hafiz)               - Double Q-Learning
     (Two-State Q)        Hiperparâmetros     - Rotações / Inclinação   - Agendamento de
   - Deep Q-Networks    - Early-Exit MDP          do Campo Visual         Decaimento eps
     (DQN / Dueling)      Control             - Ajuste de Resolução     - Recompensas Atrasadas
```

#### Família 1: Representação e Dimensionalidade do Espaço de Valor Q
- **Q-Learning Tabular Canónico:**  
  Armazena os valores discretos numa matriz $Q \in \mathbb{R}^{|\mathcal{S}| \times |\mathcal{A}|}$. Possui garantias formais de convergência para o ponto ótimo sob condições de Robbins-Monro ($\sum_t \alpha_t = \infty$ e $\sum_t \alpha_t^2 < \infty$), mas é limitado a espaços discretos de dimensões tratáveis.
- **Abordagens de Estado Minimalista (*Two-State Q-Learning*):**  
  Reduz deliberadamente o espaço contínuo ou volumoso de características para uma partição binária ($|\mathcal{S}| = 2$) baseada em métricas de incerteza (ex.: estado 0 = baixa concordância de *scores*, estado 1 = concordância consolidada). Elimina a necessidade de redes profundas para aproximar $Q$, assegurando tempos de treino em milissegundos e robustez a sobreajuste.
- **Deep Q-Networks (DQN) e Extensões Parametrizadas:**  
  Substitui a tabela por uma rede neuronal $Q(s, a; \theta) \approx Q^*(s, a)$ para processar entradas sensoriais contínuas de alta dimensão (ex.: tensores de imagem bruta). Utiliza perdas de erro quadrático médio em relação ao alvo temporal:
  $$\mathcal{L}(\theta) = \mathbb{E}_{(s, a, r, s') \sim \mathcal{D}} \left[ \left( r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta) \right)^2 \right]$$

#### Família 2: Meta-Otimização Estrutural (Neural Architecture Search - NAS)
- **Construção Sequencial de Topologias (MetaQNN / Block-QNN):**  
  O processo de desenho da rede é modelado como um MDP em que cada estado $s_t$ representa a sequência parcial de camadas construídas até ao passo $t$. As ações $a_t$ correspondem a selecionar uma nova camada (ex.: Convolução com $F$ filtros e filtro $K \times K$, *Max-Pooling*, *Dropout*, *Dense*) e os seus respetivos hiperparâmetros. A recompensa é atribuída de forma atrasada (*delayed reward*) com base na acurácia de validação do modelo totalmente treinado.
- **Controlo de Alocação de Camadas e Compressão de Modelos:**  
  Agentes de Q-Learning que selecionam dinamicamente rácios de poda (*pruning*), fatores de quantização por camada, ou decidem a inserção estratégica de ramificações de saída precoce (*Early-Exits*).

#### Família 3: Controlo Ativo de Entrada e Alinhamento Híbrido (*Active Visual Adaptation*)
- **Transformações Geométricas de Campo Visual:**  
  O agente atua como um sistema foveal ou de foco ativo: dada uma imagem ambígua ou não centrada, o agente Q-Learning seleciona perturbações geométricas ativas (rotações discretas em ângulos $+\theta, -\theta$ ou translações espaciais) para alinhar a entrada antes da extração de características por extratores convolucionais consolidados e congelados (ex.: ResNet50, InceptionV3, AlexNet).
- **Classificadores Híbridos em Cascata:**  
  Sistemas em que o Q-Learning decide se a imagem deve ser classificada de imediato pela representação original ou se uma versão manipulada/permutada deve alimentar um segundo classificador para maximizar a confiança combinada.

#### Família 4: Técnicas de Regularização e Estabilização de Treino
- **Repetição de Experiência (*Experience Replay*):**  
  Armazenamento de transições passadas num *buffer* finito $\mathcal{D} = \{ (s_t, a_t, r_{t+1}, s_{t+1}) \}$. Ao amostrar mini-batches uniformemente de $\mathcal{D}$, quebra-se a forte correlação temporal intrínseca a sequências contínuas de observação e estabiliza-se o gradiente.
- **Redes Alvo Desacopladas (*Target Networks* $\theta^-$):**  
  Utilização de um conjunto congelado de pesos para computar a estimativa de Bellman $y_t = r + \gamma \max_{a'} Q(s', a'; \theta^-)$, atualizado periodicamente a cada $C$ passos para evitar divergências circulares.
- **Agendamento Estrito de Decaimento $\epsilon$ (*Epsilon Decay Schedule*):**  
  Controlo rigoroso do balanço entre exploração inicial e explotação madura, crucial para lidar com a estocasticidade induzida pelo treino estocástico (SGD/Adam) das arquiteturas avaliadas.

---

### 1.4 Dicionário de Conceitos-Chave (Pedagógico e Rigoroso)

*   **Agente (*Agent*):**  
    Entidade algorítmica autónoma responsável pela perceção do estado corrente, tomada de decisões através de uma política de controlo e atualização dos valores de utilidade com base no feedback retornado pelo ambiente.
*   **Estado (*State* $S \in \mathcal{S}$):**  
    Representação descritiva da configuração atual do ambiente. Em tarefas de arquitetura neural (NAS), sintetiza a sequência ordenada de camadas já concatenadas; em abordagens como o *Two-State Q-Learning*, corresponde a um indicador discreto binário que codifica o nível de concordância ou dispersão de *scores* de predição.
*   **Espaço de Ações (*Action Space* $\mathcal{A}$):**  
    Conjunto exaustivo de decisões operacionais permitidas ao agente num determinado instante. Pode compreender escolhas discretas de hiperparâmetros de camadas (tamanho de *kernel*, canais de saída, passo de *stride*) ou transformações espaciais na imagem de entrada (rotação anti-horária, rotação horária, translação).
*   **Sinal de Recompensa (*Reward Signal* $R \in \mathbb{R}$):**  
    Valor escalar de retorno fornecido pelo ambiente após a execução de uma ação. Pode apresentar-se como uma **recompensa atrasada** (*delayed reward*), atribuída apenas no final de um episódio (ex.: a acurácia de validação após treinar uma CNN), ou como uma **recompensa imediata**, calculada pela variação na margem de confiança do classificador.
*   **Tabela Q (*Q-Table*):**  
    Estrutura de dados matricial em que as linhas indexam estados $s \in \mathcal{S}$ e as colunas indexam ações $a \in \mathcal{A}$, armazenando os valores numéricos $Q(s, a)$ que representam a utilidade esperada a longo prazo.
*   **Função Ação-Valor ($Q(s, a)$):**  
    Medida matemática do retorno acumulado com desconto temporal que o agente espera auferir a partir do estado $s$, executando a ação $a$ e adotando a política ótima nos passos subsequentes.
*   **Equação de Bellman (*Bellman Optimality Equation*):**  
    Relação recursiva que estabelece que o valor de um par estado-ação ótimo equivale à recompensa imediata somada ao valor esperado descontado da melhor ação possível no próximo estado.
*   **Erro de Diferença Temporal (*TD Error* $\delta_t$):**  
    A discrepância aritmética entre a nova estimativa do valor futuro (o alvo TD: $R_{t+1} + \gamma \max_a Q(S_{t+1}, a)$) e a estimativa atual $Q(S_t, A_t)$. Constitui o sinal de condução de toda a aprendizagem.
*   **Exploração vs. Explotação (*Exploration-Exploitation Dilemma*):**  
    Tensão fundamental entre a necessidade de escolher ações não consolidadas para descobrir se produzem recompensas superiores (*exploração*) e a necessidade de escolher a ação com maior valor $Q$ conhecido para maximizar o ganho presente (*explotação*).
*   **Política $\epsilon$-greedy:**  
    Estratégia probabilística de seleção de ações em que o agente adota uma ação aleatória uniforme com probabilidade $\epsilon \in (0, 1)$ e seleciona a ação gananciosa $\operatorname{argmax}_a Q(s, a)$ com probabilidade $1 - \epsilon$. O parâmetro $\epsilon$ decresce tipicamente ao longo do treino (*epsilon decay*).
*   **Replay de Experiência (*Experience Replay Buffer*):**  
    Memória circular de armazenamento onde são registadas as tuplas de experiência passadas $(S_t, A_t, R_{t+1}, S_{t+1})$. Ao amostrar transições de forma aleatória e desacoplada, quebra-se a autocorrelação dos dados sequenciais e estabiliza-se o processo estocástico de otimização.
*   **Processo de Decisão de Markov (MDP):**  
    Modelo matemático formal de controlo estocástico caracterizado pela propriedade de Markov: a probabilidade de transição para o estado seguinte depende estritamente do estado e da ação imediatamente anteriores, sendo condicionalmente independente de todo o histórico prévio.
*   **Two-State Q-Learning:**  
    Formulações especializadas de Q-Learning que condensam o espaço de estados a apenas dois níveis discretos ($|\mathcal{S}| = 2$), simplificando radicalmente a matriz de valores, eliminando hiperparâmetros de redes de valor e viabilizando a convergência quase instantânea da função de controlo.
*   **Pesquisa de Arquitetura Neuronal (NAS - *Neural Architecture Search*):**  
    Campo da inteligência artificial focado em automatizar a descoberta e o dimensionamento de topologias de redes neuronais, substituindo o desenho empírico manual por agentes autónomos de aprendizagem por reforço ou algoritmos evolucionários.
*   **Estocasticidade da Recompensa:**  
    Variabilidade intrínseca no sinal de feedback observada quando a recompensa depende de processos estocásticos secundários, tais como a inicialização aleatória de pesos de uma CNN ou a convergência por gradiente descendente estocástico (SGD), exigindo métodos de regularização temporal no agente de RL.

---

## 2. Protocolo de Pesquisa Sistemática (Search Strategy)

Para assegurar uma cobertura exaustiva, metodologicamente irrepreensível e isenta de enviesamento na literatura de Q-Learning em redes neuronais e visão computacional, agentes autónomos e investigadores devem cumprir o protocolo sistemático detalhado infra.

### 2.1 Venues Científicos Prioritários
As pesquisas bibliográficas devem priorizar estritamente publicações que tenham passado por rigoroso processo de revisão por pares (*peer-review*) nas seguintes conferências e revistas de topo internacional:

| Categoria | Sigla / Nome do Venue | Qualificação / Foco Científico |
|:---|:---|:---|
| **Conferências de IA & Aprendizagem Automática (Core A\*)** | **NeurIPS** (Neural Information Processing Systems) | Fundamentos de Aprendizagem por Reforço, garantias de Bellman e DQN |
| | **ICML** (International Conference on Machine Learning) | Teoria de Q-learning, bounds de convergência, off-policy learning e NAS |
| | **ICLR** (International Conference on Learning Representations) | Representação profunda em RL, meta-aprendizagem de arquiteturas neuronais |
| | **AAAI** (Association for the Advancement of Artificial Intelligence) | Algoritmos de decisão sequencial, heurísticas de exploração e modelos híbridos |
| **Conferências de Visão Computacional & Robótica** | **CVPR** (Computer Vision and Pattern Recognition) | Visão ativa, seleção de ações geométricas e alinhamento visual com CNNs |
| | **ICCV** / **ECCV** (Int. Conf. on Computer Vision) | Redes convolucionais adaptativas, deteção ativa e atenção guiada por RL |
| | **IROS** / **ICRA** (Robotics & Automation) | Controlo de foveação visual e tomada de decisão motora baseada em Q-Learning |
| **Revistas Científicas de Referência** | **IEEE T-PAMI** (Trans. on Pattern Analysis and Machine Intelligence) | Fundamentos matemáticos de visão e convergência de modelos híbridos CNN-RL |
| | **IEEE T-NNLS** (Trans. on Neural Networks and Learning Systems) | Otimização temporal, estabilidade de Q-tables e arquiteturas neuronais |
| | **JMLR** (Journal of Machine Learning Research) | Provas teóricas de convergência em espaços tabulares e aproximações lineares |
| | **Machine Learning** (Springer) | Métodos seminais de RL, exploração estocástica e processos de Markov |
| **Repositórios de Preprints Verificados** | **arXiv** (`cs.LG`, `cs.AI`, `cs.CV`, `cs.RO`) | Avanços emergentes de fronteira nos últimos 12 a 36 meses |

---

### 2.2 Equações de Pesquisa Booleanas Recomendadas

As seguintes cadeias de pesquisa encontram-se estruturadas para motores de indexação científica avançados (ex.: IEEE Xplore, Google Scholar, ACM Digital Library, Web of Science, Scopus e ScienceDirect):

#### Bloco A: Fundamentos Teóricos e Formulações Minimalistas de Q-Learning
```text
("Q-learning" OR "Q learning" OR "temporal difference" OR "Bellman optimality") AND ("model-free" OR "off-policy" OR "experience replay") AND ("two-state" OR "state space reduction" OR "convergence analysis")
```

#### Bloco B: Q-Learning Aplicado à Pesquisa de Arquiteturas Neuronais (NAS)
```text
("Q-learning" OR "Q-Learning agent") AND ("Neural Architecture Search" OR "NAS" OR "CNN architecture search" OR "automated deep learning") AND ("convolutional neural network" OR "layer selection" OR "MetaQNN")
```

#### Bloco C: Modelos Híbridos de Visão Computacional e Controlo Ativo de Imagem
```text
("Q-learning" OR "reinforcement learning") AND ("image classification" OR "computer vision") AND ("hybrid classifier" OR "image rotation" OR "geometric transformation" OR "visual field tilt") AND ("ResNet" OR "Inception" OR "CNN")
```

#### Bloco D: Estabilidade em Ambientes com Recompensa Estocástica
```text
("Q-learning" OR "Deep Q-Network") AND ("stochastic reward" OR "delayed reward" OR "noisy environment") AND ("epsilon decay" OR "exploration-exploitation" OR "experience replay")
```

---

### 2.3 Janela Temporal de Análise

A recolha bibliográfica deve ser estruturada em duas janelas cronológicas complementares:

1. **Janela Seminal e Fundacional (1989 – 2017):**
   - **Objetivo:** Compreender as bases matemáticas da aprendizagem por diferenças temporais e a génese da fusão entre RL e redes profundas.
   - **Marcos Fundacionais:**
     - Watkins (1989 - *Learning from Delayed Rewards*): Formulação pioneira da regra de atualização do Q-Learning.
     - Watkins & Dayan (Machine Learning 1992 - *Q-Learning*): Prova formal de convergência do algoritmo sob processos de Markov discretos.
     - Sutton & Barto (MIT Press 1998/2018 - *Reinforcement Learning: An Introduction*): Consolidação epistemológica do TD-learning e controlo off-policy.
     - Mnih et al. (Nature 2015 - *Human-level control through deep reinforcement learning*): Introdução do Deep Q-Network (DQN), acoplando representações convolucionais e *experience replay*.
     - Baker et al. (ICLR 2017 - *Designing Neural Network Architectures using Reinforcement Learning / MetaQNN*): Demonstração pioneira do uso de agentes Q-Learning para descobrir arquiteturas de CNN competitivas.
     - Zoph & Le (ICLR 2017 - *Neural Architecture Search with Reinforcement Learning*): Automatização de desenho de redes via políticas de decisão sequencial.

2. **Janela de Avanços Recentes e Estado da Arte (Últimos 3 a 5 anos):**
   - **Objetivo:** Analisar as fronteiras modernas em simplificação de espaços de estado, classificação visual ativa e integração eficiente de RL em sistemas de visão.
   - **Marcos Recentes Relevantes:**
     - Hafiz (Handbook of Intelligent Computing 2022 - *Image Classification by Reinforcement Learning with Two-State Q-Learning*): Proposta de um classificador híbrido ultraleve com apenas 2 estados para governar rotações ativas do campo visual em cooperação com CNNs consolidadas.
     - He, Shen & Xu (IEEE T-NNLS 2021/2023 - *Reinforcement Learning for Computer Vision: A Survey*): Taxonomia aprofundada de abordagens de RL em deteção ativa, atenção sequencial e navegação visual.
     - Avanços contemporâneos em AutoML e NAS adaptativo sob restrições estritas de hardware periférico (*Edge AI*), integrando Q-Learning com orçamentos de latência e consumo de FLOPs.

---

## 3. Critérios de Elegibilidade (Inclusão e Exclusão)

Para preservar o rigor conceptual e a consistência epistemológica da base documental, todas as publicações identificadas devem ser submetidas aos critérios de filtragem descritos no fluxograma e secções seguintes:

```
                            Artigo Localizado na Pesquisa
                                          |
                    +---------------------+---------------------+
                    |                                           |
                    v                                           v
           Critérios de Inclusão (+)                   Critérios de Exclusão (-)
        - Formulação explícita de MDP               - Ausência de definição de MDP/Ações
        - Validação contra baselines reais          - Sem baselines comparativos consolidados
        - Detalhe de hiperparâmetros (alpha, eps)   - Modelos puramente conceituais/sem código
        - Trade-offs de acurácia vs. complexidade   - Treino opaco sem curvas de convergência
        - Reprodutibilidade e dados públicos        - Veículos sem revisão por pares credível
                    |                                           |
                    v                                           v
             ACEITE NA TABELA                           REJEITADO
```

### 3.1 Critérios de Inclusão (+)
O documento deve satisfazer cumulativamente pelo menos **três** dos seguintes requisitos para inclusão formal:
1. **Formulação Rigorosa do MDP:** O artigo especifica claramente a modelação do ambiente como Processo de Decisão de Markov, definindo formalmente o Espaço de Estados ($\mathcal{S}$), o Espaço de Ações ($\mathcal{A}$), a Função de Recompensa ($\mathcal{R}$) e o Fator de Desconto ($\gamma$).
2. **Comparação com *Baselines* Reconhecidos:** Avalia empiricamente o algoritmo contra arquiteturas de referência concebidas por humanos (ex.: ResNet50, InceptionV3, AlexNet) ou contra abordagens alternativas de busca/classificação (ex.: algoritmos genéticos, pesquisa aleatória ou classificadores estáticos).
3. **Transparência Algorítmica e de Hiperparâmetros:** Reporta os valores exatos de taxa de aprendizagem ($\alpha$), agendamento de decaimento de exploração ($\epsilon$), dimensão do *buffer* de repetição de experiências e passos de treino/convergência.
4. **Análise de Custo Computacional e Complexidade:** Documenta o impacto da introdução do Q-Learning em termos de número de parâmetros a otimizar, tempo de procura (*search time*) ou eficiência de inferência face a abordagens de força bruta.
5. **Validação em Conjuntos de Dados Padronizados:** Realiza ensaios empíricos em *benchmarks* públicos e auditáveis (ex.: ImageNet, CIFAR-10/100, Caltech-101, ProstateX, MNIST, SVHN).

### 3.2 Critérios de Exclusão (-)
Devem ser liminarmente rejeitados os documentos que incidam em qualquer uma das seguintes insuficiências:
1. **Definição Ambígua da Recompensa ou Ações:** Trabalhos que utilizam a terminologia de Q-Learning mas não explicitam matematicamente a função de recompensa nem detalham as ações operatórias disponíveis para o agente.
2. **Ausência de Avaliação Empírica Comparativa:** Textos que reportam apenas o desempenho da rede encontrada ou alinhada sem confrontar os resultados com os modelos originais sob as mesmas condições de teste.
3. **Complexidade Oculta Intratável:** Propostas em que o tempo de treino do agente de controlo é incomensuravelmente superior aos benefícios operacionais obtidos, sem fornecer uma discussão crítica sobre a viabilidade prática.
4. **Documentos Sem Revisão por Pares Idónea:** Artigos provenientes de editoras predatórias desprovidas de indexação em bases científicas de referência, notas de fóruns informais, ou relatórios sem metodologia replicável.

---

## 4. Roteiro de Extração e Síntese Analítica

Para cada artigo validado na fase de elegibilidade, o curador ou agente autónomo de IA deve preencher exaustivamente a seguinte **Checklist de 5 Pontos**:

```
+---------------------------------------------------------------------------------------------------+
|                     CHECKLIST ANALÍTICA DE EXTRAÇÃO DE ARTIGOS DE Q-LEARNING                     |
+---------------------------------------------------------------------------------------------------+
| 1. Problema e Motivação                                                                           |
|    - Que lacuna metodológica aborda (viés manual em CNNs, explosão de estados, alinhamento visual)?   |
|    - Porque falham os classificadores estáticos ou os métodos clássicos de otimização?             |
+---------------------------------------------------------------------------------------------------+
| 2. Inovação Metodológica / Arquitetural                                                           |
|    - Como está formulado o MDP (espaço de estados, ações ativas, estrutura da Q-table ou rede)?      |
|    - Qual a equação da função de recompensa (imediata vs. atrasada)?                             |
|    - Que mecanismos de estabilização foram empregues (epsilon decay, Experience Replay)?          |
+---------------------------------------------------------------------------------------------------+
| 3. Datasets e Protocolo de Avaliação                                                              |
|    - Quais os conjuntos de dados utilizados (ImageNet, ProstateX, Caltech-101, CIFAR-10)?         |
|    - Quais os baselines de confronto (ResNet50, InceptionV3, modelos desenhados por peritos)?     |
+---------------------------------------------------------------------------------------------------+
| 4. Resultados Empíricos e Trade-offs (Precisão vs. Eficiência)                                    |
|    - Qual o ganho quantitativo de acuidade face aos modelos estáticos não modificados?            |
|    - Houve redução no número de parâmetros otimizados (ex.: tabela de apenas 2 estados)?          |
|    - Como se comportou o tempo de convergência da política ótima?                                 |
+---------------------------------------------------------------------------------------------------+
| 5. Limitações e Desafios em Aberto (Open Challenges)                                             |
|    - Como afeta a estocasticidade do treino por SGD a estabilidade do sinal de recompensa?         |
|    - Como escala o método para espaços visuais contínuos de maior complexidade?                   |
+---------------------------------------------------------------------------------------------------+
```

### Detalhe Operacional de Cada Ponto da Checklist:

1. **Problema e Motivação:**  
   Identificar com precisão o estrangulamento abordado na interseção entre aprendizagem por reforço e visão. Caracterizar se o objetivo é automatizar a síntese estrutural de camadas (*Neural Architecture Search*) para eliminar preconceitos de engenharia humana ou se pretende introduzir adaptação geométrica foveal (*Two-State Q-Learning*) para corrigir desalinhamentos em imagens de teste.
2. **Inovação Metodológica/Arquitetural:**  
   Mapear matematicamente a modelação do MDP: especificação do espaço de estados $\mathcal{S}$ (ex.: estados baseados em desvio padrão de predições vs. cadeia de camadas), catálogo de ações $\mathcal{A}$ (ex.: rotações $+\theta, -\theta$ vs. hiperparâmetros de convolução), formulação escalar da recompensa $R$, valor de $\gamma$, e presença de memórias de repetição de experiências.
3. **Datasets e Protocolo de Avaliação:**  
   Extrair os detalhes experimentais: bases de imagens utilizadas, divisões de treino/validação/teste, extratores convolucionais base utilizados (ResNet50, AlexNet, InceptionV3), e procedimentos de normalização e pré-processamento.
4. **Resultados Empíricos e Trade-offs:**  
   Registar as métricas objetivas reportadas: percentagem de acurácia de classificação, matriz de confusão, número de parâmetros da função $Q$ (ex.: $2 \times 2$ vs. redes profundas densas), e comparação direta com modelos de topo sem o agente de RL.
5. **Limitações e Desafios em Aberto:**  
   Registar com honestidade intelectual as fragilidades e desafios sublinhados pelos autores: ruído estocástico decorrente de gradientes estocásticos (SGD), incapacidade de convergência em dados puramente sintéticos ruidosos, custo de avaliação exaustiva em conjuntos de grande escala e dependência de hiperparâmetros de decaimento de exploração.

---

## 5. Padrão de Formatação para a Tabela de Literatura (`README.md`)

O ficheiro central `README.md` localizado no mesmo diretório reúne o estado da arte numa tabela padronizada de 6 colunas. Qualquer novo registo deve cumprir escrupulosamente a estrutura especificada.

### 5.1 Especificação Rigorosa das 6 Colunas Obrigatórias

| Coluna | Título | Tipo de Conteúdo e Regras de Formatação |
|:---:|:---|:---|
| **1** | **Nome** | Título oficial e integral do artigo em língua inglesa, sem omissões ou alterações de grafia. |
| **2** | **Detalhes** | Metadados catalográficos estruturados verticalmente com tags HTML `<br><br>`, compreendendo: Autores, Data de publicação, Publisher, Livro/Journal/Conferência, Volume, Número, Páginas e hiperligação DOI ativa. |
| **3** | **Abstract** | Citações textuais integrais e literais dos excertos mais relevantes do resumo original da publicação, delimitadas obrigatoriamente entre aspas duplas (`"..."`). |
| **4** | **Conclusion** | Citações textuais fiéis e diretas retiradas da secção de conclusões ou considerações finais do artigo original, delimitadas entre aspas duplas (`"..."`). |
| **5** | **Resumo (NotebookLM)** | Síntese analítica redigida em língua portuguesa, estruturada impreterivelmente em **quatro parágrafos encadeados** separados por tags `<br><br>` (ver secção 5.2). |
| **6** | **Citação** | Referência bibliográfica formatada em estrita conformidade com a **Norma Vancouver (NLM)**, finalizada com a hiperligação DOI funcional. |

---

### 5.2 Estrutura Padronizada do "Resumo (NotebookLM)" (4 Parágrafos)

A quinta coluna deve obedecer rigidamente à seguinte progressão conceptual de quatro blocos:

*   **Parágrafo 1 — Contexto e Problema de Investigação:**  
    Apresentação do artigo, enquadramento da abordagem de Q-Learning no domínio da aprendizagem profunda ou visão computacional e explicitação clara da limitação prática ou teórica que o estudo visa superar (ex.: complexidade de desenho manual de CNNs ou desalinhamento ótico em inferência).
*   **Parágrafo 2 — Mecanismo Proposto e Inovação Arquitetural:**  
    Descrição técnica da formulação do MDP concebida pelos investigadores: definição do espaço de estados, catálogo de ações operatórias, estrutura da função $Q$ (tabela reduzida vs. aproximador paramétrico), formulação da função de recompensa e estratégias de exploração/estabilização adotadas.
*   **Parágrafo 3 — Validação Experimental e Métricas Numéricas:**  
    Resumo dos ensaios empíricos conduzidos, indicando conjuntos de dados utilizados, modelos de comparação (*baselines*), métricas quantitativas de acurácia alcançadas e ganhos observados em eficiência ou dimensionalidade de otimização.
*   **Parágrafo 4 — Significado Teórico, Limitações e Desafios:**  
    Apreciação do impacto científico da contribuição na área de modelos híbridos ou AutoML, acompanhada da enumeração explícita das limitações reportadas (ex.: sensibilidade à estocasticidade da recompensa, tempo de treino ou desafios de generalização) e direções futuras sugeridas pelos autores.

---

### 5.3 Exemplo Canónico de Registo na Tabela

Para efeitos de calibração metodológica, apresenta-se de seguida o modelo de preenchimento integral extraído de uma das referências centrais do repositório:

```markdown
| Image Classification by Reinforcement Learning with Two-State Q-Learning | **Autores:** Abdul Mueed Hafiz<br><br>**Data de publicação:** 11 fevereiro 2022<br><br>**Publisher:** Wiley-Scrivener<br><br>**Livro/Journal:** Handbook of Intelligent Computing and Optimization for Sustainable Development<br><br>**Volume:** 3<br><br>**Páginas:** 171-181<br><br>**DOI:** https://doi.org/10.1002/9781119792642.ch9Digital | "In this paper, a simple and efficient Hybrid Classifier is presented which is based on deep learning and reinforcement learning."<br><br>"Here, Q-Learning has been used with two states and 'two or three' actions."<br><br>"Because the proposed technique uses only two Q-states it is straightforward and consequently has much lesser number of optimization parameters, and thus also has a simple reward function."<br><br>"The proposed approach outperforms others techniques on all the datasets used." | "In this paper, a straightforward and efficient learning system is investigated which combines deep learning with reinforcement learning."<br><br>"The proposed technique is simpler than other contemporary techniques found elsewhere. This is for the reason that others use high number of states while as the proposed approach uses only two states."<br><br>"A novel technique i.e. rotation has been used which is similar to tilt of visual field."<br><br>"The proposed approach outperforms other approaches including ResNet50, InceptionV3, etc. on all the three datasets used." | Este artigo apresenta um classificador híbrido simples e eficiente para classificação de imagens, combinando modelos de *deep learning* (redes neuronais convolucionais) e *reinforcement learning* (Q-Learning).<br><br>Diferente de outras abordagens na literatura que utilizam um elevado número de estados devido à grande dimensão dos mapas de características, este método emprega apenas dois estados Q e duas ou três ações, com destaque para a ação inovadora de rotação da imagem.<br><br>Esta arquitetura torna a abordagem mais direta, reduzindo drasticamente os parâmetros de otimização e simplificando a função de recompensa.<br><br>As experiências efetuadas em bases de dados conhecidas (ImageNet, Cats and Dogs e Caltech-101) comprovam que esta técnica supera o desempenho de algoritmos padrão recentes, como o ResNet50 e o InceptionV3. | Hafiz AM. Image Classification by Reinforcement Learning with Two-State Q-Learning. In: Handbook of Intelligent Computing and Optimization for Sustainable Development. Wiley-Scrivener; 2022. Vol. 3, p. 171–81. https://doi.org/10.1002/9781119792642.ch9Digital. |
```

---

## 6. Guardrails de Qualidade & Anti-Alucinação

Para garantir que a base documental mantém integridade científica inatacável e cumpre os mais rigorosos padrões da comunidade internacional de inteligência artificial, agentes autónomos e curadores devem obedecer obrigatoriamente às seguintes diretrizes:

### 6.1 Verificação Rigorosa de DOIs e Metadados
- **Resolução Funcional:** Todo o DOI introduzido no ficheiro deve ser autenticado em bases oficiais através do prefixo canónico `https://doi.org/...`. Não é permitida a inclusão de identificadores fictícios, extrapolados ou com ligações quebradas.
- **Auditoria de Catalogação:** Os nomes dos autores, título integral, ano, veículo de publicação, volume e intervalo de páginas devem ser confirmados por cruzamento com plataformas catalográficas confiáveis (DBLP, CrossRef, IEEE Xplore, ACM Digital Library ou Google Scholar).

### 6.2 Proibição Estrita de Interpolação e Fabricação Numérica
- Os resultados experimentais reportados (taxas de acurácia, número de estados, número de ações, reduções de parâmetros de otimização) devem espelhar com exatidão matemática o que se encontra demonstrado no artigo original.
- É categoricamente proibido arredondar números de forma conveniente, inventar métricas em conjuntos de dados não testados ou generalizar desempenhos particulares como se fossem universais.

### 6.3 Fidelidade Literal das Citações Textuais
- Os campos **Abstract** e **Conclusion** destinam-se exclusivamente a **transcrições literais de excertos da obra**.
- É terminantemente vedada a introdução de paráfrases subjetivas, resumos pessoais ou alterações no vocabulário original no interior das aspas delimitadoras.

### 6.4 Sobriedade Epistemológica e Linguagem Científica Neutra
- O texto das análises e sínteses deve pautar-se por uma linguagem técnica formal, neutra e equilibrada.
- São expressamente banidos termos promocionais ou hiperbólicos (tais como "algoritmo revolucionário", "precisão mágica", "desempenho perfeito", "solução definitiva"). O foco analítico deve incidir estritamente na fundamentação do MDP, nas equações de Bellman, nas evidências empíricas auditáveis e nas limitações metodológicas transparentes.
