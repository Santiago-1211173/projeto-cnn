# Protocolo de Curadoria e Navegação na Literatura: Reinforcement Learning

> **Ficheiro Central Associado:** `README.md` (no mesmo diretório)  
> **Natureza do Documento:** Guia metodológico de recolha, análise crítica e referenciação científica para agentes autónomos e investigadores.  
> **Domínio Científico:** Inteligência Artificial / Aprendizagem Automática / Reinforcement Learning (Aprendizagem por Reforço) e Controlo Ótimo Sequencial.

---

## 1. Enquadramento Teórico & Âmbito do Tópico

### 1.1 Definição Canónica na Literatura Internacional de IA
O **Reinforcement Learning (RL)**, ou Aprendizagem por Reforço, é o paradigma da Inteligência Artificial e da Aprendizagem Automática (*Machine Learning*) dedicado à resolução de problemas de **tomada de decisão sequencial sob incerteza**. Diferencia-se fundamentalmente da Aprendizagem Supervisionada (que depende de exemplos rotulados por um supervisor externo) e da Aprendizagem Não-Supervisionada (que procura descobrir padrões latentes ou estruturas geométricas não-rotuladas nos dados) por operar mediante um processo interativo de **tentativa e erro orientada por objetivos**.

No modelo canónico de RL, uma entidade autónoma denominada **Agente** interage ativamente com um sistema dinâmico externo denominado **Ambiente**. A formalização matemática padrão deste processo assenta na teoria dos **Processos de Decisão de Markov (Markov Decision Processes - MDP)**, descritos formalmente como um tuplo de cinco elementos:

$$\mathcal{M} = \langle \mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma \rangle$$

onde:
*   $\mathcal{S}$ é o conjunto de todos os estados válidos do ambiente (espaço de estados, contínuo ou discreto).
*   $\mathcal{A}$ é o conjunto de ações admissíveis para o agente (espaço de ações, contínuo ou discreto).
*   $\mathcal{P}(s' \mid s, a) = \mathbb{P}(S_{t+1} = s' \mid S_t = s, A_t = a)$ define o modelo de transição dinâmica do ambiente, satisfazendo a **Propriedade de Markov** (o futuro depende unicamente do estado e ação presentes, sendo condicionalmente independente de todo o histórico prévio de trajetórias).
*   $\mathcal{R}(s, a, s') = \mathbb{E}[R_{t+1} \mid S_t = s, A_t = a, S_{t+1} = s']$ denota a função de recompensa escalar imediata emitida pelo ambiente após uma transição.
*   $\gamma \in [0, 1)$ é o **fator de desconto temporal**, que pondera matematicamente a relevância presente de incentivos recebidos em timesteps futuros.

Quando o agente não tem acesso ao estado completo do sistema e apenas recebe sinais sensoriais parciais ou ruidosos, a formulação é expandida para um **Processo de Decisão de Markov Parcialmente Observável (POMDP)**:

$$\mathcal{M}_{\text{POMDP}} = \langle \mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \Omega, \mathcal{O}, \gamma \rangle$$

onde $\Omega$ representa o espaço de observações e $\mathcal{O}(o \mid s') = \mathbb{P}(O_{t+1} = o \mid S_{t+1} = s')$ representa a função de emissão observacional condicional.

O comportamento do agente é governado por uma **política** $\pi$, formalizada como uma distribuição de probabilidade condicional de selecionar uma ação dado o estado atual: $\pi(a \mid s) = \mathbb{P}(A_t = a \mid S_t = s)$ (ou $\pi: \mathcal{S} \to \mathcal{A}$ no caso determinístico).

O objetivo central do agente consiste em determinar a política ótima $\pi^*$ que maximiza o valor esperado do **retorno cumulativo descontado** ($G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$) ao longo da sua trajetória de vida:

$$J(\pi) = \mathbb{E}_{\tau \sim \pi} \left[ G_0 \right] = \mathbb{E}_{\tau \sim \pi} \left[ \sum_{t=0}^{\infty} \gamma^t R_{t+1} \right], \quad \text{onde } \tau = (s_0, a_0, r_1, s_1, a_1, \dots)$$

```
 +---------------------------------------------------------------------------------------+
 |                     CICLO CANÓNICO DE INTERAÇÃO AGENTE-AMBIENTE EM RL                 |
 +---------------------------------------------------------------------------------------+

                                 Ação Executada: A_t
                   +--------------------------------------------+
                   |                                            |
                   v                                            |
         +-------------------+                        +-------------------+
         |                   |   Observação / Estado: |                   |
         |                   |           S_t          |                   |
         |     AMBIENTE      |----------------------->|      AGENTE       |
         |    (Dinâmica      |                        |    (Política      |
         |    P(s'|s,a))     |   Sinal de Recompensa: |     \pi(a|s))     |
         |                   |          R_{t+1}       |                   |
         |                   |----------------------->|                   |
         +-------------------+                        +-------------------+
                   |                                            |
                   |                                            v
                   | Transição Interna                  +-------------------+
                   | de Estado:                         | Replay Buffer /   |
                   +---> S_{t+1}                        | Atualização Bellman|
                                                        | via Gradiente     |
                                                        +-------------------+
```

Para operacionalizar este objetivo, o RL decompõe a avaliação do comportamento através de duas funções fundamentais:
1.  **Função de Valor de Estado ($V^{\pi}(s)$):** Mede o retorno futuro esperado caso o agente inicie no estado $s$ e continue a agir sob a política $\pi$:
    $$V^{\pi}(s) = \mathbb{E}_{\pi} \left[ \sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \;\middle|\; S_t = s \right]$$
2.  **Função de Valor de Ação ou Função Q ($Q^{\pi}(s, a)$):** Mede o retorno esperado ao tomar a decisão arbitrária $a$ no estado $s$, aderindo subsequentemente à política $\pi$:
    $$Q^{\pi}(s, a) = \mathbb{E}_{\pi} \left[ \sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \;\middle|\; S_t = s, A_t = a \right]$$

A pedra basilar da resolução recursiva destas funções reside nas **Equações de Expectativa e Otimalidade de Bellman**:

$$V^{\pi}(s) = \sum_{a \in \mathcal{A}} \pi(a \mid s) \sum_{s', r} \mathcal{P}(s', r \mid s, a) \left[ r + \gamma V^{\pi}(s') \right]$$

$$Q^*(s, a) = \sum_{s', r} \mathcal{P}(s', r \mid s, a) \left[ r + \gamma \max_{a' \in \mathcal{A}} Q^*(s', a') \right]$$

---

### 1.2 Problema Fundamental que Aborda

O Reinforcement Learning foi desenvolvido para ultrapassar limitações teóricas e práticas intransponíveis dos métodos clássicos de engenharia de controlo e dos paradigmas convencionais de redes neuronais:

1.  **Ausência de Rótulos de Supervisão Direta e Ambientes Não-Estacionários:**
    - Em tarefas do mundo real (como condução autónoma, navegação de robôs móveis, exploração tátil ou jogos de estratégia), inexiste um conjunto pré-compilado de dados com a "ação correta" rotulada para cada milissegundo.
    - O agente deve aprender autonomamente a partir de recompensas esparsas, atrasadas e frequentemente ambíguas.
2.  **O Dilema Exploração vs. Explotação (*Exploration-Exploitation Dilemma*):**
    - Para maximizar a recompensa, o agente deve selecionar ações que historicamente produziram altos retornos (*explotação*); contudo, para descobrir essas ações ótimas, é obrigado a tentar opções desconhecidas cujas consequências ainda não domina (*exploração*).
    - O equilíbrio inadequado conduz ou à estagnação em sub-ótimos locais catastróficos ou à divergência por amostragem desordenada.
3.  **O Problema da Atribuição de Crédito Temporal (*Credit Assignment Problem*):**
    - Quando um agente atinge um objetivo complexo após centenas de decisões encadeadas (ou falha criticamente), torna-se matematicamente desafiante determinar com precisão quais as ações passadas que foram determinantes para o resultado.
4.  **A "Tríade Mortal" (*The Deadly Triad*) e Instabilidade no Deep RL:**
    - Ao combinar **aproximação de funções não-lineares** (Redes Neuronais Profundas), **estimativas recursivas baseadas em bootstrapping** (aprendizagem por Temporal-Difference) e **amostragem fora da política (*off-policy training*)**, as garantias clássicas de convergência do RL tabular colapsam, introduzindo riscos severos de divergência numérica e instabilidade assintótica.
5.  **Sobre-estimativa Sistemática dos Valores de Ação (*Overestimation Bias*):**
    - Nas abordagens baseadas em Q-learning e DQN, a aplicação do operador $\max_{a'} Q(s', a')$ sobre valores calculados por aproximadores com ruído estatístico gera um enviesamento positivo cumulativo, distorcendo gravemente as políticas aprendidas.
6.  **Ineficiência de Amostragem (*Sample Inefficiency*) e Desafios de Sim-to-Real:**
    - Algoritmos ingénuos de Deep RL necessitam de dezenas de milhões de interações para aprender comportamentos estáveis. Em sistemas físicos robóticos, esse volume de ensaios causaria desgaste mecânico rápido ou danos operacionais, exigindo o recurso a simuladores com transferência *sim-to-real* robusta via aleatorização de domínio (*domain randomization*).

---

### 1.3 Taxonomia e Principais Famílias de Abordagens no Estado da Arte

O ecossistema contemporâneo do *Reinforcement Learning* estrutura-se em torno de famílias metodológicas complementares, organizadas conforme a representação da política, a presença de modelos dinâmicos do mundo e o mecanismo de exploração:

```
                                Taxonomia de Reinforcement Learning (RL)
                                                   |
         +--------------------+--------------------+--------------------+--------------------+
         |                    |                    |                    |                    |
   1. Baseados em       2. Gradientes de     3. Model-Based &     4. Otimização de     5. RL Agêntico,
      Valor (Value) &      Política e           Planeamento MCTS     Replay Buffer &      Linguagem &
      Deep Q-Networks      Ator-Crítico         (World Models)       Amostragem           Perceção Ativa
         |                    |                    |                    |                    |
     - Q-Learning         - REINFORCE (PG)     - Dyna-Q             - Uniform Replay     - RLHF / DPO / GRPO
     - DQN (Deep Q-Net)   - A2C / A3C          - AlphaZero          - PER (Prioritized)  - Verifier-Free RL
     - Double-DQN         - DDPG / TD3         - MuZero             - CER (Combined)       (NOVER, CLS-RL)
     - Averaged-DQN       - PPO (Clipping)     - EfficientZero      - LAP / PAL          - Agentic Memory
     - TAO-DQN            - SAC (Max-Entropy)  - LightZero (Bench)  - Flexible Replay      (DeltaMem)
                                                                      (Curriculum)       - Hard Attention
                                                                                           (RAM, APPLE)
```

#### Família 1: Abordagens Baseadas em Valor (*Value-Based*) e Deep Q-Networks
- **Princípio Operacional:** O agente não parametriza explicitamente a política de ações; em vez disso, aprende uma função de valor $Q(s, a; \theta)$ que quantifica a qualidade esperada de cada ação e infere a política de forma gananciosa: $\pi(s) = \arg\max_a Q(s, a)$.
- **Modelos e Contribuições Centrais:**
  - **DQN (Deep Q-Network - Mnih et al.):** Pioneiro na fusão de redes convolucionais com Q-learning, introduzindo o *Experience Replay* e a *Target Network* para estabilizar a divergência.
  - **Double-DQN & Averaged-DQN (Anschel et al.):** Desacoplam a seleção da ação da sua avaliação, ou calculam médias móveis sobre os últimos $K$ estimadores $Q$, mitigando drasticamente a sobre-estimativa e a variância dos alvos.
  - **TAO-DQN (Target Accelerated Optimization - Zigon & Song):** Elimina o recálculo redundante de alvos durante as iterações do minibatch, pré-calculando o target no momento da inserção da transição para reduzir a complexidade e a acumulação de erros de maximização.

#### Família 2: Métodos de Gradiente de Política e Arquiteturas Ator-Crítico (*Actor-Critic*)
- **Princípio Operacional:** A política $\pi_\theta(a \mid s)$ é diretamente parametrizada por uma rede neuronal e otimizada por ascensão de gradiente com recurso ao **Teorema do Gradiente de Política (*Policy Gradient Theorem*)**:
  $$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \nabla_\theta \log \pi_\theta(a \mid s) Q^{\pi_\theta}(s, a) \right]$$
- **Modelos e Contribuições Centrais:**
  - **REINFORCE & Modelos Recorrentes de Atenção (Mnih et al. - RAM):** Empregam gradientes de política estocásticos para otimizar decisões não-diferenciáveis, tais como a seleção de coordenadas de foco visual foveal (*hard attention*).
  - **DDPG & TD3 (Twin Delayed DDPG):** Estendem o gradiente de política a espaços contínuos através de dois críticos (*twin critics*) e atraso nas atualizações da política para combater o enviesamento de sobre-estimativa.
  - **PPO (Proximal Policy Optimization):** Algoritmo *on-policy* de referência na indústria, que utiliza uma função objetivo com recorte estocástico (*clipping*) para prevenir passos de atualização excessivamente destrutivos na política:
    $$L^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t \left[ \min\left(r_t(\theta)\hat{A}_t, \, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right) \right]$$
  - **SAC (Soft Actor-Critic):** Enquadramento *off-policy* baseado no paradigma de **Entropia Máxima (*Maximum Entropy RL*)**, que otimiza simultaneamente o retorno esperado e a entropia da política ($\mathcal{H}(\pi(\cdot \mid s_t))$), promovendo exploração contínua e prevenindo o colapso prematuro da política.

#### Família 3: RL Baseado em Modelos e Planeamento por Pesquisa em Árvore (*Model-Based & MCTS*)
- **Princípio Operacional:** O agente aprende explicitamente um modelo do ambiente (que prevê o próximo estado latente e a recompensa esperada) ou executa algoritmos de pesquisa antecipada para planear trajetórias no futuro antes de atuar fisicamente.
- **Modelos e Contribuições Centrais:**
  - **AlphaZero & MuZero:** Unem a Pesquisa em Árvore Monte Carlo (*Monte Carlo Tree Search - MCTS*) a representações latentes profundas aprendidas sem necessidade de um simulador físico perfeito ou regras analíticas prévias.
  - **LightZero (Niu et al.):** Plataforma de referência modular que decompõe o planeamento MCTS em submódulos desacoplados (*Data Collector*, *Data Arranger*, *Agent Learner*, *Agent Evaluator*), viabilizando variantes eficientes (como EfficientZero e Sampled MuZero) em domínios contínuos e estocásticos.

#### Família 4: Mecanismos de Replay Buffer, Amostragem e Aprendizagem por Currículo
- **Princípio Operacional:** Foco na eficiência de amostragem, gestão dinâmica de transições passadas e progressão pedagógica do agente em cenários com recompensas complexas ou esparsas.
- **Modelos e Contribuições Centrais:**
  - **Prioritized Experience Replay (PER):** Amostra transições proporcionalmente ao erro temporal-diferença ($|\delta_t|^\alpha$).
  - **Combined Experience Replay (CER - Zhang & Sutton):** Força a inclusão direta da transição mais recente no minibatch com complexidade $\mathcal{O}(1)$, eliminando o atraso de atualização em grandes buffers.
  - **LAP e PAL (Fujimoto et al.):** Demonstram a equivalência matemática entre amostragem não-uniforme e ajuste de funções de perda, provando que o objetivo do PER pode ser reproduzido sob amostragem uniforme ajustando a função de perda (Loss-Adjusted Prioritized).
  - **Reward Curriculum (Freitag et al. - RC-SAC, RC-TD3):** Currículo em duas fases com buffers adaptativos para evitar *reward hacking* em tarefas com múltiplos objetivos conflitantes.

#### Família 5: RL Agêntico, Alinhamento de Modelos de Linguagem e Perceção Ativa
- **Princípio Operacional:** Aplicação de mecanismos de RL para pós-treino e raciocínio de Grandes Modelos de Linguagem (LLMs/MLLMs), gestão de memória a longo prazo e exploração ativa de sensores.
- **Modelos e Contribuições Centrais:**
  - **Incentive Training e RL Sem Verificador (NOVER - Liu et al., CLS-RL - Li et al.):** Utilizam métricas intrínsecas de consistência e perplexidade como proxy de recompensa para induzir raciocínio passo a passo (*chain-of-thought*) sem depender de verificadores externos rígidos.
  - **Gestão de Memória Agêntica (DeltaMem - Zhang et al.):** Formula a atualização, retenção e recuperação de memórias em agentes autónomos como um MDP otimizado por RL com recompensas baseadas na distância de edição de embeddings.
  - **Perceção Ativa e Exploração Corporizada (APPLE - Schneider et al., Sharafeldin et al.):** Formulação POMDP para controlo de sensores foveais e táteis orientados por minimização de incerteza Bayesiana e *predictive coding*.

---

### 1.4 Dicionário de Conceitos-Chave (Pedagógico e Rigoroso)

*   **Processo de Decisão de Markov (MDP):**  
    Estrutura matemática fundamental que modela problemas de decisão estocástica onde as transições de estado exibem a propriedade markoviana: a probabilidade de transição para o estado seguinte depende estritamente do estado e da ação presentes, e não do historial prévio.
*   **POMDP (Partially Observable MDP):**  
    Generalização do MDP para ambientes onde o agente não tem perceção sensorial do estado ontológico completo, recebendo apenas observações parciais ou locais que exigem a manutenção de um estado de crença (*belief state*) ou memória temporal (ex.: RNNs ou Transformers).
*   **Retorno Descontado ($G_t$):**  
    A soma ponderada de todas as recompensas obtidas pelo agente a partir do instante $t$, onde o fator de desconto $\gamma \in [0, 1)$ assegura a convergência analítica da série infinita e reflete uma preferência matemática por recompensas imediatas face a ganhos distantes.
*   **Política ($\pi(a \mid s)$):**  
    A lei de controlo ou estratégia de tomada de decisão do agente, que mapeia estados do ambiente em distribuições de probabilidade sobre o conjunto de ações possíveis.
*   **Função de Valor de Ação ($Q(s, a)$):**  
    A quantificação matemática do retorno futuro esperado que o agente receberá ao adotar a ação $a$ quando se encontra no estado $s$, aderindo subsequentemente à política de referência.
*   **Equação de Bellman:**  
    Identidade recursiva essencial em programação dinâmica e RL que expressa o valor de qualquer estado (ou par estado-ação) como o somatório da recompensa imediata esperada e do valor descontado do estado seguinte.
*   **Erro Temporal-Diferença (TD-Error, $\delta_t$):**  
    A discrepância aritmética entre a estimativa prévia do valor de um estado-ação e o valor corrigido resultante da recompensa observada no mundo real somada à estimativa descontada do estado sucessor:
    $$\delta_t = R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)$$
*   **Aprendizagem On-Policy vs. Off-Policy:**  
    Algoritmos *on-policy* (ex.: SARSA, PPO) avaliam e melhoram exatamente a mesma política que está a ser utilizada para recolher dados de experiência no ambiente. Algoritmos *off-policy* (ex.: Q-learning, SAC, DDPG) conseguem avaliar e otimizar uma política-alvo utilizando trajetórias geradas por uma política de comportamento totalmente distinta, permitindo o reaproveitamento intensivo de dados através de *replay buffers*.
*   **Buffer de Repetição de Experiências (*Experience Replay Buffer*):**  
    Estrutura de dados circular utilizada em algoritmos *off-policy* para armazenar transições $(s, a, r, s')$. Ao amostrar aleatoriamente minibatches a partir do buffer, quebram-se as fortes correlações temporais entre observações sucessivas, estabilizando o treino de redes neuronais profundas.
*   **Dilema Estabilidade-Plasticidade e Catastrophic Forgetting em RL:**  
    Tendência indesejada das redes neuronais para sobrescrever conhecimentos de políticas e dinâmicas ambientais consolidadas ao longo de episódios anteriores quando confrontadas com novos fluxos de transição num ambiente dinâmico.
*   **Sobre-estimativa de Valor (*Value Overestimation*):**  
    Fenómeno no qual o operador $\max$ presente nas equações de Bellman calcula o máximo sobre aproximações com ruído estatístico, propagando sistematicamente valores de Q inflacionados que induzem o agente a escolher ações sub-ótimas.
*   **Teorema do Gradiente de Política (*Policy Gradient Theorem*):**  
    Resultado analítico fundamental que permite calcular analiticamente o gradiente do desempenho global de uma política parametrizada sem necessitar de conhecer ou derivar o modelo dinâmico de transições do ambiente.
*   **Ator-Crítico (*Actor-Critic*):**  
    Família arquitetural que combina duas estruturas neuronais cooperantes: o **Ator** (responsável por parametrizar e atualizar a política $\pi_\theta(a \mid s)$) e o **Crítico** (responsável por estimar a função de valor $V(s)$ ou $Q(s, a)$ e fornecer um sinal de vantagem escalonado para direcionar os gradientes do ator).
*   **Vantagem ($A(s, a)$):**  
    Métrica que quantifica o ganho relativo de selecionar uma ação específica $a$ no estado $s$ em relação à média ponderada de todas as ações possíveis sob a política vigente: $A(s, a) = Q(s, a) - V(s)$.
*   **Entropia Máxima (*Maximum Entropy RL*):**  
    Formulação estendida de RL (central ao algoritmo SAC) que adiciona um termo de penalização de regularização por entropia à função de recompensa, compelindo o agente a maximizar o retorno enquanto preserva a maior aleatoriedade e dispersão de exploração possível.
*   **Pesquisa em Árvore Monte Carlo (*Monte Carlo Tree Search - MCTS*):**  
    Algoritmo heurístico de planeamento antecipado que constrói assimetricamente uma árvore de decisões com base em simulações iterativas de trajetórias futuras, combinando seleção guiada por limites superiores de confiança (*Upper Confidence Bounds*), expansão, avaliação neural e retropropagação de valores.
*   **Transferência Sim-to-Real e Aleatorização de Domínio (*Domain Randomization*):**  
    Técnica de treino onde o agente é confrontado com milhares de variações estocásticas de propriedades físicas (fricção, massa, iluminação, ruído de latência de sensores) em ambientes simulados, gerando políticas robustas capazes de atuar diretamente em robôs físicos sem falhas decorrentes do *reality gap*.
*   **Exploração Intrínseca e Predictive Coding:**  
    Mecanismo de recompensa interna não-estratificada emitido pelo próprio agente com base na minimização de incerteza epistémica ou no erro de predição dos seus modelos de dinâmica latente, incentivando a descoberta de estados novos sem depender de recompensas externas.
*   **Treino por Incentivo Sem Verificador (*Verifier-Free RL*):**  
    Paradigma emergente para alinhamento e indução de raciocínio lógico em Modelos de Linguagem e Agentes Inteligentes, onde sinais estatísticos internos (como perplexidade normalizada ou coerência de pensamento) substituem oráculos ou funções de recompensa programadas manualmente.

---

## 2. Protocolo de Pesquisa Sistemática (Search Strategy)

Para assegurar uma cobertura bibliográfica completa, rigorosa e representativa da fronteira de conhecimento em *Reinforcement Learning*, investigadores e agentes autónomos devem executar o seguinte protocolo estruturado.

### 2.1 Venues Científicos Prioritários

A recolha bibliográfica deve incidir prioritariamente sobre publicações com revisão científica por pares de padrão internacional elevado nos seguintes fóruns:

| Categoria | Sigla / Nome do Venue | Foco Temático e Relevância |
|:---|:---|:---|
| **Conferências Principais de IA e ML (Core A\*)** | **NeurIPS** (Neural Information Processing Systems) | Fundamentos de Deep RL, teoremas de convergência, MCTS e RLHF |
| | **ICML** (International Conference on Machine Learning) | Algoritmos de gradiente de política, otimização de replay buffers, equivalências de perda |
| | **ICLR** (International Conference on Learning Representations) | Representações latentes, AutoRL, modelos mundiais e arquiteturas Transformer para RL |
| | **AAAI** / **IJCAI** | Raciocínio autónomo, agentes heurísticos e métodos híbridos de controlo |
| **Conferências de Robótica e Agentes Autónomos** | **AAMAS** (Autonomous Agents and Multiagent Systems) | Currículo de aprendizagem, sistemas multi-agente e formulações CMDP |
| | **RSS** (Robotics: Science and Systems) | Políticas de manipulação contínua, sim-to-real transfer e controlo reativo |
| | **IEEE ICRA / IROS** | Aplicação prática de DRL a robôs móveis, preensão tátil e braços manipuladores |
| | **CoRL** (Conference on Robot Learning) | Fusão de perceção ativa com controlo por reforço em ambientes físicos |
| **Conferências de Visão e Domínios Específicos** | **CVPR / ICCV / ECCV** | Atenção visual recorrente (RAM), Few-Shot RL e segmentação guiada por RL |
| | **EWRL** (European Workshop on Reinforcement Learning) | Workstations avançados e desenvolvimentos metodológicos de ponta em RL |
| **Revistas Científicas de Elevado Fator de Impacto** | **JMLR** (Journal of Machine Learning Research) | Demonstrações matemáticas formais e artigos seminais de referência |
| | **IEEE T-PAMI / IEEE T-NNLS** | Métodos analíticos de estabilização de redes profundas e dinâmicas de decisão |
| | **IEEE T-RO** (Transactions on Robotics) | Engenharia de controlo avançada suportada por aprendizagem por reforço |
| **Repositórios de Preprints Científicos** | **arXiv** (`cs.LG`, `cs.AI`, `cs.RO`, `cs.CV`, `stat.ML`) | Descobertas de última hora e validação de algoritmos recentes (últimos 12-24 meses) |

---

### 2.2 Equações de Pesquisa Booleanas Recomendadas

As equações em baixo foram calibradas para execução direta nos motores académicos de referência (*Google Scholar, IEEE Xplore, ScienceDirect, ACM Digital Library, DBLP e Scopus*):

#### Bloco A: Fundamentos de Deep RL, Algoritmos Ator-Crítico e Redução de Variância
```text
("reinforcement learning" OR "deep reinforcement learning") AND ("actor-critic" OR "policy gradient" OR "Q-learning" OR "DQN") AND ("overestimation bias" OR "variance reduction" OR "stabilization") AND ("SAC" OR "PPO" OR "TD3" OR "Double DQN")
```

#### Bloco B: Otimização de Replay Buffers, Amostragem e Equivalência de Perdas
```text
("reinforcement learning" OR "deep RL") AND ("experience replay" OR "replay buffer") AND ("prioritized experience replay" OR "PER" OR "non-uniform sampling" OR "loss function" OR "target computation") AND ("sample efficiency" OR "convergence")
```

#### Bloco C: Aprendizagem por Currículo, Concepção de Recompensas e Inverse RL
```text
("curriculum learning" OR "curriculum reinforcement learning") AND ("reward function" OR "reward shaping" OR "complex rewards") AND ("inverse reward design" OR "multi-stage" OR "reward curriculum" OR "sparse reward")
```

#### Bloco D: Planeamento Baseado em Modelos, Pesquisa em Árvore e World Models
```text
("model-based reinforcement learning" OR "MCTS" OR "Monte Carlo tree search") AND ("MuZero" OR "AlphaZero" OR "learned model" OR "planning") AND ("sequential decision making" OR "benchmark")
```

#### Bloco E: Perceção Ativa, Atenção Rígida e Exploração Corporizada
```text
("reinforcement learning" OR "deep RL") AND ("active perception" OR "active sensing" OR "hard attention" OR "visual attention") AND ("predictive coding" OR "uncertainty minimization" OR "POMDP" OR "tactile")
```

#### Bloco F: RL Agêntico, Memória de Longo Prazo e Treino Sem Verificador de LLMs
```text
("reinforcement learning" OR "RL") AND ("large language models" OR "agentic memory" OR "verifier-free" OR "incentive training" OR "AutoRL") AND ("reasoning" OR "memory management" OR "long-term memory")
```

---

### 2.3 Janela Temporal de Análise

A estratégia de curadoria deve articular harmoniosamente duas dimensões temporais distintas:

1.  **Janela Seminal e Fundacional (1950 – 2017):**
    - **Objetivo:** Compreender as bases matemáticas rigorosas que sustentam toda a arquitetura de decisão sequencial e os primeiros marcos do Deep RL.
    - **Trabalhos e Marcos Indispensáveis:**
      - Bellman (1957 - Formulação da Programação Dinâmica e Equações de Bellman);
      - Sutton & Barto (1988/1998 - Aprendizagem por Diferenças Temporais e Tratado Canónico de RL);
      - Watkins & Dayan (1992 - Demonstração teórica do Q-Learning);
      - Williams (1992 - Algoritmo REINFORCE e gradientes de política analíticos);
      - Lin (1992 - Criação do mecanismo de *Experience Replay*);
      - Konda & Tsitsiklis (2000 - Fundamentação assintótica dos métodos Ator-Crítico);
      - Mnih et al. (2013/2015 - O algoritmo Nature DQN e a revolução do Deep RL no Arcade Learning Environment);
      - Mnih et al. (2014 - *Recurrent Models of Visual Attention* e otimização por PG de decisões visuais foveais);
      - Silver et al. (2016/2017 - AlphaGo e AlphaZero: fusão de MCTS com redes neuronais profundas);
      - Schaul et al. (2016 - *Prioritized Experience Replay* - PER);
      - Schulman et al. (2017 - *Proximal Policy Optimization* - PPO);
      - Zhang & Sutton (2017 - Análise crítica e falhas estruturais do Replay Buffer / proposta do CER).

2.  **Janela do Estado da Arte e Fronteira Emergente (Últimos 3 a 5 anos):**
    - **Objetivo:** Mapear os modelos de ponta focados em eficiência de amostragem, modelação de recompensas sem verificação humana, planeamento unificado e convergência com modelos fundamentais (*Foundation Models*).
    - **Marcos Recentes Relevantes:**
      - Fujimoto et al. (NeurIPS 2020 - Equivalência matemática entre funções de perda e amostragem no replay buffer / LAP e PAL);
      - Afshar et al. (2022 - Panorama geral e taxonomia de *Automated Reinforcement Learning* - AutoRL);
      - Niu et al. (NeurIPS 2023 - Plataforma LightZero e padronização unificada de MCTS/MuZero em ambientes gerais);
      - Freitag et al. (EWRL 2024 / AAMAS 2025 - Currículos de recompensa em duas etapas para funções complexas e buffers flexíveis);
      - Sharafeldin et al. (Patterns 2024 - Perceção ativa orientada por *predictive coding* e minimização de incerteza);
      - Schneider et al. (ICLR 2026 - APPLE: Estrutura generalizada de RL para perceção tátil ativa e exploração com Transformers);
      - Zhang et al. (2026 - DeltaMem: Gestão fim-a-fim de memória agêntica de longo prazo via RL);
      - Liu et al. (EMNLP 2025 - NOVER: Treino por incentivo de modelos de linguagem via RL sem necessidade de verificador externo).

---

## 3. Critérios de Elegibilidade (Inclusão e Exclusão)

Para blindar o repositório contra publicações metodologicamente frágeis ou desprovidas de valor científico comprovado, adota-se o seguinte fluxo sistemático de triagem:

```
                      Artigo Candidato Identificado na Pesquisa
                                          |
                    +---------------------+---------------------+
                    |                                           |
                    v                                           v
           Critérios de Inclusão (+)                  Critérios de Exclusão (-)
       - Protocolo multi-semente (seeds >= 5)      - Testes com semente única (sem variância)
       - Baselines canónicos (PPO, SAC, TD3, DQN)  - Baselines obsoletos ou intencionalmente fracos
       - Transparência de hiperparâmetros          - Omissão de detalhes de treino e arquitetura
       - Eficiência de amostragem detalhada        - Métricas restritas a um único ambiente não-padrão
       - Benchmarks abertos (Gym, MuJoCo, ALE)     - Opacidade metodológica e código inacessível
                    |                                           |
                    v                                           v
            ACEITE NO REPOSITÓRIO                       REJEITADO
```

### 3.1 Critérios de Inclusão (+)
O artigo deve satisfazer cumulativamente pelo menos **três** dos seguintes requisitos formais:
1.  **Validação Empírica Rigorosa com Múltiplas Sementes Aleatórias:** Demonstração de estabilidade estatística sustentada por testes em, no mínimo, $5$ sementes aleatórias (*random seeds*) distintas, apresentando curvas de aprendizagem com desvio-padrão ou intervalos de confiança sombreados.
2.  **Confrontação Direta com Baselines Reconhecidos:** Comparação explícita com os algoritmos de referência consagrados na área correspondente (ex.: contra SAC, TD3 ou PPO em controlo contínuo; contra DQN, Rainbow ou MuZero em espaços discretos; contra heurísticas clássicas de active learning).
3.  **Transparência Arquitetural e Hiperparamétrica Integral:** Detalhe exaustivo das taxas de aprendizagem do ator e do crítico, fator de desconto $\gamma$, capacidade do replay buffer, tamanho do minibatch, frequência de atualização das redes alvo ($\tau$ ou intervalo de cópia), esquemas de exploração ($\epsilon$-greedy ou coeficiente de entropia $\alpha$) e funções de recompensa exatas.
4.  **Avaliação Quantitativa da Eficiência de Amostragem (*Sample Efficiency*):** Análise explícita da velocidade de convergência em função do número total de transições do ambiente (*environment steps/timesteps*), e não apenas em função de épocas ou tempo computacional arbitrário.
5.  **Utilização de Ambientes e Benchmarks Padronizados:** Validação em plataformas públicas e reproduzíveis da comunidade (ex.: Gymnasium/OpenAI Gym, Arcade Learning Environment - ALE, MuJoCo, DeepMind Control Suite, Robosuite, MiniGrid ou benchmarks de memória como LoCoMo).

### 3.2 Critérios de Exclusão (-)
Devem ser rejeitados sumariamente estudos que apresentem qualquer uma das seguintes falhas metodológicas:
1.  **Relato de Desempenho com Semente Única (*Single Seed Reporting*):** Artigos cujos gráficos e tabelas mostram apenas uma execução arbitrária, mascarando a elevada sensibilidade e instabilidade estocástica típica do Deep RL.
2.  **Ausência ou Manipulação Desleal de Modelos de Base (*Weak/Handpicked Baselines*):** Trabalhos que omitem baselines SOTA óbvios ou que os avaliam com hiperparâmetros sub-ótimos para inflacionar artificialmente o mérito do método proposto.
3.  **Modelação de Recompensa Falaciosa (*Reward Hacking* Oculto):** Artigos que ajustam a função de recompensa exclusivamente para forçar trajetórias específicas sem realizar estudos de ablação transparentes sobre os termos da recompensa.
4.  **Vazamento de Informação Privilegiada no Ambiente (*Environment Leakage*):** Casos em que o agente tem acesso a variáveis internas do simulador não acessíveis em condições operacionais reais (ex.: acesso à velocidade perfeita de obstáculos num cenário supostamente baseado apenas em píxeis).
5.  **Estudos Conceituais Sem Prova Empírica ou Revisão Fiável:** Publicações preliminares desprovidas de testes experimentais, resumos de conferências sem avaliação técnica integral ou publicações em periódicos predatórios desprovidos de arbitragem credível.

---

## 4. Roteiro de Extração e Síntese Analítica

Para cada artigo selecionado, o agente autónomo ou investigador deve preencher obrigatoriamente a seguinte **Checklist de Avaliação Analítica em 5 Pontos**:

```
+---------------------------------------------------------------------------------+
|               CHECKLIST OBRIGATÓRIA DE EXTRAÇÃO ANALÍTICA EM RL                 |
+---------------------------------------------------------------------------------+
| 1. Problema e Motivação                                                         |
|    - Que estrangulamento específico de RL o artigo aborda?                      |
|    - Porque falham os algoritmos padrão (DQN, PPO, SAC, etc.) nesse cenário?    |
+---------------------------------------------------------------------------------+
| 2. Inovação Metodológica / Arquitetural                                         |
|    - Qual a modificação na Equação de Bellman, na função de perda ou no buffer?|
|    - Que novo enquadramento matemático, rede neuronal ou currículo é proposto?  |
+---------------------------------------------------------------------------------+
| 3. Datasets, Simuladores e Protocolo de Avaliação                               |
|    - Que ambientes de teste foram utilizados (MuJoCo, Atari, Robosuite, Gym)?  |
|    - Quantas sementes e timesteps foram executados em cada ensaio?              |
+---------------------------------------------------------------------------------+
| 4. Resultados Empíricos e Trade-offs (Precisão vs. Eficiência de Amostragem)    |
|    - Quais os ganhos absolutos no retorno acumulado ou na taxa de sucesso?      |
|    - Qual o custo em tempo de computação por iteração e memória utilizada?     |
+---------------------------------------------------------------------------------+
| 5. Limitações e Desafios em Aberto (Open Challenges)                           |
|    - Sob que condições de dinâmica ou ruído o algoritmo colapsa?               |
|    - Que hiperparâmetros permanecem excessivamente sensíveis à sintonização?    |
+---------------------------------------------------------------------------------+
```

### Detalhe Operacional de Cada Ponto da Checklist:

1.  **Problema e Motivação:**
    - Isolar o estrangulamento teórico central: sobre-estimativa de valores Q; instabilidade causada por amostragem desfasada em buffers grandes; paralisia de exploração perante recompensas esparsas; dependência de verificadores externos rígidos em LLMs; ou lentidão no treino *online* de robôs reais.
2.  **Inovação Metodológica/Arquitetural:**
    - Mapear a formulação matemática introduzida: novas perdas (como a equivalência LAP/PAL), inclusão imediata de transições recentes (CER), cálculo desacoplado de alvos (TAO-DQN), estruturas em duas etapas de currículo (RC-SAC) ou métricas de distância de edição para memória agêntica (DeltaMem).
3.  **Datasets, Simuladores e Protocolo de Avaliação:**
    - Registar o domínio de validação: nomes específicos dos ambientes de simulação (ex.: HalfCheetah-v3, Ant-v3, Breakout, Ms. Pac-Man, Franka Emika Panda Manipulator, CartPole), número de passos de interação e intervalo de confiança dos ensaios.
4.  **Resultados Empíricos e Trade-offs:**
    - Registar métricas numéricas rigorosas: percentagem de redução de timesteps para convergência, tempo total de computação (*wall-clock time*), taxas de sucesso na preensão robótica (ex.: 87% vs. 82%), e compromissos inerentes (ex.: redução de FLOPs à custa de maior consumo de memória RAM).
5.  **Limitações e Desafios em Aberto:**
    - Identificar honestamente as restrições metodológicas: dependência de demonstrações prévias de especialistas (*expert demonstrations*), degradação perante ambientes estocásticos com observação altamente ruidosa, ou necessidade de calibração manual de múltiplos pesos de funções de recompensa auxiliares.

---

## 5. Padrão de Formatação para a Tabela de Literatura (`README.md`)

O ficheiro `README.md` localizado neste diretório atua como o catálogo documental central da literatura de Reinforcement Learning. Qualquer nova entrada deve respeitar escrupulosamente a estrutura de seis colunas padronizadas.

### 5.1 Especificação Rigorosa das 6 Colunas Obrigatórias

| Coluna | Título | Tipo de Conteúdo e Regras de Formatação |
|:---:|:---|:---|
| **1** | **Nome** | Título oficial completo do artigo científico em língua inglesa, exatamente como registado na publicação original. |
| **2** | **Detalhes** | Metadados editoriais estruturados verticalmente com quebras duplas de linha em HTML (`<br><br>`), contendo obrigatoriamente: Autores, Data de publicação, Publisher, Livro/Journal/Conferência, Volume, Número (quando existir), Páginas e hiperligação ativa do DOI. |
| **3** | **Abstract** | Citações textuais integrais e literais retiradas do resumo original da publicação, delimitadas rigorosamente por aspas duplas (`"..."`). Devem espelhar a motivação, arquitetura e conclusões declaradas pelos autores. |
| **4** | **Conclusion** | Citações textuais literais extraídas da secção final de conclusões do artigo original, delimitadas rigorosamente por aspas duplas (`"..."`). |
| **5** | **Resumo (NotebookLM)** | Síntese analítica estruturada em língua portuguesa, composta imperativamente por **quatro parágrafos encadeados** separados por `<br><br>` (ver secção 5.2). |
| **6** | **Citação** | Referência bibliográfica completa em conformidade com as diretrizes da **Norma Vancouver (NLM)**, finalizada com a hiperligação HTTPS ativa e verificada do DOI. |

---

### 5.2 Estrutura Padronizada do "Resumo (NotebookLM)" (4 Parágrafos)

A coluna 5 deve obedecer impreterivelmente à seguinte arquitetura dissertativa em quatro blocos lógicos:

*   **Parágrafo 1 — Contexto e Problema de Investigação:**  
    Apresentação do artigo, contextualização do desafio em Reinforcement Learning e enunciação clara da lacuna metodológica enfrentada (ex.: ineficiência de amostragem em funções de recompensa complexas, custo proibitivo de recálculo de alvos no replay buffer, ou limites dos verificadores em LLMs).
*   **Parágrafo 2 — Mecanismo Proposto e Inovação Arquitetural:**  
    Descrição aprofundada da técnica matemática ou estrutural concebida (ex.: currículo de recompensa em duas etapas, modificação do operador de amostragem, integração de auto-atenção foveal em POMDPs ou mecanismo de recompensa sem verificador).
*   **Parágrafo 3 — Protocolo Experimental e Resultados Quantitativos:**  
    Discriminação dos ambientes e benchmarks utilizados (ambientes do Gym, simuladores MuJoCo, robôs reais ou conjuntos de dados de memória), baselines concorrentes e indicação expressa dos resultados numéricos alcançados (taxas de sucesso, redução de episódios para convergência, retornos acumulados).
*   **Parágrafo 4 — Significado Teórico, Limitações e Desdobramentos:**  
    Avaliação crítica do impacto para a disciplina, viabilidade prática em sistemas reais (*sim-to-real* ou inferência de agentes) e desafios em aberto identificados pelos investigadores.

---

### 5.3 Exemplo Canónico de Registo na Tabela

Apresenta-se seguidamente o modelo de referência absoluta extraído diretamente do registo existente no `README.md` deste repositório, correspondente ao trabalho de Freitag et al. (AAMAS 2025):

```markdown
| Sample-efficient curriculum reinforcement learning for complex reward functions | **Autores:** Kilian Freitag, Kristian Ceder, Rita Laezza, Knut Åkesson, Morteza Haghir Chehreghani<br><br>**Data de publicação:** 2024<br><br>**Publisher:** International Foundation for Autonomous Agents and Multiagent Systems<br><br>**Livro/Journal:** Proceedings of the 24th International Conference on Autonomous Agents and Multiagent Systems (AAMAS 2025)<br><br>**DOI:** https://doi.org/10.48550/arXiv.2410.16790 | "Reinforcement learning (RL) shows promise in control problems, but its practical application is often hindered by the complexity arising from intricate reward functions with constraints."<br><br>"To mitigate reward exploitation in such complex settings, we propose a novel two-stage reward curriculum combined with a flexible replay buffer that adaptively samples experiences."<br><br>"Our approach first learns on a subset of rewards before transitioning to the full reward, allowing the agent to learn trade-offs between objectives and constraints."<br><br>"After transitioning to a new stage, our method continues to make use of past experiences by updating their rewards for sample-efficient learning." | "In this work, we introduced a novel two-stage reward curriculum that enables sample-efficient learning for complex rewards."<br><br>"Our approach demonstrates the importance of carefully selecting the initial subset of rewards and reusing experiences between curriculum phases, highlighting the need for a nuanced understanding of reward function design and reward curricula."<br><br>"Through our experiments, we showcased the efficacy of our approach in learning complex policies and underscored the potential benefits of reward curricula in improving learning efficiency and stability."<br><br>"The promising results provide a strong base for further developments of more sophisticated reinforcement learning algorithms that can tackle complex real-world problems." | Este artigo introduz uma nova estrutura de aprendizagem por currículo de recompensa em duas etapas (*two-stage reward curriculum*) combinada com um *buffer* de *replay* flexível, concebida para otimizar funções de recompensa complexas e multifacetadas em Aprendizagem por Reforço (*Reinforcement Learning* - RL) sem incorrer em exploração indesejada de recompensas (*reward exploitation*) ou estagnação em sub-ótimos locais.<br><br>A metodologia desdobra-se nos seguintes pilares centrais: 1. Conceção de Recompensas e Seleção de Subconjunto Inicial: O artigo formula uma função de recompensa para navegação de robôs móveis com restrições (*soft constraints* como velocidade de referência e *hard constraints* como colisões). O agente começa por otimizar apenas um subconjunto simplificado da recompensa ($r_{gpv}$), prevenindo que penalizações de restrições dominem precocemente a aprendizagem; 2. Buffer de Replay Flexível e Transferência de Experiência: Na transição para a segunda etapa ($r_{full}$), o enquadramento RC-SAC reutiliza todas as transições recolhidas no *replay buffer* com valores de recompensa atualizados, executando passos adicionais de gradiente *off-policy* para adaptação estável; 3. Momento Ideal de Transição: Os autores demonstram que iniciar a segunda fase após a convergência da perda da temperatura de entropia ($J(\alpha)$) maximiza a taxa de sucesso.<br><br>Avaliado em simulações de navegação de robôs móveis com obstáculos estáticos e dinâmicos, o algoritmo RC-SAC aumentou a taxa de sucesso de 42% (da *baseline* sem currículo) para 66%, obtendo maior recompensa real (*true reward*) e demonstrando superioridade perante currículos ingénuos que descartam a experiência passada.<br><br>O trabalho salienta o papel fundamental da seleção criteriosa do subconjunto inicial de incentivos e do reaproveitamento contínuo de transições passadas, evidenciando que abordagens de currículo de recompensa são vitais para viabilizar o controlo robótico por RL em ambientes reais. | Freitag K, Ceder K, Laezza R, Åkesson K, Chehreghani MH. Sample-efficient curriculum reinforcement learning for complex reward functions. In: Proceedings of the 24th International Conference on Autonomous Agents and Multiagent Systems (AAMAS 2025). 2024. https://doi.org/10.48550/arXiv.2410.16790 |
```

---

## 6. Guardrails de Qualidade & Anti-Alucinação

Para assegurar a máxima integridade metodológica e fiabilidade documental do projeto, investigadores humanos e agentes autónomos devem operar sob estrita observância das quatro diretrizes infra:

### 6.1 Validação Rigorosa de DOIs e Metadados Catalográficos
- **Verificação Ativa da Resolução do DOI:** Todas as hiperligações inseridas devem estar no formato canónico padrão `https://doi.org/10.xxxx/...` e ser ativamente verificáveis perante os servidores da CrossRef ou repositórios oficiais. É terminantemente proibido conjeturar ou forjar prefixos ou sufixos de DOI.
- **Auditoria Cruzada de Metadados:** Os nomes dos autores, ano de publicação, título oficial do artigo e páginas devem ser confrontados diretamente com fontes de indexação internacional consolidadas (*DBLP, Google Scholar, Semantic Scholar, IEEE Xplore, ACM Digital Library ou páginas oficiais do editor*).

### 6.2 Proibição Estrita de Interpolação ou Arredondamento de Métricas
- **Fidelidade Numérica Absoluta:** Todas as pontuações e grandezas empíricas devem ser extraídas exatamente como expressas no manuscrito original. Se um artigo reporta uma taxa de sucesso de $87.0\%$ com intervalo de confiança de $[86.1, 87.9]$ e superação de baselines em $15\%$, é expressamente proibido converter estes valores para "cerca de 90%" ou omitir desvios-padrão conhecidos.
- **Discriminação Rigorosa de Métricas:** Devem ser mantidas as distinções estritas entre métricas de treino e teste (*evaluation return* vs. *training episode reward*), timesteps de ambiente vs. atualizações de gradiente, e médias pontuais vs. medianas interquartis (*interquartile mean - IQM*).

### 6.3 Fidelidade Literal das Citações Diretas
- Os campos **Abstract** e **Conclusion** da tabela devem conter excertos ipsis verbis entre aspas duplas (`"..."`), extraídos diretamente do documento original. É proibido efetuar paráfrases livres, emendar frases de secções distintas sem marcação formal, ou introduzir adjetivações e interpretações do curador dentro das colunas de citação direta.

### 6.4 Sobriedade Terminológica e Neutralidade Epistemológica
- As análises dissertativas no "Resumo (NotebookLM)" devem primar pela neutralidade descritiva, utilizando terminologia científica rigorosa e evitando afirmações promocionais desmedidas (ex.: banir termos como "algoritmo revolucionário", "solução mágica", "superação avassaladora").
- O foco deve manter-se sempre nos compromissos de engenharia (*trade-offs* entre estabilidade e velocidade de aprendizagem, complexidade de amostragem versus custo computacional) e na identificação transparente das limitações operacionais da arquitetura.
