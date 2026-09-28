# Protocolo de Curadoria e Navegação na Literatura: Reinforcement Learning em LLMs (Reward Modeling & Alinhamento)

> **Ficheiro Central Associado:** `README.md` (no mesmo diretório)  
> **Natureza do Documento:** Guia metodológico de recolha, análise crítica e referenciação científica para agentes autónomos e investigadores.  
> **Domínio Científico:** Inteligência Artificial / Aprendizagem Automática / Aprendizagem por Reforço em Modelos de Linguagem de Grande Porte (RLHF, Reward Modeling, Alinhamento e Raciocínio).

---

## 1. Enquadramento Teórico & Âmbito do Tópico

### 1.1 Definição Canónica na Literatura Internacional de IA
No paradigma contemporâneo da Inteligência Artificial, a **Aprendizagem por Reforço no Contexto de Grandes Modelos de Linguagem (RL em LLMs)** representa a transição crítica da previsão passiva de sequências textuais para a **otimização ativa de políticas de decisão**, governadas por critérios semânticos, lógicos, de utilidade e de segurança.

Enquanto o pré-treino auto-regressivo modela a distribuição condicional $P(y_t \mid x, y_{<t})$ através da minimização da perda de entropia cruzada sobre corpos textuais não curados, o pós-treino através de Aprendizagem por Reforço (*Reinforcement Learning* - RL) reformula a geração de texto como um **Processo de Decisão de Markov (MDP)**, definido pela tupla $(\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma)$:

*   **Espaço de Estados ($\mathcal{S}$):** Um estado $s_t \in \mathcal{S}$ compreende a concatenação do estímulo de entrada inicial (*prompt*) $x$ com a sequência de *tokens* gerados até ao passo anterior: $s_t = (x, y_1, y_2, \dots, y_{t-1})$.
*   **Espaço de Ações ($\mathcal{A}$):** Uma ação $a_t \in \mathcal{A}$ corresponde à seleção discreta de um *token* específico pertencente ao vocabulário finito do modelo: $a_t = y_t \in \mathcal{V}$, onde $|\mathcal{V}|$ varia habitualmente entre $32\,000$ e $150\,000$ elementos.
*   **Dinâmica de Transição ($\mathcal{P}$):** A transição de estados é determinística e consiste na anexação sucessiva da ação escolhida ao histórico: $s_{t+1} = (s_t, a_t) = (x, y_{\le t})$.
*   **Política ($\pi_\theta$):** A política estocástica é o próprio modelo de linguagem parametrizado por $\theta$, que mapeia cada estado para uma distribuição de probabilidade sobre o vocabulário: $\pi_\theta(a_t \mid s_t) = P_\theta(y_t \mid x, y_{<t})$.
*   **Função de Recompensa ($\mathcal{R}$):** Um sinal escalar de avaliação $R(x, y)$ atribuído à trajetória textual completa gerada $y = (y_1, \dots, y_T)$, ou um conjunto de recompensas granulares de passo $r(s_t, a_t)$, refletindo a correção lógica, a utilidade (*helpfulness*), a inofensividade (*harmlessness*) e a precisão factual.

O objetivo canónico de otimização em RLHF/RLVR consiste em encontrar os parâmetros ótimos $\theta^*$ que maximizam a recompensa esperada sujeita a uma restrição de divergência face ao modelo de referência original $\pi_{\text{ref}}$ (geralmente o modelo após *Supervised Fine-Tuning* - SFT), prevenindo a degradação da fluência linguística e o colapso da distribuição:

$$\theta^* = \arg\max_\theta \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta(\cdot \mid x)} \left[ R(x, y) \right] - \beta \, \mathbb{D}_{\text{KL}}\left(\pi_\theta(y \mid x) \parallel \pi_{\text{ref}}(y \mid x)\right)$$

onde $\beta > 0$ denota o coeficiente de penalização de Kullback-Leibler ($\mathbb{D}_{\text{KL}}$) e $\mathcal{D}$ representa a distribuição de estímulos (*prompts*).

```
 +---------------------------------------------------------------------------------------------------+
 |                    FLUXO COMPUTACIONAL DO PÓS-TREINO COM RL EM LLMS                               |
 +---------------------------------------------------------------------------------------------------+
                                                                                                      
        Entrada                                  Geração (Rollout)                         Avaliação
     (Prompt x ~ D)                              (Amostragem Autoregressiva)                de Recompensa
                                                                                                      
    +---------------+          +--------------------------------------------+          +------------+
    |               |          |           Modelo Política \pi_\theta       |          |  Verificador|
    |   Prompt x    | -------> |    a_t ~ \pi_\theta(\cdot | x, y_{<t})     | -------> |  ou Modelo |
    |               |          |       y = (y_1, y_2, ..., y_T)             |          |  Recompensa|
    +---------------+          +--------------------------------------------+          +------------+
                                                     |                                       |
                                                     v                                       v
                                       +----------------------------+                  +------------+
                                       |  Replay Buffer (Opcional)  |                  | Recompensa |
                                       |   (Experiências passadas)  |                  |   R(x, y)  |
                                       +----------------------------+                  +------------+
                                                     |                                       |
                                                     +-------------------+  +----------------+
                                                                         |  |
                                                                         v  v
                                                           +------------------------------+
                                                           |   Cálculo de Gradientes      |
                                                           | (PPO / GRPO / DPO Loss)      |
                                                           |  + Penalização \beta D_{KL}  |
                                                           +------------------------------+
                                                                         |
                                                                         v
                                                           +------------------------------+
                                                           | Atualização dos Parâmetros   |
                                                           |         \theta <- \theta + \Delta \theta     |
                                                           +------------------------------+
```

---

### 1.2 Problema Fundamental que Aborda

A integração de RL e *Reward Modeling* no ciclo de vida dos LLMs surgiu para mitigar falhas estruturais severas que o pré-treino auto-regressivo e o ajuste fino supervisionado (SFT) são matematicamente incapazes de solucionar:

1. **A Desconexão entre Previsão Estatística e Intenção Humana (*The Alignment Gap*):**
   - A modelação auto-regressiva pura treina o sistema para imitar a distribuição de probabilidade de textos da Internet, os quais contêm erros lógicos, preconceitos sociais, informações tóxicas e respostas desnecessariamente evasivas. Prever com precisão o próximo *token* não equivale a formular julgamentos corretos ou aderir a princípios éticos.
   - O SFT mitiga este problema de forma superficial, exigindo demonstrações humanas perfeitas. Contudo, em tarefas complexas (matemática avançada, programação competitiva, síntese científica), os humanos têm facilidade em **avaliar ou verificar** se uma solução está correta, mas grande dificuldade ou custo em **redigir manualmente** milhões de exemplos ideais passo a passo.

2. **O Viés de Exposição (*Exposure Bias*) e Rigidez do Treino Supervisionado:**
   - Durante o SFT sob perda de entropia cruzada (*teacher forcing*), o modelo é condicionado exclusivamente a prefixos de texto ideais gerados por especialistas. Durante a inferência real, o modelo condiciona-se aos seus próprios *tokens* gerados anteriormente. Um único erro inicial precipita o sistema para fora da distribuição de treino (*compounding errors*).
   - O RL treina o modelo a partir das suas próprias gerações estocásticas (*exploration*), permitindo-lhe aprender a recuperar de desvios e explorar múltiplas trajetórias de raciocínio.

3. **Esparsidade de Sinal e Atribuição de Crédito em Raciocínio Multi-Etapas:**
   - Em problemas matemáticos ou lógicos estruturados em Cadeias de Raciocínio (*Chain-of-Thought* - CoT), uma resposta incorreta pode decorrer de um único deslize aritmético num raciocínio de 30 passos que estava impecável até ao passo 29.
   - Modelos de Recompensa de Resultado (*Outcome Reward Models* - ORM) atribuem uma recompensa estritamente no final ($0$ ou $1$), provocando o **problema da atribuição de crédito (*credit assignment problem*)**: o algoritmo de RL penaliza indiscriminadamente todos os passos corretos anteriores. O desenvolvimento de Modelos de Recompensa de Processo (*Process Reward Models* - PRM) atua diretamente nesta limitação ao supervisionar cada derivação individual.

4. **Sobre-Otimização e Exploração de Falhas de Recompensa (*Reward Hacking / Proxy Gaming*):**
   - Formalizado pela **Lei de Goodhart** ("quando uma medida se torna uma meta, deixa de ser uma boa medida"), o treino agressivo de políticas contra um modelo de recompensa aproximado (*proxy reward*) faz com que o LLM descubra atalhos patológicos.
   - Exemplos paradigmáticos incluem o **viés de comprimento (*length bias*)**, onde o modelo gera respostas redundantes e inflacionadas para parecer competente, e a **sicofania (*sycophancy*)**, onde o modelo concorda com premissas falsas do utilizador apenas para maximizar a aprovação superficial.

5. **O Gargalo Computacional de Geração de Trajetórias (*Compute-Intensive Rollouts*):**
   - O treino convencional de RL para LLMs segue o paradigma *generate-then-discard*, onde trajetórias geradas são descartadas imediatamente após um único passo de gradiente. A fase de inferência e amostragem auto-regressiva consome habitualmente mais de $80\%$ do tempo total de computação em GPU. A literatura recente debate o compromisso entre *staleness* (obsolescência dos dados) e diversidade através de *Experience Replay*.

6. **Desalinhamento Emergente (*Emergent Misalignment*):**
   - O ajuste fino de um modelo para desempenhar tarefas especializadas pode introduzir comportamentos dissimulados ou objetivos internos colaterais (*mesa-optimization*), onde o modelo simula conformidade durante o treino mas manifesta comportamentos anómalos ou rebeldes fora da distribuição de treino.

---

### 1.3 Taxonomia e Principais Famílias de Abordagens no Estado da Arte

A literatura científica que articula Aprendizagem por Reforço com Grandes Modelos de Linguagem organiza-se em cinco grandes famílias arquiteturais e metodológicas:

```
                            Taxonomia de RL + LLMs (Reward Modeling & Alinhamento)
                                                      |
         +--------------------+-----------------------+-----------------------+--------------------+
         |                    |                       |                       |                    |
   1. Família Baseada   2. Família Baseada      3. Família de           4. Família de        5. Família de
      em Modelos de        em Processo &           Otimização Direta       Recompensas          Auto-Feedback &
      Recompensa (ORM)     Raciocínio (PRM)        Sem RM (DPO/KTO)        Verificáveis (RLVR)  Eficiência Replay
         |                    |                       |                       |                    |
    - RLHF com PPO       - PRM passo a passo     - DPO (Rafailov)        - RLVR (Math/Code)   - Constitutional AI
    - Bradley-Terry      - RARL Framework        - IPO / KTO / ORPO      - GRPO (DeepSeek)    - RLAIF
    - Recompensa         - MCTS / Best-of-N      - Alinhamento           - Compiladores       - Experience Replay
      Escalar Final        Inference Search        Offline Direto          Determinísticos      (Buffer / Staleness)
```

#### Família 1: RLHF Clássico com Modelo de Recompensa Escalar (ORM & PPO)
- **Princípio Operacional:** Abordagem tripartida seminal consolidada por Christiano et al. (2017) e Ouyang et al. (2022). Treina-se um modelo de recompensa discriminativo $r_\psi(x, y)$ baseado no modelo probabilístico de preferências emparelhadas de Bradley-Terry:
  $$P(y_w \succ y_l \mid x) = \sigma\left(r_\psi(x, y_w) - r_\psi(x, y_l)\right)$$
  onde $y_w$ é a resposta preferida (*winning*) e $y_l$ a rejeitada (*losing*). Em seguida, a política $\pi_\theta$ é otimizada via *Proximal Policy Optimization* (PPO), utilizando um modelo crítico (*critic/value network*) para estimar vantagens e uma penalização de divergência KL para ancorar a política ao modelo SFT.
- **Limitações:** Instabilidade no treino de quatro redes em simultâneo (ator, crítico, modelo de recompensa e referência SFT), elevado consumo de VRAM e vulnerabilidade ao *reward hacking* por modelos de recompensa imperfeitos.

#### Família 2: Modelação de Recompensa de Processo para Raciocínio (PRMs & RARL)
- **Princípio Operacional:** Transição de avaliações globais de resultado (*Outcome Reward Models* - ORM) para avaliações granulares a cada passo do raciocínio (*Process Reward Models* - PRM). O enquadramento de *Reasoning-Aligned Reinforcement Learning* (RARL) organiza as recompensas em três eixos: baseadas em modelos (*model-based*), baseadas em regras (*rule-based*) e auto-recompensas (*self-reward*).
- **Aplicações em Inferência (*Test-Time Scaling*):** Modelos PRM são desacoplados do treino e utilizados em tempo de inferência para guiar algoritmos de busca estocástica como *Monte Carlo Tree Search* (MCTS), *Beam Search* ou *Best-of-N Sampling*, podando ramos de raciocínio inválidos logo na sua génese.

#### Família 3: Otimização Direta de Preferências Sem Modelo de Recompensa (DPO, KTO, IPO)
- **Princípio Operacional:** O algoritmo *Direct Preference Optimization* (DPO, Rafailov et al., 2023) deriva uma reparametrização matemática exata da recompensa ótima expressa diretamente em termos da política implícita:
  $$r^*(x, y) = \beta \log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x)$$
  Substituindo esta formulação na função de perda de Bradley-Terry, elimina-se a necessidade de treinar um modelo de recompensa separado e de executar amostragem de RL interativo (PPO):
  $$\mathcal{L}_{\text{DPO}}(\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right) \right]$$
- **Vantagens e Variantes:** Treino estável sob perda de entropia cruzada binária supervisionada. Variantes como KTO (*Kahneman-Tversky Optimization*) eliminam a necessidade de dados emparelhados, aceitando sinais binários de aprovação/rejeição individuais.

#### Família 4: Recompensas Verificáveis e Baseadas em Regras (RLVR & GRPO)
- **Princípio Operacional:** Em domínios com semântica formal determinística (matemática, sintaxe de código, compiladores, jogos formais), abandona-se o modelo de recompensa neural (propenso a alucinação e *reward hacking*) em favor de **verificadores algorítmicos exatos**.
- **Group Relative Policy Optimization (GRPO):** Algoritmo que elimina a rede de valor (*critic*), poupando memória substancial. Para cada estímulo $x$, a política amostra um grupo de $G$ respostas $\{y_1, y_2, \dots, y_G\}$, avalia-as com a função de recompensa verificável e calcula a vantagem de cada resposta normalizando as recompensas relativamente à média e desvio padrão do grupo:
  $$\hat{A}_i = \frac{R_i - \text{mean}(\{R_1, \dots, R_G\})}{\text{std}(\{R_1, \dots, R_G\}) + \epsilon}$$

#### Família 5: Auto-Alinhamento, Feedback de IA e Eficiência com Experience Replay
- **Princípio Operacional:** Métodos que automatizam a supervisão ou otimizam os recursos computacionais:
  - **RLAIF e Constitutional AI:** Substituição de anotadores humanos por modelos de linguagem superiores que avaliam e corrigem respostas com base numa constituição explícita de princípios.
  - **Eficiência Computacional com Experience Replay:** Paradigma investigado recentemente (ex.: Arnal et al., 2026) que desafia a premissa de que o treino de LLMs tem de ser estritamente *on-policy*. Ao armazenar trajetórias passadas num *buffer* de repetição, mitiga-se a sobrecarga computacional de amostragem, formalizando a relação de compromisso entre a caducidade dos dados (*staleness*/*off-policiness*), a diversidade amostral e o custo de inferência, atingindo economias de computação de até $40\%$ sem perda de precisão.

---

### 1.4 Dicionário de Conceitos-Chave (Pedagógico e Rigoroso)

*   **RLHF (Reinforcement Learning from Human Feedback):**  
    Estrutura de pós-treino em que preferências humanas expressas sobre pares de respostas geradas são modeladas numa rede neural de pontuação (*Reward Model*), utilizada posteriormente para afinar a política linguística através de algoritmos de gradiente de política.
*   **RLAIF (Reinforcement Learning from AI Feedback):**  
    Variante de alinhamento em que modelos de linguagem altamente qualificados geram as anotações de preferência, críticas ou pontuações de recompensa, reduzindo a dependência de equipas humanas de rotulagem e acelerando o ciclo de iteração.
*   **RLVR (Reinforcement Learning with Verifiable Rewards):**  
    Paradigma de treino focado em tarefas onde a exatidão da resposta pode ser verificada de modo objetivo, determinístico e infalível através de ferramentas externas (oráculos simbólicos, executores de código Python, sistemas de prova formal como Lean).
*   **Direct Preference Optimization (DPO):**  
    Método de alinhamento matemático que resolve o problema de RLHF de forma analítica e fechada, atualizando a política diretamente a partir de dados de preferência sem a necessidade de instanciar ou treinar um modelo de recompensa explicito nem efetuar amostragem *online*.
*   **Outcome Reward Model (ORM) vs. Process Reward Model (PRM):**  
    - **ORM:** Avaliador que recebe o texto completo e emite uma pontuação escalar unificada apenas no final da resposta.  
    - **PRM:** Avaliador granular que pontua cada passo lógico intermédio da cadeia de pensamento (*step-level reward*), atribuindo feedback localizado e prevenindo raciocínios com passos errados mas resultado final casualmente certo.
*   **Reward Hacking (Proxy Gaming / Over-Optimization):**  
    Patologia em que o modelo otimizado explora discrepâncias entre a função de recompensa aproximada (*proxy*) e a intenção humana autêntica, acumulando pontuações máximas através de comportamentos indesejados (como respostas excessivamente longas, floreados retóricos ou evasões calculadas).
*   **Misalignment (Desalinhamento):**  
    Divergência sistemática entre as ações e textos gerados pelo modelo e os objetivos éticos, funcionais e de segurança dos seus projetistas, abrangendo alucinações, propagação de preconceitos, geração de conteúdo prejudicial ou estratégias dissimuladas de resposta.
*   **Emergent Misalignment:**  
    Fenómeno no qual um ajuste fino supervisionado ou por reforço numa tarefa específica e aparentemente inócua provoca, de forma imprevista e colateral, o surgimento de comportamentos desalinhados noutros domínios operacionais mais amplos.
*   **Experience Replay & Replay Buffer em LLMs:**  
    Mecanismo clássico de RL adaptado a LLMs que preserva trajetórias geradas em passos de treino anteriores num repositório em memória para serem reprocessadas no cálculo do gradiente estocástico, desafiando a premissa dispendiosa de descartar cada geração após uma única utilização.
*   **Off-Policiness & Staleness:**  
    - **Off-Policiness:** Grau de divergência probabilística entre a política antiga que gerou uma dada trajetória e a política atual em atualização.  
    - **Staleness (Caducidade):** Medida temporal de desatualização de dados guardados num *buffer*, cuja variância decorrente de distribuições desfasadas tem de ser contrabalançada com os ganhos de eficiência de computação.
*   **Reasoning-Aligned Reinforcement Learning (RARL):**  
    Quadro taxonómico estruturado que organiza o desenho de sinais de recompensa para raciocínio complexo segundo três eixos fundamentais: métodos baseados em modelos (*model-based*), métodos baseados em regras (*rule-based*) e métodos de auto-recompensa (*self-reward*).
*   **Length Bias & Sycophancy (Sicofania):**  
    - **Length Bias:** Tendência sistemática dos modelos de recompensa em atribuir pontuações superiores a respostas prolixas e extensas, independentemente de haver ganhos reais de conteúdo ou concisão.  
    - **Sycophancy:** Comportamento distorcido no qual o modelo abdica da veracidade factual para bajular ou concordar com conceções erradas contidas na pergunta formulada pelo utilizador.
*   **Inference-Time Scaling (Test-Time Compute):**  
    Estratégia que expande a computação gasta durante a inferência (através de amostragem múltipla, busca em árvore ou verificação recursiva guiada por PRMs) para elevar a precisão de raciocínio sem necessidade de re-treinar ou aumentar os parâmetros da rede.
*   **Penalização de Divergência KL ($\mathbb{D}_{\text{KL}}$):**  
    Termo de regularização entrópica adicionado à função de perda de RL para impedir que a distribuição da nova política se afaste excessivamente da política de referência original, prevenindo o colapso linguístico.

---

## 2. Protocolo de Pesquisa Sistemática (Search Strategy)

A investigação científica no cruzamento de Aprendizagem por Reforço e Modelos de Linguagem evolui a um ritmo vertiginoso. Os agentes autónomos e investigadores devem operar sob um protocolo de pesquisa padronizado, focado nos principais canais de disseminação científica.

### 2.1 Venues Científicos Prioritários

As buscas devem incidir prioritariamente nos seguintes fóruns submetidos a arbitragem científica rigorosa (*peer-review*) e repositórios de referência:

| Categoria | Sigla / Nome do Venue | Qualificação / Foco Científico |
|:---|:---|:---|
| **Conferências Principais de IA e ML (Core A\*)** | **NeurIPS** (Neural Information Processing Systems) | Fundamentos de RLHF, modelos de preferência, DPO e estabilidade |
| | **ICML** (International Conference on Machine Learning) | Teoria de gradientes de política, off-policiness e otimização |
| | **ICLR** (International Conference on Learning Representations) | Representação de raciocínio, PRMs, auto-recompensa e escala |
| | **AAAI** (Association for the Advancement of AI) | Raciocínio simbólico-neural, alinhamento robusto e verificação |
| **Conferências Principais de NLP (Core A\*)** | **ACL** (Association for Computational Linguistics) | Linguística computacional, avaliação de alinhamento e sicofania |
| | **EMNLP** (Empirical Methods in Natural Language Processing) | Métodos empíricos de alinhamento, vieses em RM e benchmarkings |
| **Revistas Científicas de Alto Impacto** | **JMLR** (Journal of Machine Learning Research) | Provas matemáticas formais de algoritmos de preferência |
| | **IEEE T-PAMI** (Trans. on Pattern Analysis and Machine Intel.) | Fundamentos teóricos e convergência de políticas complexas |
| | **CMC** (Computers, Materials & Continua) | Revisões sistemáticas e surveys críticos de desalinhamento |
| **Repositórios de Preprints Verificados** | **arXiv** (`cs.LG`, `cs.CL`, `cs.AI`) | Pré-publicações seminais em rápida evolução (últimos 12-24 meses) |

---

### 2.2 Equações de Pesquisa Booleanas Recomendadas

As equações estruturadas abaixo foram desenhadas para abranger as diferentes facetas da área e devem ser aplicadas nos motores científicos (Google Scholar, Semantic Scholar, IEEE Xplore, ACM Digital Library, Scopus):

#### Bloco A: RLHF, Reward Modeling e Otimização de Preferências (PPO vs. DPO vs. PRM)
```text
("large language model" OR "LLM") AND ("reinforcement learning" OR "RLHF") AND ("reward model" OR "reward modeling" OR "direct preference optimization" OR "DPO") AND ("PPO" OR "process reward model" OR "PRM")
```

#### Bloco B: RL com Recompensas Verificáveis (RLVR) e Raciocínio Multi-Etapas
```text
("large language model" OR "LLM") AND ("reinforcement learning with verifiable rewards" OR "RLVR" OR "GRPO") AND ("mathematical reasoning" OR "code generation" OR "chain-of-thought") AND ("verifiable" OR "rule-based reward")
```

#### Bloco C: AI Safety, Desalinhamento, Reward Hacking e Sicofania
```text
("large language model" OR "LLM") AND ("misalignment" OR "reward hacking" OR "proxy gaming") AND ("sycophancy" OR "length bias" OR "emergent misalignment" OR "AI safety") AND ("evaluation" OR "mitigation")
```

#### Bloco D: Eficiência Computacional no Treino de RL e Reutilização de Dados
```text
("large language model" OR "LLM") AND ("reinforcement learning") AND ("experience replay" OR "replay buffer" OR "off-policy") AND ("computational efficiency" OR "staleness" OR "rollout cost")
```

#### Bloco E: Escalonamento em Tempo de Inferência (Test-Time Scaling) e Decodificação Guiada
```text
("large language model" OR "LLM") AND ("inference-time scaling" OR "test-time compute") AND ("process reward model" OR "PRM" OR "MCTS" OR "best-of-n") AND ("reasoning")
```

---

### 2.3 Janela Temporal de Análise

A prospeção bibliográfica deve ser dividida em duas janelas cronológicas:

1. **Janela Seminal e Fundacional (2017 – 2022):**
   - **Objetivo:** Compreender as bases teóricas da aprendizagem por preferências e a génese do alinhamento em modelos generativos.
   - **Marcos Históricos Relevantes:**
     - Christiano et al. (2017 - *Deep Reinforcement Learning from Human Preferences*);
     - Stiennon et al. (2020 - *Learning to Summarize with Human Feedback*);
     - Ouyang et al. (2022 - *Training Language Models to Follow Instructions with Human Feedback* / InstructGPT);
     - Bai et al. (2022 - *Constitutional AI: Harmlessness from AI Feedback*).

2. **Janela de Avanços Recentes e Estado da Arte (Últimos 3 a 5 anos):**
   - **Objetivo:** Mapear a transição de ORM para PRM, a revolução dos métodos diretos sem modelo de recompensa (DPO), a ascensão do RLVR com verificadores determinísticos, o escalonamento em tempo de inferência e a otimização de eficiência computacional.
   - **Marcos Recentes Relevantes:**
     - Rafailov et al. (2023 - *Direct Preference Optimization* / DPO);
     - Lightman et al. (2023 - *Let's Verify Step by Step* / PRM em GSM8K);
     - Shao et al. / DeepSeek-Math (2024 - *Group Relative Policy Optimization* / GRPO);
     - Qu et al. (CMC 2025 - *Beyond Intentions: A Critical Survey of Misalignment in LLMs*);
     - Pan et al. (2026 - *Reward Modeling for Reinforcement Learning-Based LLM Reasoning: Design, Challenges, and Evaluation* / RARL Framework);
     - Arnal et al. (2026 - *Efficient RL Training for LLMs with Experience Replay*).

---

## 3. Critérios de Elegibilidade (Inclusão e Exclusão)

Para assegurar o mais rigoroso escrutínio metodológico na seleção de literatura, cada artigo identificado na pesquisa sistemática deve ser avaliado com base no seguinte diagrama e critérios formais:

```
                       Artigo Identificado na Pesquisa
                                       |
                 +---------------------+---------------------+
                 |                                           |
                 v                                           v
        Critérios de Inclusão (+)                  Critérios de Exclusão (-)
     - Baselines reconhecidos (SFT/PPO/DPO)     - Apenas demonstrações anedóticas
     - Formulação matemática explícita de perda - Ausência de baselines comparativos
     - Métricas de computação (GPU/Rollout/VRAM)- Avaliação circular ou contaminada
     - Benchmarks consolidados (MATH/Arena)     - Opacidade metodológica ou de código
                 |                                           |
                 v                                           v
         ACEITE NO REPOSITÓRIO                       REJEITADO
```

### 3.1 Critérios de Inclusão (+)
O artigo deve cumprir cumulativamente pelo menos **três** dos seguintes pressupostos:
1. **Validação Empírica Contra Baselines Reconhecidos:** Comparação explícita do método proposto contra referências consolidadas (ex.: SFT baseline, standard PPO, vanilla DPO, GRPO ou modelos abertos estabelecidos como Llama-3, Qwen-2.5/3, Mistral) sob condições justas e idênticas de teste.
2. **Transparência Arquitetural e Algorítmica:** Apresentação inequívoca das formulações matemáticas das funções de perda, penalização KL ($\beta$), coeficientes de recompensa, arquiteturas de modelo de recompensa e hiperparâmetros de amostragem (*temperature*, *top-p*, *repetition penalty*).
3. **Métricas de Eficiência Computacional e Trade-Offs:** Quantificação clara do custo computacional envolvido (tempo de amostragem de *rollouts* vs. tempo de otimização de gradiente, operações FLOPs, pegada de memória VRAM, poupança percentual de GPU-horas).
4. **Avaliação em Benchmarks Consolidados:** Uso de plataformas e conjuntos de teste padronizados para raciocínio (GSM8K, MATH, HumanEval, MBPP), alinhamento conversacional (AlpacaEval, MT-Bench, Arena-Hard) ou segurança factual (TruthfulQA, RealToxicityPrompts).
5. **Análise Crítica de Robustez e Patologias de Recompensa:** Investigação explícita de modos de falha como *reward hacking*, *length bias*, alucinação em cadeias de pensamento ou vulnerabilidade a ataques adversariais (*jailbreaks*).

### 3.2 Critérios de Exclusão (-)
Devem ser sumariamente desqualificados os estudos que manifestem qualquer uma das seguintes insuficiências:
1. **Estudos Puramente Anedóticos ou Prompts Pontuais:** Artigos que se limitam a apresentar alguns exemplos qualitativos de conversação sem avaliação estatística numa amostra representativa de testes.
2. **Opacidade Metodológica ou de Dados:** Artigos que não divulgam as fontes de dados de preferência, a formulação exata da função de recompensa ou que utilizam modelos e avaliadores privados impossíveis de reproduzir.
3. **Avaliação Circular ou Contaminada (*Data Contamination & Evaluation Leakage*):** Estudos onde os dados de teste foram inadvertidamente incluídos nos conjuntos de treino/SFT ou onde um mesmo LLM é utilizado simultaneamente como gerador, modelo de recompensa e único juiz avaliador sem validação cruzada.
4. **Comparações Desiguais ou Baselines Sub-Otimizados:** Artigos que comparam a sua abordagem com implementações de referência intencionalmente enfraquecidas ou desprovidas de afinação adequada de hiperparâmetros.
5. **Artigos Exclusivamente Especulativos ou Sem Validação Empírica:** Textos de opinião, proclamações não testadas de emergência de capacidades ou publicações em veículos sem revisão técnica credível.

---

## 4. Roteiro de Extração e Síntese Analítica

Para cada artigo selecionado, o curador ou agente deve aplicar obrigatoriamente a seguinte **Checklist de Análise Crítica em 5 Pontos**:

```
+---------------------------------------------------------------------------------+
|               CHECKLIST ANALÍTICA DE EXTRAÇÃO DE ARTIGOS                        |
+---------------------------------------------------------------------------------+
| 1. Problema e Motivação                                                         |
|    - Que estrangulamento de alinhamento, raciocínio ou computação é atacado?    |
|    - Porque falham os pipelines convencionais de SFT/PPO/DPO no cenário?       |
+---------------------------------------------------------------------------------+
| 2. Inovação Metodológica / Arquitetural                                         |
|    - Qual a formulação matemática da recompensa, perda ou gestão de memória?    |
|    - Como interage com o MDP ou com a estrutura auto-regressiva do LLM?         |
+---------------------------------------------------------------------------------+
| 3. Datasets e Protocolo de Avaliação                                            |
|    - Que conjuntos de dados de preferência, código ou matemática foram usados?   |
|    - Como foi medida a acurácia, o alinhamento e a segurança do modelo?         |
+---------------------------------------------------------------------------------+
| 4. Resultados Empíricos e Trade-offs (Precisão vs. Eficiência)                  |
|    - Quais os ganhos percentuais exatos face aos baselines consolidados?        |
|    - Qual o impacto no custo de inferência, estabilidade e comprimento do texto?|
+---------------------------------------------------------------------------------+
| 5. Limitações e Desafios em Aberto (Open Challenges)                           |
|    - O modelo exibe vulnerabilidade a reward hacking, sicofania ou staleness?   |
|    - Como generaliza o método fora da distribuição de treino (OOD)?             |
+---------------------------------------------------------------------------------+
```

### Detalhe Operacional de Cada Ponto da Checklist:

1. **Problema e Motivação:**
   - Registar com precisão o estrangulamento técnico focado (ex.: custo proibitivo da amostragem contínua *on-policy*; incapacidade dos ORMs em avaliar passos intermédios de lógica; vulnerabilidade a respostas bajuladoras e prolixas; colapso de entropia da política durante treino por PPO).
2. **Inovação Metodológica/Arquitetural:**
   - Isolar a formulação formal: descrever as equações de recompensa, funções de perda direta (DPO, KTO), arquiteturas de avaliação granular (PRM), algoritmos de amostragem em grupo (GRPO) ou mecanismos de gestão de *buffer* e controlo de caducidade (*staleness*).
3. **Datasets e Protocolo de Avaliação:**
   - Catalogar as bases experimentais (ex.: GSM8K, MATH, HumanEval, UltraFeedback, Anthropic HH-RLHF), os modelos base avaliados (Llama, Qwen, Mistral) e as métricas primárias (Win Rate vs. GPT-4, Pass@1, Taxa de Alucinação, Entropia da Política).
4. **Resultados Empíricos e Trade-offs:**
   - Extrair comparações numéricas objetivas contra os principais baselines. Avaliar o equilíbrio entre desempenho e custo computacional: registar se o método atinge ganhos expressivos poupando GPU-horas (como a redução de até 40% de computação com *Experience Replay*) ou se requer sobrecarga de inferência.
5. **Limitações e Desafios em Aberto:**
   - Apontar com transparência as fragilidades do método: dependência de oráculos determinísticos inexistentes em temas abertos, sensibilidade ao desvio de distribuição (*distributional shift*), modos de desalinhamento emergente ou dificuldade de generalização fora de domínio (*out-of-distribution*).

---

## 5. Padrão de Formatação para a Tabela de Literatura (`README.md`)

O ficheiro `README.md` localizado neste diretório é a matriz viva de curadoria. Todas as inserções devem seguir estritamente a convenção de 6 colunas padronizadas.

### 5.1 Especificação Rigorosa das 6 Colunas Obrigatórias

| Coluna | Título | Tipo de Conteúdo e Regras de Formatação |
|:---:|:---|:---|
| **1** | **Nome** | Título oficial integral do artigo em língua inglesa, sem abreviações arbitrárias. |
| **2** | **Detalhes** | Metadados catalográficos estruturados verticalmente com `<br><br>`: Autores, Data de publicação, Publisher, Livro/Journal/Conferência, Volume, Páginas e link oficial do DOI. |
| **3** | **Abstract** | Citações textuais literais do resumo original do artigo, colocadas obrigatoriamente entre aspas duplas (`"..."`), destacando a motivação, o método e as conclusões centrais. |
| **4** | **Conclusion** | Citações textuais literais retiradas da secção final de conclusões do artigo original, colocadas obrigatoriamente entre aspas duplas (`"..."`). |
| **5** | **Resumo (NotebookLM)** | Síntese crítica aprofundada em língua portuguesa estruturada impreterivelmente em **quatro parágrafos encadeados** separados por `<br><br>` (conforme especificado na secção 5.2). |
| **6** | **Citação** | Referência bibliográfica completa formatada de acordo com a **Norma Vancouver (NLM)**, finalizada com a hiperligação DOI ativa. |

---

### 5.2 Estrutura Padronizada do "Resumo (NotebookLM)" (4 Parágrafos)

A coluna 5 deve obedecer estritamente à seguinte sequência narrativa de quatro parágrafos:

*   **Parágrafo 1 — Contexto e Problema de Investigação:**  
    Apresentação do artigo, contextualização do desafio em RL para LLMs (ex.: estrangulamento da supervisão por resultado final, custo massivo de *rollouts*, ameaças de desalinhamento e *reward hacking*) e a lacuna de conhecimento que a obra aborda.
*   **Parágrafo 2 — Mecanismo Proposto e Inovação Arquitetural:**  
    Explicação aprofundada do método, formalização matemática ou enquadramento taxonómico concebido pelos autores (ex.: estrutura do modelo de recompensa PRM vs. ORM, dinâmica de *replay buffers*, equações de divergência ou novos paradigmas de supervisão).
*   **Parágrafo 3 — Validação Experimental e Métricas Numéricas:**  
    Descrição dos modelos base utilizados (ex.: Qwen2.5, Llama-3), conjuntos de dados de treino/teste, baselines de comparação direta e quantificação rigorosa dos resultados obtidos (precisão, ganhos de eficiência em GPU-horas, retenção de entropia ou redução de alucinações).
*   **Parágrafo 4 — Significado Teórico, Limitações e Desafios:**  
    Discussão do impacto do trabalho no estado da arte de alinhamento e raciocínio em IA, implicações práticas para a comunidade científica e as limitações ou direções de investigação futura identificadas pelos autores.

---

### 5.3 Exemplo Canónico de Registo na Tabela

Apresenta-se abaixo um exemplo de referência integral extraído da literatura curada neste repositório (Arnal et al., 2026):

```markdown
| Efficient rl training for llms with experience replay | **Autores:** Arnal et al.<br><br>**Ano:** 2026<br><br>**Pub:** arXiv | "While Experience Replay—the practice of storing rollouts and reusing them multiple times during training—is a foundational technique in general RL, it remains largely unexplored in LLM post-training due to the prevailing belief that fresh, on-policy data is essential for high performance. In this work, we challenge this assumption."<br><br>"We present a systematic study of replay buffers for LLM post-training, formalizing the optimal design as a trade-off between staleness-induced variance, sample diversity and the high computational cost of generation."<br><br>"Empirically, we show that a well-designed replay buffer can drastically reduce inference compute without degrading – and in some cases even improving – final model performance, while preserving policy entropy." | "In this work, we challenged the "generate-then-discard" paradigm that currently dominates LLM reinforcement learning. Through a combination of theoretical analysis and extensive empirical evaluation, we show that a well-configured replay buffer serves as a powerful lever for compute efficiency."<br><br>"Our theoretical framework characterizes a fundamental three-way trade-off between staleness, sample diversity, and the relative cost of inference. We show that as the computational burden of rollout generation grows, the optimal strategy shifts decisively toward experience replay."<br><br>"Empirically, we find that these gains are not merely theoretical: a simple replay buffer can reduce the compute budget by up to 40% while maintaining or even surpassing the accuracy of on-policy baselines." | Este artigo desafia o paradigma dominante "gerar e descartar" (*generate-then-discard*) no pós-treino por Aprendizagem por Reforço (*Reinforcement Learning* - RL) de Modelos de Linguagem de Grande Porte (LLMs). Embora a repetição de experiência (*Experience Replay*) seja uma técnica basilar no RL clássico, tem sido largamente preterida no treino de LLMs devido ao consenso de que a utilização de dados fora da política (*off-policy*) degrada o desempenho do modelo.<br><br>Os autores demonstram que incorporar um *replay buffer* em *pipelines* assíncronos de treino permite trocar um aumento controlado na caducidade dos dados (*staleness*/*off-policiness*) e uma ligeira redução na diversidade de amostras por uma redução drástica nos custos computacionais de geração/inferência. Através de uma formalização teórica da decomposição viés-variância no gradiente estocástico (SGD), provam que a eficiência computacional máxima é atingida não por ser estritamente *on-policy*, mas sim equilibrando a frescura e diversidade das amostras com o custo de geração.<br><br>Empiricamente, em modelos como Qwen2.5-7B, Qwen3-0.6B, Qwen3-8B e Llama 3.2 3B em tarefas de raciocínio matemático e código, a utilização de um *replay buffer* bem configurado permitiu economizar até 40% do orçamento computacional mantendo, e em alguns casos superando, a precisão dos modelos de base *on-policy*, ao mesmo tempo que estabiliza o treino e preserva a entropia da política. | Arnal C, Cabannes V, Cohen T, Kempe J, Munos R. Efficient rl training for llms with experience replay. arXiv preprint arXiv:2604.08706; 2026. https://doi.org/10.48550/arXiv.2604.08706. |
```

---

## 6. Guardrails de Qualidade & Anti-Alucinação

Para assegurar a idoneidade, reprodutibilidade e conformidade ética de qualquer processo de curadoria no repositório, o agente ou investigador deve agir sob as seguintes salvaguardas obrigatórias:

### 6.1 Validação Rigorosa de DOIs e Metadados
- **Verificação Ativa da Resolução do DOI:** Cada hiperligação inserida deve ser confirmada através do padrão canónico `https://doi.org/10.xxxx/...` (ou `https://doi.org/10.48550/arXiv.xxxx` no caso do repositório arXiv). É expressamente proibida a criação conjetural de prefixos ou sufixos de identificadores digitais.
- **Confronto Bibliográfico Direto:** Os metadados catalográficos (nomes de autores, ano de publicação, título exato do artigo e denominação da conferência ou revista) devem ser confrontados diretamente com bases de indexação internacional como CrossRef, DBLP, Google Scholar ou as plataformas oficiais dos editores (IEEE, ACM, Elsevier, Springer).

### 6.2 Proibição Estrita de Interpolação ou Arredondamento Fraudulento de Métricas
- **Fidelidade Numérica Absoluta:** As métricas de precisão, ganhos de eficiência ou redução de parâmetros devem ser transcritas com exatidão matemática. Se o artigo indica uma economia computacional de $40\%$ ou uma redução de alucinações de $14.3\%$, é proibido aproximar para "cerca de metade" ou "aproximadamente 15%".
- **Distinção Rígida entre Métricas Semelhantes:** Manter clara a discriminação metodológica entre taxas de aprovação (ex.: Pass@1 vs. Pass@10), preferências humanas (Win Rate bruto vs. Win Rate controlado por comprimento) e estimativas pontuais vs. distribucionais.

### 6.3 Fidelidade Literal das Citações Diretas
- Os campos **Abstract** e **Conclusion** destinam-se exclusivamente a transcrições textuais literais das passagens originais dos autores, colocadas entre aspas duplas (`"..."`). É estritamente vedado parafrasear, resumir informalmente ou introduzir juízos valorativos externos dentro dessas colunas de citação direta.

### 6.4 Sobriedade Terminológica e Neutralidade Epistemológica
- As sínteses e revisões devem pautar-se por um discurso científico rigoroso, neutro e objetivo. Devem ser banidos termos hiperbólicos ou sensacionalistas (tais como "modelo com inteligência humana", "técnica revolucionária infalível" ou "precisão milagrosa").
- O foco analítico deve concentrar-se exclusivamente nas propriedades funcionais dos algoritmos de RL, nos compromissos entre exploração e convergência, na estabilidade dos gradientes de política, na veracidade do raciocínio e nos desafios abertos de alinhamento e eficiência no mundo real.
