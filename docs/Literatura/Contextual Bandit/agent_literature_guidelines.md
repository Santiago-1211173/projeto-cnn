# Protocolo de Curadoria e Navegação na Literatura: Contextual Bandit

> **Ficheiro Central Associado:** `README.md` (no mesmo diretório)  
> **Natureza do Documento:** Guia metodológico de recolha, análise crítica e referenciação científica para agentes autónomos e investigadores.  
> **Domínio Científico:** Inteligência Artificial / Aprendizagem Automática / Aprendizagem por Reforço e Tomada de Decisão Sequencial sob Incerteza.

---

## 1. Enquadramento Teórico & Âmbito do Tópico

### 1.1 Definição Canónica na Literatura Internacional de IA
O problema do **Bandido Multi-Braços Contextual** (*Contextual Multi-Armed Bandit* - CMAB ou simplesmente *Contextual Bandit*) é uma formulação de aprendizagem por reforço (*Reinforcement Learning* - RL) e tomada de decisão sequencial em que um agente interage com um ambiente estocástico ao longo de uma sequência discreta de rondas $t = 1, 2, \dots, T$.

Em cada ronda $t$:
1. O ambiente disponibiliza uma informação contextual (vetor de contexto ou *covariates*) $x_t \in \mathcal{X} \subseteq \mathbb{R}^D$, representativa do estado do sistema, utilizador ou tarefa;
2. O agente, condicionado a $x_t$ e ao histórico acumulado $\mathcal{H}_{t-1} = \{(x_\tau, a_\tau, r_\tau)\}_{\tau=1}^{t-1}$, seleciona uma ação ou "braço" $a_t$ a partir de um conjunto de ações admissíveis $\mathcal{A}_t \subseteq \mathcal{A}$;
3. O ambiente devolve uma recompensa numérica escalar imediata $r_t \in \mathbb{R}$, regida por uma distribuição com valor esperado condicional desconhecido:
   $$\mathbb{E}[r_t \mid x_t, a_t] = h(x_t, a_t)$$
   onde $h: \mathcal{X} \times \mathcal{A} \to \mathbb{R}$ denota a função latente de recompensa média, frequentemente corrompida por ruído estocástico $\epsilon_t$ de média nula ($\mathbb{E}[\epsilon_t \mid x_t, a_t] = 0$).

A característica distintiva crucial do *Contextual Bandit* reside no **feedback parcial (ou feedback de bandido)**: o agente observa unicamente a recompensa $r_t$ referente à ação selecionada $a_t$, não tendo acesso às recompensas contrafatuais que teriam sido geradas caso tivesse escolhido qualquer outro braço alternativo $a' \neq a_t$.

```
                 +---------------------------------------------+
                 |                Ambiente                     |
                 +---------------------------------------------+
                        |                             ^
       1. Contexto xt   |                             | 2. Ação at
                        v                             |
                 +---------------------------------------------+
                 |            Agente de Tomada                 |
                 |              de Decisão                     |
                 +---------------------------------------------+
                        ^
       3. Recompensa rt | (Feedback parcial: apenas para at)
                        |
                 +---------------------------------------------+
                 |                Ambiente                     |
                 +---------------------------------------------+
```

---

### 1.2 Problema Fundamental que Aborda
O paradigma de *Contextual Bandit* resolve uma lacuna metodológica crítica situada entre a Aprendizagem Supervisionada clássica e a Aprendizagem por Reforço em Processos de Decisão de Markov (MDPs):

1. **Limitação da Aprendizagem Supervisionada (Supervised Learning):**
   - Na aprendizagem supervisionada, assume-se acesso ao "rótulo verdadeiro" ou ação ótima para cada exemplo do conjunto de treino (perda de informação nula).
   - Em ambientes dinâmicos do mundo real (ex.: sistemas de recomendação, seleção de anúncios, escolha de intervenções terapêuticas, compressão adaptativa), o ambiente não fornece a melhor ação teórica, fornecendo apenas a resposta observada à escolha efetuada. A aprendizagem supervisionada convencional é incapaz de gerir ativamente a recolha de novos dados e o risco de exploração.

2. **Diferença face ao Bandido Clássico sem Contexto (Multi-Armed Bandit - MAB):**
   - No MAB tradicional (ex.: UCB1, Thompson Sampling clássico), assume-se que as propriedades dos braços são idênticas para qualquer cenário, ignorando variáveis de contexto.
   - O *Contextual Bandit* introduz a personalização: o braço ótimo $a^*(x) = \arg\max_{a \in \mathcal{A}} h(x, a)$ varia dinamicamente em função do vetor $x_t$.

3. **Diferença face ao Reinforcement Learning Completo (MDP / Q-Learning / Policy Gradient):**
   - No RL em MDP completo, a ação tomada altera o estado futuro do ambiente ($s_{t+1} \sim P(\cdot \mid s_t, a_t)$) e as recompensas podem ser diferidas no tempo com créditos temporais complexos (fator de desconto $\gamma$).
   - No *Contextual Bandit*, a decisão é de passo único (*one-step*): a ação $a_t$ gera uma recompensa imediata, mas não condiciona a transição estocástica do próximo contexto $x_{t+1}$ (tipicamente gerado segundo uma distribuição fixa ou independente). Isto elimina o problema do horizonte temporal longo, permitindo garantias teóricas mais robustas e convergência substancialmente mais rápida.

4. **Objetivo Teórico: Minimização de *Regret* Sublinear:**
   O desempenho de um agente de *Contextual Bandit* é quantificado formalmente pelo **Arrependimento Acumulado** (*Cumulative Pseudo-Regret*) ao longo de $T$ rondas:
   $$R_T = \sum_{t=1}^T \left[ \max_{a \in \mathcal{A}} h(x_t, a) - h(x_t, a_t) \right]$$
   Um algoritmo é teoricamente viável e convergente se o seu arrependimento for **sublinear no tempo**, isto é, $\lim_{T \to \infty} \frac{R_T}{T} = 0$, garantindo que a recompensa média por ronda converge assintoticamente para a política ótima de oráculo.

---

### 1.3 Taxonomia e Principais Famílias de Abordagens no Estado da Arte

As metodologias de *Contextual Bandit* distribuem-se em quatro grandes famílias teóricas e arquiteturais:

```
                                Taxonomia de Contextual Bandit
                                               |
         +--------------------+----------------+--------------------+--------------------+
         |                    |                                     |                    |
  1. Abordagens        2. Abordagens                         3. Abordagens        4. Abordagens
     Lineares e           Não-Paramétricas                      Neuronais Profundas  Off-Policy &
     Paramétricas         (Kernel / k-NN)                       (Deep / Neural)      Simulação
         |                    |                                     |                    |
   - LinUCB             - k-NN UCB                            - NeuralUCB (NTK)    - Inverse Propensity
   - LinTS              - KernelUCB                           - NeuralTS             Scoring (IPS)
   - GLM-UCB            - Partições em Árvores                - Dual-Network       - Doubly Robust
                          (Adaptação a d < D)                   (ex.: EE-Net)      - Benchmarks Re-param.
```

#### Família 1: Abordagens Lineares e Lineares Generalizadas (Paramétricas Clássicas)
- **Hipótese Central:** A função de recompensa média é uma combinação linear das características do contexto e do braço: $h(x, a) = x^\top \theta_a^*$ (ou $x^\top \theta^*$ num modelo de parâmetros partilhados), onde $\theta^*$ é desconhecido.
- **Modelos Canónicos:**
  - **LinUCB:** Aplica regressão linear de cume (*Ridge Regression*) para estimar $\hat{\theta}_a$ e constrói uma elipse de confiança via análise de grandes desvios. A decisão segue o princípio do otimismo face à incerteza: $a_t = \arg\max_a \left( x_t^\top \hat{\theta}_{a, t-1} + \alpha \sqrt{x_t^\top A_{a, t-1}^{-1} x_t} \right)$, onde $A_a$ é a matriz de covariância acumulada.
  - **LinTS (Linear Thompson Sampling):** Mantém uma distribuição Gaussiana *a posteriori* sobre os parâmetros $\theta_a \sim \mathcal{N}(\hat{\theta}_a, v^2 A_a^{-1})$ e amostra parâmetros em cada ronda para guiar a exploração.
  - **GLM-UCB / Logistic Bandits:** Modela recompensas binárias via modelos lineares generalizados com ligação logística $h(x, a) = \sigma(x^\top \theta_a)$.
- **Limitações:** Falha expressiva quando as relações reais entre o contexto e a recompensa são altamente não-lineares, multimodais ou descontínuas.

#### Família 2: Abordagens Não-Paramétricas e Baseadas em Variedades (Manifold-Adaptive)
- **Hipótese Central:** A função de recompensa $h(x, a)$ satisfaz apenas propriedades fracas de regularidade (ex.: continuidade Lipschitziana ou suavidade Hölderiana), sem assumir qualquer vetor de pesos rígido.
- **Modelos Canónicos:**
  - **$k$-NN UCB (Nonparametric Stochastic Contextual Bandits):** Estima a recompensa esperada localmente ponderando os $k$ vizinhos mais próximos do vetor $x_t$ no histórico de dados já recolhidos para cada braço.
  - **KernelUCB:** Projeta contextos para um Espaço de Hilbert de Reprodução de Kernel (*RKHS*), calculando limites de confiança através do produto interno no espaço induzido.
- **Vantagem Notável:** Capacidade intrínseca de contornar a maldição da dimensionalidade quando os dados observados habitam numa variedade (*manifold*) de menor dimensão intrínseca $d \ll D$, garantindo limites de arrependimento da ordem de $\tilde{O}(T^{\frac{1+d}{2+d}})$ em vez da dimensão ambiente $D$.

#### Família 3: Abordagens Neuronais Profundas (Deep / Neural Contextual Bandits)
- **Hipótese Central:** A função de recompensa é aproximada por uma Rede Neuronal Profunda (*Deep Neural Network* - DNN) $f(x, a; \theta)$, permitindo capturar abstrações complexas e representações latentes ricas.
- **Modelos Canónicos:**
  - **NeuralUCB (com base em NTK):** Utiliza uma rede profunda e constrói o termo de incerteza a partir do vetor de gradientes dos parâmetros $\nabla_\theta f(x_t, a; \theta)$. Provas de arrependimento $\tilde{O}(\tilde{d}\sqrt{T})$ são derivadas sob a teoria do *Neural Tangent Kernel* (NTK) para redes sobre-parametrizadas.
  - **NeuralTS:** Amostra parâmetros perturbados ao longo da direção da covariância dos gradientes neuronais.
  - **Arquiteturas Dual-Network Desacopladas (ex.: EE-Net):**
    - Separação explícita entre a **Rede de Aproveitamento** ($f_1$, que aprende a função de recompensa a partir dos dados observados) e a **Rede de Exploração** ($f_2$, que aprende a prever o ganho potencial da exploração com base nos gradientes de $f_1$).
    - *Exploração Bidirecional:* Distingue formalmente entre *Upward Exploration* (quando a rede $f_1$ subestima a recompensa real) e *Downward Exploration* (quando $f_1$ sobrestima a recompensa real, necessitando de uma correção negativa).
    - *Eficiência Computacional:* Elimina a necessidade de manter e inverter matrizes quadradas de covariância de gradientes $A_t = \sum_{\tau} g_\tau g_\tau^\top \in \mathbb{R}^{p \times p}$ (reduzindo o custo de memória de $O(p^2)$ para $O(p)$ e acelerando o tempo de inferência entre 30% a 60%).

#### Família 4: Aprendizagem Fora de Política (*Off-Policy Evaluation*) e Simulação de Ambientes
- **Hipótese Central:** Dado que a interação online em sistemas de produção é dispendiosa ou eticamente arriscada, a literatura desenvolve métodos para simular e avaliar algoritmos a partir de dados históricos logados sob políticas de amostragem arbitrárias.
- **Técnicas Canónicas:**
  - **Inverse Propensity Scoring (IPS) & Doubly Robust (DR):** Re-ponderação de recompensas observadas pela probabilidade com que a ação foi escolhida pela política de registo.
  - **Simulated Contextual Bandits from Recommendation Datasets:** Metodologias para transformar repositórios abertos de recomendação e matrizes de utilidade (ex.: MovieLens, IMDb, Netflix Prize) em ambientes dinâmicos de *Contextual Bandit*, reparametrizando perfis de utilizador em espaços de contexto $\mathcal{X}$, itens em espaços de ação $\mathcal{A}$ e classificações em sinais estocásticos de recompensa $r_t$.

---

### 1.4 Dicionário de Conceitos-Chave (Pedagógico e Rigoroso)

*   **Contexto ($x_t$ / Context Vector / Feature Space $\mathcal{X}$):**  
    Vetor numérico de entrada multidimensional que sumariza o ambiente no instante da decisão. Em problemas de recomendação, pode conter histórico de cliques, categorias preferidas e contexto horário; em visão computacional, pode conter vetores de *embeddings* ou características estatísticas da cena.
*   **Braço / Ação ($a_t$ / Action Space $\mathcal{A}$):**  
    Cada uma das escolhas discretas (ou contínuas) disponíveis para seleção em cada ronda. A terminologia deriva da analogia histórica com os "braços" das máquinas de jogo de azar (*one-armed bandits*).
*   **Recompensa ($r_t$ / Reward Signal):**  
    Sinal escalar numérico retornado pelo ambiente. Pode ser binário ($r \in \{0, 1\}$, ex.: clique/conversão ou acerto/erro na classificação) ou contínuo ($r \in [0, 1]$ ou $r \in \mathbb{R}$, ex.: receita gerada ou grau de similaridade semântica).
*   **Dilema Aproveitamento vs. Exploração (Exploitation vs. Exploration - Trade-off EE):**  
    O conflito fundamental em qualquer decisão sob incerteza:
    - *Exploitation (Aproveitamento):* Escolher a ação que o modelo atual estima ter a maior recompensa média, maximizando os ganhos imediatos com base no conhecimento consolidado.
    - *Exploration (Exploração):* Escolher ações com recompensa estimada inferior ou com elevada incerteza, com o propósito de recolher nova informação estatística para refinar estimativas futuras e evitar mínimos locais subótimos.
*   **Arrependimento / Perda Relativa (Regret / Cumulative Pseudo-Regret $R_T$):**  
    A métrica de referência na teoria de bandidos. Corresponde à diferença acumulada entre o ganho que teria sido alcançado por um decisor perfeito (oráculo que conhece a função $h(x, a)$ a priori) e o ganho real obtido pela política do agente ao longo de $T$ rondas.
*   **Upper Confidence Bound (UCB):**  
    Princípio determinista de exploração otimista: para cada braço, calcula-se a recompensa esperada somada a um termo proporcional à incerteza estatística (desvio padrão da estimativa). O agente seleciona o braço com maior valor somado, garantindo exploração natural dos braços pouco visitados.
*   **Thompson Sampling (TS / Posterior Sampling):**  
    Abordagem probabilística/Bayesiana de exploração: mantém-se uma distribuição de probabilidade sobre os parâmetros do modelo. Em cada ronda, amostra-se um conjunto aleatório de parâmetros da distribuição *a posteriori* e escolhe-se o braço ótimo sob os parâmetros amostrados.
*   **Neural Tangent Kernel (NTK):**  
    Ferramenta teórica fundamental na análise de redes neuronais sobre-parametrizadas, que demonstra que durante o treino com gradiente descendente em regime de largura infinita, o comportamento da rede aproxima-se do de um modelo linear no espaço induzido pelos gradientes dos pesos iniciais. É a base das provas de convergência de algoritmos como NeuralUCB.
*   **Upward vs. Downward Exploration:**  
    Conceito avançado em arquiteturas de exploração aprendida:
    - *Upward Exploration:* Ocorre quando a rede de aproveitamento subestima o valor real da recompensa ($h(x) > f_1(x)$), exigindo somar um valor positivo para viabilizar a escolha do braço.
    - *Downward Exploration:* Ocorre quando a rede de aproveitamento sobrestima o valor de um braço com recompensa real reduzida ($h(x) < f_1(x)$), exigindo aplicar um desconto negativo para suprimir explorações repetidas e inúteis de braços fracos.
*   **Ganho Potencial (Potential Gain):**  
    Diferença residual entre a recompensa real observada $r_t$ e a predição da rede de aproveitamento $f_1(x_t)$, denotada por $\Delta_t = r_t - f_1(x_t)$, utilizada como função de perda para treinar módulos autónomos de exploração.
*   **Dimensão Intrínseca ($d$) vs. Dimensão Ambiente ($D$):**  
    Em espaços de características de alta dimensionalidade (onde $x \in \mathbb{R}^D$), os dados frequentemente residem confinados a uma subvariedade suave de dimensão efetiva $d \ll D$. Algoritmos adaptativos exploram essa topologia latente para convergir a velocidades que dependem de $d$ e não do ruído de $D$.

---

## 2. Protocolo de Pesquisa Sistemática (Search Strategy)

Para assegurar uma cobertura abrangente, rigorosa e isenta de enviesamento na literatura de *Contextual Bandit*, os agentes e investigadores devem seguir o protocolo sistemático detalhado infra.

### 2.1 Venues Científicos Prioritários
As pesquisas bibliográficas devem priorizar estritamente publicações que tenham passado por rigoroso processo de revisão por pares (*peer-review*) nas seguintes conferências e revistas de topo internacional:

| Categoria | Sigla / Nome do Venue | Qualificação / Foco |
|:---|:---|:---|
| **Conferências de IA & ML (Core A\*)** | **NeurIPS** (Neural Information Processing Systems) | Teoria de RL, avanços em NTK e algoritmos de bandidos |
| | **ICML** (International Conference on Machine Learning) | Otimização estocástica, limites de regret e bandits |
| | **ICLR** (International Conference on Learning Representations) | Representação profunda e redes neuronais em bandits |
| | **AAAI** (Association for the Advancement of Artificial Intelligence) | Modelos não-paramétricos, raciocínio sob incerteza |
| | **AISTATS** (International Conference on Artificial Intelligence and Statistics) | Fundamentação estatística, Thompson Sampling e UCB |
| **Revistas Científicas de Referência** | **JMLR** (Journal of Machine Learning Research) | Artigos extensos com provas matemáticas integrais |
| | **IEEE T-PAMI** (Trans. on Pattern Analysis and Machine Intelligence) | Fundamentos de decisão e representação em larga escala |
| | **IEEE T-NNLS** (Trans. on Neural Networks and Learning Systems) | Otimização e arquiteturas neurais aplicadas a controle |
| **Conferências de Sistemas e Aplicação** | **ACM RecSys** (ACM Conference on Recommender Systems) | Bandidos aplicados a personalização e recomendação |
| | **ACM KDD** (Knowledge Discovery and Data Mining) | Algoritmos em larga escala e simulação em grafos/redes |
| **Repositórios de Preprints Verificados** | **arXiv** (`cs.LG`, `stat.ML`, `cs.AI`, `cs.CV`) | Descobertas recentes nos últimos 12 meses (validação cuidadosa) |

---

### 2.2 Equações de Pesquisa Booleanas Recomendadas

As seguintes strings de pesquisa encontram-se otimizadas para utilização em motores de busca científicos (ex.: Google Scholar, Semantic Scholar, IEEE Xplore, ACM Digital Library e Scopus):

#### Bloco A: Fundamentos Teóricos e Algoritmos Lineares/Kernel
```text
("contextual bandit" OR "contextual multi-armed bandit") AND ("regret bound" OR "sublinear regret") AND ("LinUCB" OR "Thompson Sampling" OR "nonparametric" OR "kernel")
```

#### Bloco B: Bandidos Neuronais Profundos (Deep & Neural Bandits)
```text
("contextual bandit" OR "contextual bandits") AND ("neural" OR "deep neural network") AND ("NeuralUCB" OR "NeuralTS" OR "neural tangent kernel" OR "exploration-exploitation network")
```

#### Bloco C: Exploração Adaptativa, Eficiência e Desacoplamento de Redes
```text
("contextual bandit" OR "multi-armed bandit") AND ("exploration direction" OR "upward exploration" OR "downward exploration" OR "potential gain" OR "dual network")
```

#### Bloco D: Simulação, Benchmarking e Tarefas de Personalização
```text
("contextual bandit" OR "contextual bandits") AND ("personalization" OR "recommender systems") AND ("simulated environment" OR "offline evaluation" OR "benchmark dataset")
```

---

### 2.3 Janela Temporal de Análise

A recolha bibliográfica deve ser estratificada em duas janelas temporais com objetivos complementares:

1. **Janela Seminal e Fundacional (2002 – 2017):**
   - **Objetivo:** Estabelecer as bases matemáticas do problema, compreendendo as propriedades clássicas de concentração de medida (limites de Hoeffding, desigualdades de Bernstein, azuma-martingales) e as derivações de LinUCB, LinTS e estratégias não-paramétricas.
   - **Marcos Históricos Relevantes:** Auer et al. (2002 - UCB clássico), Langford & Zhang (2008 - Epoch-Greedy), Li et al. (2010 - LinUCB no Yahoo! Today), Chapelle & Li (2011 - Validação empírica de Thompson Sampling), Agrawal & Goyal (2013 - Limites de regret para LinTS).

2. **Janela de Avanços Recentes e Estado da Arte (Últimos 3 a 5 anos):**
   - **Objetivo:** Identificar as soluções modernas que superam a linearidade através de redes neuronais profundas, redução do custo computacional de matrizes de covariância de gradiente, exploração guiada por redes auxiliares e adaptação topológica a dimensões intrínsecas baixas.
   - **Marcos Recentes Relevantes:** Zhou et al. (2020 - NeuralUCB sob NTK), Zhang et al. (2021 - NeuralTS), Guan & Jiang (AAAI 2018 - Bandits não-paramétricos adaptativos a variedades), Dereventsov & Bibin (2022 - Ambientes simulados a partir de bases de recomendação), Ban et al. (JMLR 2026 - EE-Net e exploração bidirecional aprendida).

---

## 3. Critérios de Elegibilidade (Inclusão e Exclusão)

Para manter o mais elevado rigor epistemológico na base de conhecimento, cada documento identificado na pesquisa deve ser triado segundo critérios formais e objetivos.

```
                      Artigo Identificado na Pesquisa
                                     |
               +---------------------+---------------------+
               |                                           |
               v                                           v
      Critérios de Inclusão (+)                  Critérios de Exclusão (-)
   - Prova teórica de Regret                  - Sem provas nem baselines
   - Comparação empírica com baselines        - Modelos puramente heurísticos
   - Transparência metodológica               - Sem detalhe de hiperparâmetros
   - Código/ambiente reprodutível             - Relatórios técnicos não revistos
               |                                           |
               v                                           v
       ACEITE NA TABELA                            REJEITADO
```

### 3.1 Critérios de Inclusão (+)
Um artigo deve satisfazer cumulativamente pelo menos **três** dos seguintes requisitos para inclusão:
1. **Garantias Teóricas Formais:** Apresenta derivação explícita de limites superiores de arrependimento (*regret upper bound*) com taxa sublinear (ex.: $\tilde{O}(\sqrt{T})$, $\tilde{O}(T^{\frac{1+d}{2+d}})$) e discussão das constantes dependentes da dimensão ($D$ ou $d$).
2. **Validação Empírica Robusta:** Compara a abordagem proposta contra *baselines* consolidados na literatura (ex.: LinUCB, LinTS, KernelUCB, NeuralUCB, $\epsilon$-Greedy) em múltiplos conjuntos de dados padrão abertos (ex.: MNIST, CIFAR, MovieLens, Adult, Covertype).
3. **Eficiência Computacional e Escalabilidade:** Avalia o tempo de inferência por ronda, o custo de memória (ex.: $O(p)$ vs. $O(p^2)$) e o impacto prático de operações de inversão de matrizes ou retropropagação de gradientes.
4. **Reprodutibilidade Científica:** Descreve exaustivamente a parametrização dos modelos (taxas de aprendizagem, coeficientes de exploração, larguras de rede, sementes aleatórias) e disponibiliza ou detalha o protocolo de geração dos dados e simulação.
5. **Relevância para Tomada de Decisão com Feedback Parcial:** Foca-se especificamente no cenário onde apenas a recompensa do braço executado é observada.

### 3.2 Critérios de Exclusão (-)
Devem ser sumariamente excluídos os trabalhos que apresentem qualquer uma das seguintes vulnerabilidades:
1. **Ausência de Comparação Justa:** Ensaios que comparam apenas contra variantes ingénuas (ex.: amostragem estritamente aleatória) omitindo baselines competitivos de referência internacional.
2. **Ambientes Arbitrários sem Fundamentação:** Estudos empíricos baseados exclusivamente num único ambiente sintético simplista desenhado propositadamente para favorecer o algoritmo proposto, sem testes em dados reais ou semi-reais padronizados.
3. **Opacidade Matemática e Arquitetural:** Documentos que não detalham as fórmulas de cálculo do termo de incerteza, da função de perda de exploração ou das regras de atualização dos pesos neuronais.
4. **Artigos de Divulgação e Opinião:** Ensaios conceituais, resumos de congressos sem revisão por pares técnica completa, teses sem artigos derivados em periódicos indexados e publicações em veículos predatórios sem arbitragem científica fiável.

---

## 4. Roteiro de Extração e Síntese Analítica

Ao analisar cada artigo aprovado na triagem de elegibilidade, o curador ou agente deve preencher obrigatoriamente a seguinte **Checklist Analítica de 5 Pontos**:

```
+---------------------------------------------------------------------------------+
|               CHECKLIST ANALÍTICA DE EXTRAÇÃO DE ARTIGOS                        |
+---------------------------------------------------------------------------------+
| 1. Problema e Motivação                                                         |
|    - Que lacuna teórica ou prática aborda?                                      |
|    - Porque falham os modelos lineares ou abordagens anteriores neste contexto? |
+---------------------------------------------------------------------------------+
| 2. Inovação Metodológica / Arquitetural                                         |
|    - Qual o mecanismo exato proposto (ex.: rede dual, k-NN adaptativo, NTK)?    |
|    - Como é calculada a seleção do braço (otimismo, amostragem ou rede)?        |
+---------------------------------------------------------------------------------+
| 3. Datasets e Protocolo de Avaliação                                            |
|    - Quais os conjuntos de dados utilizados (simulados, semi-reais, reais)?     |
|    - Como foi construído o espaço de estados, de ações e o sinal de recompensa? |
+---------------------------------------------------------------------------------+
| 4. Resultados Empíricos e Trade-offs (Precisão vs. Eficiência)                  |
|    - Qual a melhoria percentual ou absoluta em Regret Acumulado?                |
|    - Qual o impacto no tempo de treino e inferência por ronda?                  |
+---------------------------------------------------------------------------------+
| 5. Limitações e Desafios em Aberto (Open Challenges)                           |
|    - Sob que condições o algoritmo degrada ou falha?                            |
|    - Quais as questões teóricas ou práticas deixadas para trabalhos futuros?    |
+---------------------------------------------------------------------------------+
```

### Detalhe Operacional de Cada Ponto da Checklist:

1. **Problema e Motivação:**
   - Registar com precisão o pressuposto teórico que o artigo contesta (ex.: pressuposto de linearidade de LinUCB, complexidade $O(p^2)$ de NeuralUCB, ou escassez de dados reais abertos para tarefas de recomendação).
2. **Inovação Metodológica/Arquitetural:**
   - Documentar os blocos matemáticos centrais: definição da política de decisão $a_t$, formulação da perda de otimização $\mathcal{L}(\theta)$ e mecanismos de atualização dos parâmetros.
3. **Datasets e Protocolo de Avaliação:**
   - Registar o número total de rondas $T$, número de braços $|\mathcal{A}|$, dimensionalidade do contexto $D$, e a metodologia utilizada para sintetizar o feedback de bandido (ex.: conversão de rótulos multiclasse com ruído ou funções de utilidade de cosseno escaladas).
4. **Resultados Empíricos e Trade-offs:**
   - Extrair métricas objetivas numéricas: curvas de *Cumulative Regret*, taxa de cliques média (*CTR*), precisão acumulada e tempo de execução por iteração (milissegundos/segundos).
5. **Limitações e Desafios em Aberto:**
   - Identificar vulnerabilidades documentadas pelos próprios autores (ex.: sensibilidade a hiperparâmetros de ruído, perda de sinal na reparametrização de dados ou custos de escalabilidade para milhares de braços).

---

## 5. Padrão de Formatação para a Tabela de Literatura (`README.md`)

O ficheiro central `README.md` localizado no mesmo diretório deve manter a sua tabela de estado da arte estritamente alinhada com as seguintes especificações de formatação.

### 5.1 Especificação Rigorosa das 6 Colunas Obrigatórias

| Coluna | Título | Tipo de Conteúdo e Regras de Formatação |
|:---:|:---|:---|
| **1** | **Nome** | Título oficial e integral do artigo em língua inglesa, sem traduções ou alterações de grafia. |
| **2** | **Detalhes** | Metadados catalográficos estruturados verticalmente utilizando quebras de linha duplas em HTML (`<br><br>`), contendo obrigatoriamente: Autores, Data de publicação, Livro/Journal/Conferência, Volume, Número, Páginas e link DOI (quando disponível). |
| **3** | **Abstract** | Excertos textuais fiéis, exatos e diretos do resumo original do artigo, delimitados obrigatoriamente entre aspas duplas (`"..."`). Devem focar-se no problema, proposta e resultados centrais. |
| **4** | **Conclusion** | Excertos textuais fiéis e diretos retirados da secção de conclusão ou considerações finais do artigo original, delimitados entre aspas duplas (`"..."`). |
| **5** | **Resumo (NotebookLM)** | Síntese analítica aprofundada em língua portuguesa, estruturada de forma estrita em **quatro parágrafos encadeados** separados por `<br><br>` (ver secção 5.2). |
| **6** | **Citação** | Referência bibliográfica em conformidade rigorosa com a **Norma Vancouver (NLM)**, finalizada com a hiperligação DOI ativa. |

---

### 5.2 Estrutura Padronizada do "Resumo (NotebookLM)" (4 Parágrafos)

A quinta coluna deve obedecer escrupulosamente ao encadeamento temático dos quatro parágrafos infra, redigidos com clareza pedagógica e rigor concetual:

*   **Parágrafo 1 — Contexto e Problema de Investigação:**  
    Apresentação do artigo, introdução da classe de problema abordada dentro do espetro de *Contextual Bandits* e identificação explícita das limitações teóricas ou práticas que o artigo se propõe ultrapassar.
*   **Parágrafo 2 — Mecanismo Proposto e Inovação Arquitetural:**  
    Explicação do método ou arquitetura concebida pelos autores (ex.: divisão em redes duplas, regressão não-paramétrica por vizinhos, modelação via NTK). Deve clarificar como o mecanismo equilibra aproveitamento e exploração e como calcula os limites ou sinais de decisão.
*   **Parágrafo 3 — Validação Experimental e Métricas Numéricas:**  
    Descrição concisa dos cenários empíricos de teste (conjuntos de dados utilizados, baselines de comparação direta) e resumo quantitativo dos resultados alcançados (comportamento de *regret*, velocidade de convergência ou ganhos de eficiência de cálculo).
*   **Parágrafo 4 — Significado Teórico, Limitações e Desafios:**  
    Análise do impacto do trabalho no estado da arte, indicação transparente das limitações assumidas pelos autores e principais orientações deixadas para investigações futuras.

---

### 5.3 Exemplo Canónico de Registo na Tabela

Para servir de modelo de calibração absoluta, apresenta-se de seguida o exemplo de preenchimento correto de uma linha da tabela:

```markdown
| Neural Exploitation and Exploration of Contextual Bandits | **Autores:** Yikun Ban, Yuchen Yan, Arindam Banerjee, Jingrui He<br><br>**Data de publicação:** 5 abril 2026<br><br>**Livro/Journal:** Journal of Machine Learning Research<br><br>**Volume:** 27<br><br>**Número:** 55<br><br>**Páginas:** 1-38 | "In this paper, we introduce, EE-Net, which is a novel framework to utilize another neural network to learn the potential gain of exploitation neural network for exploration, different from UCB-based and TS-based approaches that rely on the large-deviation-based statistical confidence bound."<br><br>"In addition, we provide an instance-based $\tilde{O}(\sqrt{T})$ regret upper bound for EE-Net with a new proof workflow."<br><br>"Empirically, we show that EE-Net outperforms related linear and neural contextual bandit baselines on real-world datasets." | "In this paper, we propose a novel exploration strategy, EE-Net, by investigating the exploration direction in contextual bandits."<br><br>"In addition to a neural network that exploits collected data in past rounds, EE-Net has another neural network to learn the potential gain compared to the current estimate for adaptive exploration."<br><br>"We provide an instance-dependent regret upper bound for EE-Net and then use experiments to demonstrate its empirical performance." | Este artigo apresenta a "EE-Net", uma nova estrutura para resolver o dilema de exploração e explotação (*exploitation and exploration*) no problema dos *contextual bandits*.<br><br>Ao contrário das abordagens tradicionais (como *Thompson Sampling* e *Upper Confidence Bound*) que dependem de limites de confiança estatística, a EE-Net introduz uma rede neuronal secundária especificamente dedicada a aprender o ganho potencial em relação à estimativa atual, permitindo uma exploração mais adaptativa e direcionada.<br><br>O estudo apresenta uma prova teórica de um limite de arrependimento (*regret bound*) ótimo para o modelo e demonstra, através de experiências, que a EE-Net supera os modelos de referência (lineares e neuronais) em conjuntos de dados do mundo real. O desacoplamento em duas redes reduz ainda a sobrecarga computacional de inverter matrizes de covariância de gradientes de $O(p^2)$ para $O(p)$.<br><br>Como limitação documentada, a eficácia do método depende da calibração adequada da representação de gradientes e de técnicas de redução dimensional como LLE para viabilizar o processamento em modelos de grande escala. | Ban Y, Yan Y, Banerjee A, He J. Neural Exploitation and Exploration of Contextual Bandits. J Mach Learn Res. 2026;27(55):1–38. |
```

---

## 6. Guardrails de Qualidade & Anti-Alucinação

Para assegurar a integridade científica intransigente do repositório, qualquer agente inteligente ou investigador humano deve submeter as suas entradas aos seguintes controlos de validação obrigatórios:

### 6.1 Verificação Ativa de DOIs e Metadados
- **Validação de Links:** Todo o DOI registado na tabela deve ser resolvível através do prefixo oficial `https://doi.org/10.xxxx/...`. É estritamente proibido inventar sufixos de DOI ou gerar links quebrados.
- **Conferência em Bases Centrais:** Antes da inserção, os metadados (volume, número, paginação, data exata) devem ser validados contra o registo no CrossRef, DBLP ou nos repositórios oficiais dos editores (JMLR, IEEE, ACM, AAAI).

### 6.2 Proibição de Interpolação ou Arredondamento Fraudulento de Métricas
- Se um artigo afirma que obteve um *speedup* entre 32% e 59%, o texto da síntese **deve citar literalmente "entre 32% e 59%"**, sendo proibido escrever arbitrariamente "aproximadamente 50%" ou "cerca de 60%".
- Valores de limites de arrependimento teórico (ex.: $\tilde{O}(\sqrt{T})$, $\tilde{O}(T^{\frac{1+d}{2+d}})$) devem ser transcritos com rigor absoluto quanto às variáveis de dimensionalidade e notações assintóticas ($\tilde{O}$ omitindo fatores logarítmicos).

### 6.3 Fidelidade Textual em Citações Diretas
- As colunas **Abstract** e **Conclusion** destinam-se exclusivamente a **citações literais transcritas da fonte primária**. Não é permitida a paráfrase, alteração da ordem lógica das frases nem inserção de interpretações subjetivas dentro das aspas.

### 6.4 Neutralidade e Sobriedade Epistemológica
- É expressamente proibida a utilização de adjetivos superlativos ou promocionais desprovidos de base comparativa (ex.: "método revolucionário", "resultado milagroso", "desempenho perfeito").
- A linguagem deve manter o padrão dos periódicos científicos internacionais de topo: neutra, factual, precisa e atenta aos pressupostos e limitações dos modelos matemáticos.
