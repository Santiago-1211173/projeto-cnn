# Protocolo de Curadoria e Navegação na Literatura: Episodic Memory

> **Ficheiro Central Associado:** `README.md` (no mesmo diretório)  
> **Natureza do Documento:** Guia metodológico de recolha, análise crítica e referenciação científica para agentes autónomos e investigadores.  
> **Domínio Científico:** Inteligência Artificial / Aprendizagem Automática / Sistemas Semiparamétricos, Controlo Episódico e Memória Não-Paramétrica em Redes Neuronais.

---

## 1. Enquadramento Teórico & Âmbito do Tópico

### 1.1 Definição Canónica na Literatura Internacional de IA
A **Memória Episódica** (*Episodic Memory* - EM) no domínio da Inteligência Artificial e da Aprendizagem Profunda (*Deep Learning*) define uma classe de arquiteturas e algoritmos concebidos para registar, preservar e recuperar rapidamente instâncias pontuais de experiências passadas (trajetórias, transições de estado, representações latentes e valores de utilidade), viabilizando uma tomada de decisão adaptativa e uma retenção de conhecimento quase instantânea (*one-shot* ou *few-shot learning*).

Diferente das Redes Neuronais Profundas estritamente paramétricas — que codificam conhecimento de forma difusa e gradual na totalidade dos seus pesos sinápticos via retropropagação de gradientes (*backpropagation*) —, os sistemas dotados de memória episódica operam através de estruturas semi-tabulares ou não-paramétricas que desacoplam a **aquisição imediata de eventos** da **consolidação estatística de representações abstratas**.

Formalmente, no contexto da Aprendizagem por Reforço (*Reinforcement Learning* - RL), considere-se um ambiente modelado como um Processo de Decisão de Markov (MDP) definido pelo tuplo $\mathcal{M} = (\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma)$. O agente gera sequências temporais de interação:
$$\tau = (s_0, a_0, r_0, s_1, a_1, r_1, \dots, s_T)$$
A recompensa acumulada descontada (*return*) a partir do passo $t$ é dada por:
$$R_t = \sum_{k=0}^{T - t} \gamma^k r_{t+k}$$

Num sistema de **Controlo Episódico** (*Episodic Control*), o agente mantém um repositório explícito de memória $\mathcal{E}$, composto por pares de chave-valor ou tuplos:
$$\mathcal{E}_a = \left\{ (\phi(s_i), V(s_i, a)) \right\}_{i=1}^{M_a}, \quad \forall a \in \mathcal{A}$$
onde $\phi: \mathcal{S} \to \mathbb{R}^d$ denota uma função de projeção ou mapeamento para um espaço de *embeddings* (estática, aprendida ou projetada aleatoriamente), e $V(s_i, a)$ armazena o melhor retorno histórico observado:
$$V(s, a) \leftarrow \max \left( V(s, a), R_t \right)$$

Ao encontrar um novo estado $s$, a estimativa do valor de ação $\hat{Q}_{EC}(s, a)$ é obtida através de uma agregação não-paramétrica sobre os vizinhos mais próximos no espaço latente:
$$\hat{Q}_{EC}(s, a) = \sum_{i \in \mathcal{N}_k(\phi(s))} w_i V(s^{(i)}, a), \quad w_i = \frac{K(\phi(s), \phi(s^{(i)}))}{\sum_{j \in \mathcal{N}_k(\phi(s))} K(\phi(s), \phi(s^{(j)}))}$$
onde $\mathcal{N}_k(\phi(s))$ representa o conjunto dos $k$ vizinhos mais próximos de $\phi(s)$ na memória $\mathcal{E}_a$, e $K(\cdot, \cdot)$ é uma função de núcleo (*kernel*) de similaridade métrica.

```
+-----------------------------------------------------------------------------------+
|               ARQUITETURA CANÓNICA DE MEMÓRIA EPISÓDICA EM IA                     |
+-----------------------------------------------------------------------------------+
|                                                                                   |
|           Estado Bruto (s)                                                        |
|                  |                                                                |
|                  v                                                                |
|         +------------------+                                                      |
|         | Extrator Latente |  <--- Mapeamento phi(s) (CNN / Projeção Aleatória)   |
|         +------------------+                                                      |
|                  |                                                                |
|                  | Vetor Latente h = phi(s)                                       |
|                  +--------------------------+                                     |
|                  |                          |                                     |
|                  v                          v                                     |
|       +--------------------+      +--------------------+                          |
|       | Memória Paramétrica|      | Memória Episódica  |                          |
|       | (Neocórtex - Lento)|      | (Hipocampo - Rápido|                          |
|       +--------------------+      +--------------------+                          |
|       | - Pesos Sinápticos |      | - Dicionário DND   |                          |
|       | - Gradiente SGD    |      | - Buffer k-NN      |                          |
|       | - Regras Gerais    |      | - One-Shot Lookup  |                          |
|       +--------------------+      +--------------------+                          |
|                  \                          /                                     |
|                   \                        /                                      |
|                    v                      v                                       |
|             +------------------------------------+                                |
|             |  Decisão Integrada / Ação Ótima    |                                |
|             |  Q(s, a) = f(Q_param, Q_episodic)  |                                |
|             +------------------------------------+                                |
+-----------------------------------------------------------------------------------+
```

---

### 1.2 Problema Fundamental que Aborda
A integração de mecanismos de memória episódica na inteligência artificial resolve quatro limitações crónicas das redes neuronais convencionais:

1. **Ineficiência Crónica de Amostragem (*Extreme Sample Inefficiency*):**
   - Os algoritmos de aprendizagem profunda por reforço padrão (ex.: DQN, A3C, PPO) requerem dezenas de milhões de transições com o ambiente para aprender políticas competentes. Em contrapartida, agentes biológicos descobrem estratégias recompensadoras após uma única observação de sucesso.
   - A memória episódica contorna a lentidão de propagação dos gradientes armazenando instantaneamente o caminho de recompensa, permitindo ao agente recuperar e reproduzir a sequência ótima desde a primeira tentativa (*one-shot learning*).

2. **O Fenómeno do Esquecimento Catastrófico (*Catastrophic Forgetting*):**
   - Em cenários sequenciais e não-estacionários (*continual learning*), atualizar pesos globais através de SGD para assimilar novas distribuições sobrescreve padrões previamente adquiridos.
   - A memória episódica fornece um repositório isolado de instâncias (*coreset* ou *instance store*) que preserva eventos críticos do passado e permite reativá-los ou utilizá-los para guiar atualizações locais sem perturbar a estabilidade global.

3. **Atribuição de Crédito Temporal em Ambientes com Recompensas Esparsas (*Sparse Rewards*):**
   - Quando as recompensas são raras e distantes no tempo (ex.: labirintos complexos, jogos de estratégia), métodos de diferenças temporais (*Temporal Difference* - TD) propagam o sinal de erro a uma taxa de apenas um passo temporal por episódio, necessitando de uma quantidade astronómica de iterações.
   - A memória episódica efetua atualizações baseadas em Monte Carlo ($N$-step ou retornos integrais) que ligam diretamente o evento de sucesso ao estado inicial desencadeador.

4. **Fundamentação Neurocientífica: Teoria dos Sistemas de Aprendizagem Complementares (*CLS Theory*):**
   - Inspirada nas descobertas de McClelland, McNaughton & O'Reilly (1995) e Kumaran, Hassabis & McClelland (2016), a inteligência biológica opera através da cooperação de dois sistemas complementares:
     - **Sistema Hipocampal:** Aprendizagem rápida e episódica, com alta plasticidade, especializada em registrar instâncias específicas com separação de padrões (*pattern separation*).
     - **Sistema Neocortical:** Aprendizagem gradual e estatística, especializada em extrair estruturas conceituais compartilhadas e generalização abstrata com completamento de padrões (*pattern completion*).
   - A IA clássica tentou unificar ambas as funções num único bloco paramétrico, gerando o dilema estabilidade-plasticidade. A introdução de módulos episódicos restaura a dualidade preconizada pelo CLS.

---

### 1.3 Taxonomia e Principais Famílias de Abordagens no Estado da Arte

O ecossistema contemporâneo de Memória Episódica organiza-se em quatro famílias metodológicas distintas:

```
                                  Taxonomia de Memória Episódica
                                                |
          +--------------------+----------------+--------------------+--------------------+
          |                    |                                     |                    |
   1. Controlo Episódico 2. Adaptação Local de        3. Memórias Associativas 4. Controlo Hierárquico
      Não-Paramétrico e     Parâmetros e Sistemas        Neuromórficas e        e Memória de Eventos
      Semi-Tabular          Semiparamétricos             Baseadas em Energia     (OptionEM / Event)
          |                    |                                     |                    |
    - MFEC               - MbPA                                - Modern Hopfield    - OptionEM
    - NEC                - MbPA++                              - SQHN (MAP rule)    - Event Memory
    - NEC-RP             - Deep Semiparametric                 - CoLaNET (SNN)      - HCAM / EPN
    - DND Modules          Learning (k-NN latent)              - Bio-realistic      - SAR / 3D Nav
```

#### Família 1: Controlo Episódico Não-Paramétrico e Semi-Tabular
- **Hipótese Central:** A retenção explícita de valores $Q$ em tabelas dinâmicas indexadas por representações de estado permite aprender com ordens de grandeza menos amostras do que métodos paramétricos.
- **Modelos Canónicos:**
  - **MFEC (*Model-Free Episodic Control* - Blundell et al., 2016):** Primeiro modelo a utilizar tabelas não-paramétricas associadas a projeções aleatórias ou autoencoders variacionais (VAEs) para armazenar os retornos máximos observados e consultar valores via $k$-NN.
  - **NEC (*Neural Episodic Control* - Pritzel et al., 2017):** Introduz o *Differentiable Neural Dictionary* (DND), permitindo o treino ponta-a-ponta (*end-to-end*) do extrator de características convolucional enquanto os valores $Q$ das folhas do dicionário são atualizados instantaneamente via $N$-step Q-learning.
  - **NEC-RP (*Random Projection in Neural Episodic Control* - Nishio & Yamane, 2019):** Substitui as camadas densas totalmente ligadas por matrizes de Projeção Aleatória Gaussianas, preservando distâncias euclidianas (Lema de Johnson-Lindenstrauss), reduzindo parâmetros treináveis e estabilizando a convergência do DND.
- **Limitações:** Custos de computação da busca de vizinhos mais próximos ($k$-NN) em memórias com milhões de entradas e incapacidade de generalizar para regiões distantes do espaço de estados sem cobertura prévia.

#### Família 2: Adaptação Local de Parâmetros e Sistemas Semiparamétricos
- **Hipótese Central:** A memória episódica não deve substituir a rede profunda, mas antes modular temporariamente os seus parâmetros no momento do teste em função da vizinhança contextual do dado de entrada.
- **Modelos Canónicos:**
  - **MbPA (*Memory-based Parameter Adaptation* - Sprechmann et al., 2018):** Armazena representações latentes e rótulos numa memória não-paramétrica. Durante a inferência, recupera os vizinhos mais próximos e executa alguns passos de gradiente para ajustar local e temporariamente os pesos da camada de saída, revertendo-os após a predição.
  - **Deep Semiparametric Learning (Jain & Lindsey, 2018):** Combina uma componente paramétrica e um módulo $k$-NN diferenciável. No início do treino o modelo ancora-se na memória não-paramétrica e, à medida que os pesos convergem, transfere gradualmente a responsabilidade para o extrator paramétrico, espelhando a consolidação de sistemas complementares.
- **Vantagem Notável:** Mitiga o esquecimento catastrófico e lida eficazmente com classes raras ou distribuições de rótulos desequilibradas (*imbalanced data*).

#### Família 3: Memórias Associativas Neuromórficas e Baseadas em Energia
- **Hipótese Central:** A superação da implausibilidade biológica da retropropagação exige modelos baseados em energia, regras de plasticidade puramente locais (Hebbianas/anti-Hebbianas) e códigos esparsos para memorização contínua online.
- **Modelos Canónicos:**
  - **SQHN (*Sparse Quantized Hopfield Network* - Alonso & Krichmar, Nat. Commun. 2024):** Modelo gráfico discreto que aprende via regras locais de Máximo a Posteriori (MAP), utilizando códigos neuronais esparsos e quantizados, neurogénese dinâmica (crescimento de nós) e isolamento estrito de parâmetros para obter memorização *one-shot* e imunidade ao esquecimento catastrófico sem *backpropagation*.
  - **CoLaNET (*Columnar Spiking Neural Networks* - Larionov et al., 2025):** Redes de *spikes* estruturadas em microcolunas corticais que equilibram estabilidade e plasticidade através de plasticidade anti-Hebbiana modulada por dopamina e renormalização de recursos sinápticos.
- **Vantagem Notável:** Elevada compatibilidade com aceleradores neuromórficos (*memristores* e processadores *event-driven*) e consumo energético sustentável.

#### Família 4: Controlo Hierárquico e Memória de Eventos (*Event Memory*)
- **Hipótese Central:** A transição de tarefas sintéticas para ambientes físicos e de engenharia de alta complexidade requer decomposição hierárquica e planeamento implícito na própria estrutura de memória episódica.
- **Modelos Canónicos:**
  - **OptionEM (*Option Episodic Memory* - Zhou et al., 2023):** Estrutura de RL hierárquica baseada no *framework* de opções que substitui a exploração aleatória inválida por planeamento guiado pela memória episódica, demonstrado com sucesso no alinhamento de imagens de radar de abertura sintética (SAR) em tempo sub-segundo.
  - **Event Memory (Boyle & Blomkvist, Phil. Trans. R. Soc. B 2024):** Formalização conceptual que distingue a "memória de eventos" funcional em agentes artificiais da memória episódica com consciência autonoética humana, clarificando o papel causal das memórias em navegação, exploração e tomada de decisões estratégicas.
  - **HCAM / EPN (*Hierarchical Chunking & Episodic Planning Networks*):** Agrupamento temporal de trajetórias em episódios discretos (*chunks*), permitindo atribuir crédito temporal ao longo de longos horizontes sem necessidade de desenrolar simuladores dinâmicos exaustivos.

---

### 1.4 Dicionário de Conceitos-Chave (Pedagógico e Rigoroso)

*   **Controlo Episódico (Episodic Control):**  
    Paradigma de tomada de decisão em que o agente consulta diretamente memórias episódicas passadas de trajetórias bem-sucedidas para determinar a ação imediata, em vez de depender exclusivamente de uma função de valor paramétrica convergida lentamente.
*   **Dicionário Neural Diferenciável (Differentiable Neural Dictionary - DND):**  
    Estrutura de dados chave-valor onde as chaves são vetores de características contínuos extraídos por uma rede neuronal e os valores são estimativas de retorno acumulado. A consulta é feita via *kernel* diferenciável e as atualizações de valor são locais e imediatas.
*   **Teoria dos Sistemas de Aprendizagem Complementares (Complementary Learning Systems - CLS):**  
    Teoria neurocientífica que postula a coexistência de dois módulos integrados: o hipocampo (memorização rápida episódica de instâncias específicas com separação de padrões) e o neocórtex (consolidação gradual e lenta de representações estatísticas gerais com completamento de padrões).
*   **Projeção Aleatória (Random Projection - RP):**  
    Técnica de redução de dimensionalidade linear baseada no Lema de Johnson-Lindenstrauss. Ao projetar dados de alta dimensão para um espaço de dimensão inferior através de uma matriz estocástica normalizada, preserva aproximadamente as distâncias euclidianas sem exigir pesos treináveis ou gradientes.
*   **Adaptação de Parâmetros Baseada em Memória (Memory-based Parameter Adaptation - MbPA):**  
    Técnica semiparamétrica em que a rede neuronal geral mantém pesos globais lentos, mas, no instante da inferência de uma dada amostra, utiliza os seus vizinhos mais próximos recuperados da memória para efetuar um ajuste local e efémero dos pesos da camada final.
*   **Memória de Eventos (Event Memory):**  
    Terminologia proposta na filosofia da ciência e ciência cognitiva para descrever os módulos de memória episódica em IA. Reconhece que os agentes artificiais retêm sequências espaciotemporais de eventos com valor causal, dispensando estados mentais fenomenológicos humanos (como o sentimento de reviver o passado ou *mental time travel*).
*   **Redes de Hopfield Modernas e Quantizadas (Sparse Quantized Hopfield Networks - SQHN):**  
    Modelos de memória associativa contínua e discreta com capacidade de armazenamento exponencial ou linear super-eficiente. No caso da SQHN, utiliza representações neuronais esparsas e regras locais de Máximo a Posteriori (MAP) para memorização contínua sem colapso associativo.
*   **Redes Neuronais de Spikes Colunares (Columnar Spiking Neural Networks - CoLaNET):**  
    Arquitetura de processamento temporal inspirada nas microcolunas do neocórtex cerebral, operando com impulsos elétricos discretos (*spikes*) e plasticidade sináptica local modulada por neuromoduladores (dopamina).
*   **Controlo Episódico Baseado em Opções (Option Episodic Memory - OptionEM):**  
    Integração da memória episódica com a teoria de opções do RL hierárquico, permitindo ao agente recuperar e reutilizar políticas de sub-rotinas temporais em vez de apenas ações atómicas primitivas.
*   **Busca pelos Vizinhos Mais Próximos ($k$-Nearest Neighbors - $k$-NN):**  
    Algoritmo não-paramétrico que identifica os $k$ vetores mais próximos de um vetor de consulta num espaço métrico (geralmente sob distância euclidiana, cosseno ou Manhattan), servindo de operador de interpolação local na memória.
*   **Separação de Padrões vs. Completamento de Padrões (*Pattern Separation vs. Pattern Completion*):**  
    Dois processos cognitivos fundamentais: a separação de padrões transforma entradas semelhantes em representações ortogonais para evitar interferência (função do giro dentado no hipocampo); o completamento reconstrói padrões completos a partir de pistas parciais ou ruidosas (função da região CA3 do hipocampo e redes associativas).

---

## 2. Protocolo de Pesquisa Sistemática (Search Strategy)

Para assegurar uma cobertura exaustiva, metodologicamente calibrada e isenta de enviesamento na literatura de Memória Episódica em IA, agentes autónomos e investigadores devem seguir rigorosamente o protocolo delineado infra.

### 2.1 Venues Científicos Prioritários
As buscas devem concentrar-se nas conferências e periódicos científicos de topo com revisão por pares rigorosa (*peer-review*):

| Categoria | Sigla / Nome do Venue | Qualificação / Foco Científico |
|:---|:---|:---|
| **Conferências de IA & ML (Core A\*)** | **NeurIPS** (Neural Information Processing Systems) | Teoria de RL, modelos semiparamétricos e representações |
| | **ICML** (International Conference on Machine Learning) | Algoritmos de controlo episódico (MFEC, NEC), eficiência de amostragem |
| | **ICLR** (International Conference on Learning Representations) | Dicionários diferenciáveis, embeddings e memórias associativas |
| | **AAAI / IJCAI** | Aprendizagem contínua, planeamento e modelos cognitivos |
| **Revistas Científicas de Referência** | **Nature / Nature Communications** | Modelos neuromórficos, redes de Hopfield, avanços biológicos |
| | **Philosophical Transactions of the Royal Society B** | Fundamentação cognitiva e epistemológica de agentes com memória |
| | **IEEE T-PAMI / IEEE T-NNLS** | Redes de memória, controlo e visão computacional em larga escala |
| | **JMLR** (Journal of Machine Learning Research) | Fundamentação matemática e garantias teóricas em aprendizagem |
| **Conferências e Journals de Aplicação** | **IEEE T-GRS / Remote Sensing** | Aplicações de controlo episódico em dados de satélite e SAR |
| | **CoRL / IROS / ICRA** | Navegação robótica e exploração física com suporte episódico |
| **Repositórios de Preprints Verificados** | **arXiv** (`cs.LG`, `cs.AI`, `cs.NE`, `q-bio.NC`) | Descobertas recentes nos últimos 12 a 24 meses (validação cuidada) |

---

### 2.2 Equações de Pesquisa Booleanas Recomendadas

As seguintes cadeias de pesquisa encontram-se estruturadas e otimizadas para motores de busca indexados (Google Scholar, Semantic Scholar, IEEE Xplore, ACM Digital Library e Scopus):

#### Bloco A: Controlo Episódico Fundacional e Dicionários Diferenciáveis
```text
("episodic memory" OR "episodic control") AND ("reinforcement learning" OR "deep reinforcement learning") AND ("Model-Free Episodic Control" OR "Neural Episodic Control" OR "Differentiable Neural Dictionary" OR "sample efficiency")
```

#### Bloco B: Modelos Semiparamétricos e Adaptação Local de Parâmetros
```text
("episodic memory" OR "semi-parametric learning") AND ("Memory-based Parameter Adaptation" OR "MbPA" OR "nearest neighbor" OR "differentiable memory") AND ("continual learning" OR "catastrophic forgetting")
```

#### Bloco C: Memórias Associativas Neuromórficas e Baseadas em Energia
```text
("episodic memory" OR "associative memory") AND ("Hopfield network" OR "sparse quantized" OR "spiking neural network" OR "CoLaNET" OR "local learning rules") AND ("online continual" OR "catastrophic forgetting")
```

#### Bloco D: Controlo Hierárquico, Memória de Eventos e Aplicações em Engenharia
```text
("episodic memory" OR "event memory") AND ("hierarchical reinforcement learning" OR "OptionEM" OR "SAR image registration" OR "spatial navigation") AND ("artificial agent" OR "cognitive systems")
```

---

### 2.3 Janela Temporal de Análise

A pesquisa bibliográfica deve cobrir duas etapas cronológicas interdependentes:

1. **Janela Fundacional Seminal (2016 – 2019):**
   - **Objetivo:** Compreender a formulação matemática inicial do controlo episódico e das arquiteturas semiparamétricas pioneiras.
   - **Marcos Históricos Obrigatórios:** Blundell et al. (2016 - *Model-Free Episodic Control*), Pritzel et al. (ICML 2017 - *Neural Episodic Control*), Sprechmann et al. (2018 - *Memory-based Parameter Adaptation*), Jain & Lindsey (ICLR 2018 - *Deep Semiparametric Learning*), Nishio & Yamane (ACML 2019 - *Random Projection in NEC*).

2. **Janela do Estado da Arte Recente (Últimos 3 a 5 anos):**
   - **Objetivo:** Mapear a evolução para regras locais de plasticidade neuromórfica, sistemas imunes ao esquecimento contínuo, controlo hierárquico em problemas complexos e a conceptualização epistemológica da memória em agentes de IA.
   - **Marcos Recentes Obrigatórios:** Zhou et al. (Remote Sensing 2023 - *OptionEM para SAR*), Alonso & Krichmar (Nature Communications 2024 - *SQHN e regras MAP locais*), Boyle & Blomkvist (Phil. Trans. R. Soc. B 2024 - *Elements of Episodic Memory e Event Memory*), Larionov et al. (Opt. Mem. Neural Networks 2025 - *CoLaNET e SNNs colunares*).

---

## 3. Critérios de Elegibilidade (Inclusão e Exclusão)

Para manter o mais elevado rigor epistemológico na base de conhecimento, cada publicação científica identificada deve ser submetida a um crivo sistemático de elegibilidade.

```
                      Artigo Identificado na Pesquisa
                                     |
               +---------------------+---------------------+
               |                                           |
               v                                           v
      Critérios de Inclusão (+)                  Critérios de Exclusão (-)
   - Mecanismo episódico explícito             - Mero buffer de replay clássico
   - Ganhos comprovados em amostras            - Apenas posições opinativas
   - Baselines competitivos (DQN, PPO)         - Sem detalhes de vizinhança k-NN
   - Transparência algorítmica                 - Métricas sem variância empírica
               |                                           |
               v                                           v
        ACEITE NA TABELA                            REJEITADO
```

### 3.1 Critérios de Inclusão (+)
O artigo deve preencher cumulativamente pelo menos **três** dos seguintes requisitos para inclusão:
1. **Mecanismo de Memória Explícito:** Propõe ou analisa formalmente estruturas de memória não-paramétrica, semi-tabular, dicionários diferenciáveis ou redes associativas de alta capacidade para retenção de instâncias.
2. **Avaliação de Eficiência de Amostragem (*Sample Efficiency*):** Demonstra quantitativamente que o agente requer ordens de grandeza inferiores de interações com o ambiente face a modelos paramétricos convencionais para atingir patamares equivalentes de recompensa.
3. **Resiliência ao Esquecimento em Aprendizagem Contínua:** Apresenta medições objetivas da taxa de retenção de conhecimento passado (*backward transfer* / degradação de exatidão) em fluxos contínuos de dados não-*i.i.d.*
4. **Comparação com *Baselines* Consolidados:** Contrasta o modelo com referências canónicas de RL (DQN, Rainbow, PPO, A3C, MFEC) ou de aprendizagem contínua (EWC, GEM, iCaRL, Replay clássico).
5. **Transparência e Reprodutibilidade:** Detalha a parametrização das memórias (número de entradas, estratégia de substituição de chaves, valor de $k$ no $k$-NN, métrica de distância utilizada).

### 3.2 Critérios de Exclusão (-)
Devem ser rejeitados os documentos que apresentem alguma das seguintes deficiências:
1. **Redução Trivial a *Experience Replay* Uniforme:** Trabalhos que utilizam unicamente o *Replay Buffer* tradicional de DQN como suporte estocástico para gradiente descendente padrão, sem qualquer componente de inferência não-paramétrica, agregação por vizinhança ou controlo episódico.
2. **Ausência de Validação Empírica Controlada:** Ensaios meramente especulativos ou manifestos sem testes comparativos em *benchmarks* públicos reconhecidos (ex.: Atari 2600, Permuted MNIST, ViZDoom, Labyrinth, dados SAR).
3. **Opacidade Arquitetural:** Publicações que não fornecem as equações de atualização da memória, as regras de plasticidade ou as métricas de similaridade utilizadas na recuperação de instâncias.
4. **Publicações Sem Arbitragem Científica Idónea:** Artigos alojados em veículos predatórios desprovidos de revisão por pares independente.

---

## 4. Roteiro de Extração e Síntese Analítica

Para cada artigo selecionado, o analista deve preencher obrigatoriamente a **Checklist Analítica de 5 Pontos**:

```
+---------------------------------------------------------------------------------+
|               CHECKLIST ANALÍTICA DE EXTRAÇÃO DE ARTIGOS                        |
+---------------------------------------------------------------------------------+
| 1. Problema e Motivação                                                         |
|    - Que lacuna na eficiência de amostragem ou estabilidade é abordada?         |
|    - Por que falham os modelos paramétricos padrão neste cenário?               |
+---------------------------------------------------------------------------------+
| 2. Inovação Metodológica / Arquitetural                                         |
|    - Qual a estrutura exata da memória (DND, SQHN, MbPA, Tabela Q não-param)?   |
|    - Como são calculadas as chaves, os valores e a função de recuperação (k-NN)?|
+---------------------------------------------------------------------------------+
| 3. Datasets e Protocolo de Avaliação                                            |
|    - Quais os benchmarks testados (Atari, Permuted MNIST, Labyrinth, SAR)?      |
|    - Como foram configuradas as restrições de amostragem e horizonte temporal?  |
+---------------------------------------------------------------------------------+
| 4. Resultados Empíricos e Trade-offs (Precisão vs. Eficiência)                  |
|    - Qual a aceleração de convergência (ex.: 10x menos interações)?             |
|    - Qual a sobrecarga de memória (RAM) e latência de consulta de vizinhos?     |
+---------------------------------------------------------------------------------+
| 5. Limitações e Desafios em Aberto (Open Challenges)                           |
|    - O modelo é superado por redes paramétricas no longo prazo?                 |
|    - Como lida com a maldição da dimensionalidade no espaço de chaves?          |
+---------------------------------------------------------------------------------+
```

### Detalhe Operacional de Cada Ponto da Checklist:

1. **Problema e Motivação:**
   - Registar se o foco é acelerar o início do treino (*cold-start*), combater o esquecimento catastrófico em *online continual learning*, viabilizar aprendizagem em ambientes com recompensas esparsas, ou conferir plausibilidade biológica ao sistema.
2. **Inovação Metodológica/Arquitetural:**
   - Mapear a formulação matemática da memória: dimensionalidade do espaço de chaves ($d$), algoritmo de vizinhos (KD-Tree, FLANN, Projeção Aleatória), regra de escrita (substituição LRU, retornos máximos, neurogénese) e mecanismo de integração (leitura diferenciável, mistura convexa ou adaptação temporária de pesos).
3. **Datasets e Protocolo de Avaliação:**
   - Documentar os ambientes avaliados, o número de passos de treino permitidos (ex.: regime de dados ultrabaixo de 1 milhão a 10 milhões de frames em vez das habituais centenas de milhões), e se a avaliação foi realizada *online* ou em regime sequencial de tarefas.
4. **Resultados Empíricos e Trade-offs:**
   - Quantificar os ganhos: fator de redução de interações (ex.: $10\times$ a $50\times$), taxa de esquecimento numérico (ex.: degradação inferior a 5% após 10 tarefas no CoLaNET ou SQHN), tempo de inferência por decisão e volume de memória física ocupado pelas tabelas de instâncias.
5. **Limitações e Desafios em Aberto:**
   - Registar explicitamente se o artigo admite perda de competitividade assintótica em estágios tardios de treino face a redes puramente paramétricas (um fenómeno comum documentado no MFEC e NEC), ou desafios na escalabilidade do $k$-NN para milhões de estados.

---

## 5. Padrão de Formatação para a Tabela de Literatura (`README.md`)

O ficheiro `README.md` localizado no mesmo diretório deve manter a sua tabela do estado da arte estritamente alinhada com as seguintes especificações de formatação.

### 5.1 Especificação Rigorosa das 6 Colunas Obrigatórias

| Coluna | Título | Tipo de Conteúdo e Regras de Formatação |
|:---:|:---|:---|
| **1** | **Nome** | Título oficial e integral do artigo em língua inglesa, sem traduções ou omissões. |
| **2** | **Detalhes** | Metadados catalográficos estruturados verticalmente utilizando quebras de linha duplas em HTML (`<br><br>`), contendo obrigatoriamente: Autores, Data de publicação, Livro/Journal/Conferência, Volume, Número, Páginas e link DOI resolúvel. |
| **3** | **Abstract** | Citações textuais fiéis, exatas e diretas do resumo original do artigo, delimitadas obrigatoriamente entre aspas duplas (`"..."`). Devem focar-se no problema, mecanismo proposto e resultados centrais. |
| **4** | **Conclusion** | Citações textuais fiéis e diretas retiradas da secção de conclusão ou considerações finais do artigo original, delimitadas entre aspas duplas (`"..."`). |
| **5** | **Resumo (NotebookLM)** | Síntese analítica aprofundada em língua portuguesa, estruturada de forma estrita em **quatro parágrafos encadeados** separados por `<br><br>` (ver secção 5.2). |
| **6** | **Citação** | Referência bibliográfica em conformidade rigorosa com a **Norma Vancouver (NLM)**, finalizada com a hiperligação DOI ativa. |

---

### 5.2 Estrutura Padronizada do "Resumo (NotebookLM)" (4 Parágrafos)

A quinta coluna deve obedecer escrupulosamente ao encadeamento temático dos quatro parágrafos infra, redigidos com clareza pedagógica e rigor concetual:

*   **Parágrafo 1 — Contexto e Problema de Investigação:**  
    Apresentação do artigo, introdução do problema enfrentado no contexto de redes neuronais e aprendizagem por reforço (ex.: ineficiência extrema de dados, lentidão na propagação do gradiente, esquecimento catastrófico) e motivação teórica (inspiração em sistemas complementares biológicos ou lacuna computacional).
*   **Parágrafo 2 — Mecanismo Proposto e Inovação Arquitetural:**  
    Explicação pormenorizada da arquitetura concebida pelos autores (ex.: Dicionário Neural Diferenciável, Projeção Aleatória não-paramétrica, adaptação local de pesos baseada em memória, códigos esparsos quantizados ou microcolunas com regras locais). Deve detalhar a interação entre componentes rápidos e lentos.
*   **Parágrafo 3 — Validação Experimental e Métricas Numéricas:**  
    Descrição concisa dos cenários empíricos de teste (conjuntos de dados como Atari 2600, Permuted MNIST, satélites SAR, labirintos 3D), baselines de comparação direta (DQN, EWC, PPO) e resumo quantitativo dos resultados (ordens de grandeza de aceleração, percentagens de precisão ou taxas de retenção).
*   **Parágrafo 4 — Significado Teórico, Limitações e Desafios:**  
    Análise do impacto do trabalho no estado da arte, indicação transparente das limitações admitidas pelos autores (ex.: saturação do $k$-NN, ultrapassagem por modelos paramétricos no longo prazo, sensibilidade a hiperparâmetros) e direções abertas para investigações futuras.

---

### 5.3 Exemplo Canónico de Registo na Tabela

Para servir de modelo de calibração absoluta, apresenta-se de seguida o exemplo de preenchimento rigoroso de uma linha da tabela, extraído diretamente do estado da arte do repositório:

```markdown
| Neural Episodic Control | **Autores:** Alexander Pritzel, Benigno Uria, Sriram Srinivasan, Adrià Puigdomènech Badia, Oriol Vinyals, Demis Hassabis, Daan Wierstra, Charles Blundell<br><br>**Data de publicação:** 2017<br><br>**Publisher:** PMLR<br><br>**Livro/Journal:** Proceedings of the 34th International Conference on Machine Learning<br><br>**Volume:** 70<br><br>**Páginas:** 2827-2836 | "We propose Neural Episodic Control: a deep reinforcement learning agent that is able to rapidly assimilate new experiences and act upon them."<br><br>"Our agent uses a semi-tabular representation of the value function: a buffer of past experience containing slowly changing state representations and rapidly updated estimates of the value function."<br><br>"We show across a wide range of environments that our agent learns significantly faster than other state-of-the-art, general purpose deep reinforcement learning agents." | "We have proposed Neural Episodic Control (NEC): a deep reinforcement learning agent that learns significantly faster than other baseline agents on a wide range of Atari 2600 games."<br><br>"At the core of NEC is a memory structure: a Differentiable Neural Dictionary (DND), one for each potential action. NEC inserts recent state representations paired with corresponding value functions into the appropriate DND."<br><br>"Our experiments show that NEC requires an order of magnitude fewer interactions with the environment than agents previously proposed for data efficiency, such as Prioritised Replay (Schaul et al., 2015b) and Retrace(λ) (Munos et al., 2016)."<br><br>"Our work suggests that non-parametric methods are a promising addition to the deep reinforcement learning toolbox, especially where data efficiency is paramount." | Este artigo apresenta o Neural Episodic Control (NEC), um agente de Deep Reinforcement Learning desenhado para contornar a extrema ineficiência de dados dos métodos tradicionais, que necessitam frequentemente de milhões de interações adicionais em comparação com humanos.<br><br>O NEC introduz uma arquitetura de memória semi-tabular (o Dicionário Neural Diferenciável - DND), que lhe permite reter, assimilar e aplicar rapidamente novas experiências de alto valor, ultrapassando os tempos de aprendizagem habitualmente lentos dos processos de descida de gradiente (gradient descent).<br><br>Testes realizados no ambiente de jogos Atari 2600 comprovaram que o NEC aprende a uma velocidade significativamente superior, necessitando de uma ordem de grandeza a menos de interações com o ambiente para atingir bons resultados quando comparado com outros algoritmos de referência como DQN, Prioritised Replay e Retrace(λ).<br><br>No entanto, os autores salientam que, embora o NEC atinja patamares de excelência muito mais cedo, em fases avançadas de treino contínuo os métodos puramente paramétricos podem eventualmente igualar ou superar o modelo devido à sua capacidade superior de generalização global em espaços de grande escala. | Pritzel A, Uria B, Srinivasan S, Badia AP, Vinyals O, Hassabis D, et al. Neural Episodic Control. In: Proceedings of the 34th International Conference on Machine Learning. PMLR; 2017. p. 2827–36. (vol. 70). |
```

---

## 6. Guardrails de Qualidade & Anti-Alucinação

Para garantir a integridade científica do acervo documental e impedir distorções metodológicas, todo o agente ou investigador deve submeter as novas entradas aos seguintes controlos de validação obrigatórios:

### 6.1 Verificação Ativa de DOIs e Metadados
- **Resolução de Links:** Todo o DOI deve ser resolvível através da hiperligação canónica `https://doi.org/10.xxxx/...`. É expressamente proibida a criação de links sintéticos ou prefixos fictícios.
- **Conferência em Bases Centrais:** Antes da inserção, os metadados (volume, número, paginação, editora, data) devem ser validados contra o repositório oficial (PMLR, Nature, IEEE Xplore, Royal Society, arXiv).

### 6.2 Proibição de Interpolação ou Arredondamento Fraudulento de Métricas
- Dados empíricos devem ser citados com fidelidade matemática absoluta. Se o artigo indica "4% de degradação" ou "uma ordem de grandeza a menos de interações", a síntese deve registar exatamente esses termos, sendo vedado arredondar para números convenientes (ex.: "cerca de 5%" ou "dezenas de vezes mais rápido").
- Não criar correlações ou propriedades de convergência que não constem comprovadas no texto original do artigo.

### 6.3 Fidelidade Textual em Citações Diretas
- As colunas **Abstract** e **Conclusion** destinam-se exclusivamente a **citações literais e textuais retiradas do texto original em língua inglesa**. Não é permitida a paráfrase, a tradução intercalada ou a reestruturação da ordem sintática dentro das aspas duplas.

### 6.4 Neutralidade Epistemológica e Sobriedade Linguística
- É expressamente vedada a utilização de adjetivos sensacionalistas ou não fundamentados (ex.: "arquitetura milagrosa", "avanço perfeito", "desempenho inigualável").
- A redação deve manter rigorosamente o tom dos periódicos internacionais de topo de Inteligência Artificial: sóbria, analítica, focada em hipóteses, condições de fronteira, trade-offs e limitações empíricas.
