# Protocolo de Curadoria e Navegação na Literatura: Hopfield Networks e Memória Associativa

> **Ficheiro Central Associado:** `README.md` (no mesmo diretório)  
> **Natureza do Documento:** Guia metodológico de recolha, análise crítica e referenciação científica para agentes autónomos e investigadores.  
> **Domínio Científico:** Inteligência Artificial / Aprendizagem Automática / Redes de Hopfield, Memória Associativa e Arquiteturas Centradas em Memória (*Memory-Centric Deep Learning*).

---

## 1. Enquadramento Teórico & Âmbito do Tópico

### 1.1 Definição Canónica na Literatura Internacional de IA
As **Redes de Hopfield (*Hopfield Networks*)** constituem uma classe canónica de redes neuronais recorrentes concebidas primordialmente como sistemas de **memória associativa endereçável por conteúdo (*content-addressable associative memory*)**. Formuladas originalmente por John Hopfield em 1982 na sua versão discreta e binária com conexões sinápticas simétricas, estas redes operam através da convergência dinâmica de estados para pontos fixos de equilíbrio — denominados **atratores (*attractors*)** — que minimizam uma função escalar monotónica decrescente denominada **função de energia de Lyapunov**.

Na sua formulação clássica discreta, dado um vetor de estado binário $s \in \{-1, +1\}^d$ e uma matriz de pesos sinápticos simétrica e com diagonal nula $W \in \mathbb{R}^{d \times d}$ ($W = W^\top$, $w_{ii} = 0$), a energia do sistema é definida por:

$$E(s) = -\frac{1}{2} s^\top W s = -\frac{1}{2} \sum_{i=1}^d \sum_{j=1}^d w_{ij} s_i s_j$$

onde a regra de atualização determinística assíncrona $s_i \leftarrow \text{sign}\left(\sum_{j=1}^d w_{ij} s_j\right)$ garante que a energia decresce monotonamente até alcançar um mínimo local estável correspondente a um dos padrões previamente armazenados via aprendizagem hebbiana ($W = \sum_{\mu=1}^N x_\mu x_\mu^\top$).

#### A Revolução das Modern Hopfield Networks (MHNs) e Estados Contínuos
Durante quase quatro décadas, a utilidade prática das redes de Hopfield clássicas foi severamente limitada pela sua reduzida capacidade de armazenamento ($C \approx 0.138 d$). Esta limitação foi superada decisivamente com a introdução das **Redes de Hopfield Modernas (*Modern Hopfield Networks - MHNs*)**, também designadas na literatura por **Memórias Associativas Densas (*Dense Associative Memories - DAM*)**, iniciadas por Krotov e Hopfield (2016) e expandidas para domínios de estados contínuos e diferenciáveis por Ramsauer et al. (2020 / ICLR 2021).

Nas Modern Continuous Hopfield Networks, dado um conjunto de $N$ padrões vetoriais contínuos armazenados como colunas de uma matriz de memória $X = [x_1, x_2, \dots, x_N] \in \mathbb{R}^{d \times N}$ e um vetor de consulta ou estado $z \in \mathbb{R}^d$, a função de energia de Lyapunov contínua é formulada como:

$$E(z) = -\beta^{-1} \ln \left( \sum_{k=1}^N \exp(\beta \, x_k^\top z) \right) + \frac{1}{2} z^\top z + \frac{1}{2} M^2 + \beta^{-1} \ln N$$

onde $\beta > 0$ denota o parâmetro de temperatura inversa (fator de escala de nitidez associativa) e $M = \max_k \|x_k\|$.

Através da aplicação do procedimento de convexidade-concavidade (CCCP - *Concave-Convex Procedure*), a regra de atualização dinâmica minimizadora de energia converge para um atrator num **único passo global de contração**:

$$z^{\text{new}} = X \, \text{softmax}(\beta \, X^\top z) = \sum_{k=1}^N \frac{\exp(\beta \, x_k^\top z)}{\sum_{j=1}^N \exp(\beta \, x_j^\top z)} x_k$$

```
+---------------------------------------------------------------------------------------------------+
|               DINÂMICA DE ATRAÇÃO E EQUIVALÊNCIA COM ATENÇÃO EM MODERN HOPFIELD NETWORKS         |
+---------------------------------------------------------------------------------------------------+

     Espaço de Estados Latentes                                     Paisagem de Energia E(z)
                                                                 (Superfície de Lyapunov Contínua)
       Padrão Armazenado x_1          Padrão Armazenado x_2
             (Atrator 1)                    (Atrator 2)                    E(z) ^
                 (*)                            (*)                             |      /\          /\
                  ^                              ^                              |     /  \        /  \
                   \                            /                               |    /    \      /    \
                    \                          /                                |   /      \____/      \
                     \                        /                                 |  /        Mínimo 2    \
                      \                      /                                  | / Mínimo 1             \
       Entrada Ruidosa z o                  o Padrão Parcial z                  +----------------------------> z
       (Recuperação em 1 Passo via softmax)                                          Bacia 1       Bacia 2
       z_new = X * softmax(beta * X^T * z)                                         (Atrator x_1) (Atrator x_2)
                                                                                          ^             ^
                                                                              z_inicial --+             +-- z_parcial

+---------------------------------------------------------------------------------------------------+
|               EQUIVALÊNCIA FORMAL COM O MECANISMO DE ATENÇÃO DOS TRANSFORMERS                    |
+---------------------------------------------------------------------------------------------------+
|                                                                                                   |
|   Modern Hopfield Update:      z_new = X  *  softmax( beta * X^T * z )                            |
|                                        |                 |       |                                |
|   Transformer Attention:     Output  = V  *  softmax( (1/sqrt(d_k)) * K^T * Q )                   |
|                                                                                                   |
|   -> Padrões Armazenados (X) atuam simultaneamente como Chaves (Keys) e Valores (Values).         |
|   -> O Estado de Consulta (z) corresponde exatamente à Consulta (Query).                         |
|   -> A autoatenção padrão dos Transformers é, rigorosamente, uma camada de Modern Hopfield!       |
+---------------------------------------------------------------------------------------------------+
```

#### A Conexão Canónica com os Transformers e Backbones Visuais Centrados em Memória
Como demonstrado formalmente por Ramsauer et al. (2020), a operação de autoatenção escalada multi-cabeça (*Scaled Dot-Product Attention*) introduzida por Vaswani et al. (2017) é **matematicamente equivalente ao passo de recuperação de uma Modern Hopfield Network** onde $\beta = 1/\sqrt{d_k}$. Esta constatação teórica revolucionou o domínio, revelando que os mecanismos de atenção não são meras heurísticas de ponderação, mas sim dinâmicas exatas de memória associativa contínua com capacidade de armazenamento exponencial.

Mais recentemente, a literatura evoluiu para a concepção de **arquiteturas inteiramente centradas em memória (*memory-centric architectures*)**, como a **Vision Hopfield Memory Network (V-HMN)** (Wang et al., 2026). Nestes sistemas, a memória associativa explícita de Hopfield deixa de ser um componente auxiliar e passa a ser o núcleo computacional primário de combinação de informação (*token mixing*), substituindo tanto as convoluções tradicionais como a autoatenção irrestrita através de hierarquias de memória local (*patch-level*), memória episódica global (*scene-level*) e regras de refinamento iterativo baseadas em **codificação preditiva (*predictive coding*)**.

---

### 1.2 Problema Fundamental que Aborda

A reinvenção e aplicação contemporânea das Redes de Hopfield no ecossistema de Deep Learning resolve limitações estruturais profundas dos paradigmas conexionistas convencionais:

1. **A Barreira da Capacidade de Armazenamento e o Colapso por Estados Espúrios:**
   - As redes clássicas de Hopfield (1982) sofriam de uma capacidade teórica restrita a $C \approx 0.138 d$ padrões. Se mais padrões fossem memorizados, a sobreposição das bacias de atração provocava interferência destrutiva catastrófica, gerando **mínimos espúrios (*spurious states*)** e estados metaestáveis que corrompiam completamente a recuperação dos dados.
   - As Modern Hopfield Networks demonstraram matematicamente que a substituição de funções de energia quadráticas por funções de energia exponenciais eleva a capacidade de armazenamento para $C \approx 2^{(d-1)/2}$ em espaços de dimensão $d$, permitindo armazenar milhões de padrões e recuperá-los de forma quase isenta de erros em apenas uma iteração.

2. **A Amnésia Estrutural e a Falta de Memória Explícita dos Modelos Feedforward e Transformers:**
   - Redes profundas convencionais (CNNs, ViTs) armazenam o conhecimento adquirido exclusivamente de forma **implícita e difusa** nos pesos sinápticos fixos ($W$). Uma vez concluído o treino, a rede processa estímulos sem memória explícita de exemplos pregressos e sem capacidade de recuperar instâncias episódicas específicas.
   - A introdução de camadas de Hopfield contínuas permite dissociar o cálculo dinâmico da retenção de conhecimento: a rede passa a dispor de bancos de protótipos de memória não-paramétricos que podem ser consultados, auditados e atualizados sem necessidade de alterar a parametrização profunda.

3. **A Ineficiência Extrema no Consumo de Dados (*Data Inefficiency*):**
   - Modelos como Vision Transformers (ViT) exigem dezenas ou centenas de milhões de imagens anotadas para aprender representações robustas, devido ao seu fraco viés indutivo inicial.
   - Ao incorporar módulos de memória associativa que armazenam protótipos de treino representativos em buffers circulares, arquiteturas como a V-HMN utilizam estes protótipos como *priors* indutivos não-paramétricos. Como resultado, alcançam desempenho competitivo com apenas 10% a 30% dos dados de treino, superando CNNs, ViTs, Swin-Transformers e modelos de espaço de estados (Vim/Mamba).

4. **A Ausência de Plausibilidade Biológica e de Correção Recursiva (*Predictive Coding*):**
   - Os modelos visuais dominantes processam os sinais estritamente de baixo para cima (*bottom-up* unidirecional). Em contraste, o córtex visual humano opera através de uma interação dinâmica contínua entre sinais sensoriais ascendentes e predições contextuais descendentes (*top-down*), minimizando recursivamente o erro de predição.
   - A combinação de Hopfield Networks com princípios neurocientíficos de *predictive coding* permite implementar o **refinamento iterativo de representações**: a ativação latente atual é corrigida em direção ao protótipo de memória mais plausível através da computação contínua do erro de predição.

5. **A Fragilidade Perante Ruído, Oclusões Parciais e Corrupção Sensorial:**
   - Em classificadores padrão, uma imagem degradada por ruído gaussiano ou ocluída em 50% da sua área sofre degradação catastrófica de precisão, pois os padrões de ativação locais colapsam.
   - Como as Redes de Hopfield foram concebidas desde a sua génese para recuperação associativa por completamento de padrões (*pattern completion*), a dinâmica de atração funciona como um **filtro dinâmico de desruído (*associative denoising*)**, projetando a entrada perturbada de volta para o protótipo canónico armazenado na bacia de atração.

---

### 1.3 Taxonomia e Principais Famílias de Abordagens no Estado da Arte

O espetro contemporâneo das Redes de Hopfield e Memória Associativa divide-se em cinco grandes famílias teóricas e aplicadas:

```
                      Taxonomia das Redes de Hopfield & Memória Associativa
                                                |
        +--------------------+------------------+------------------+--------------------+
        |                    |                  |                  |                    |
   1. Hopfield          2. Dense           3. Modern          4. Universal         5. Arquiteturas
      Clássicas &          Associative        Continuous         Hopfield &           Centradas em
      Modelos de Spin      Memories (DAM)     Hopfield (MHNs)    Controlo Episódico   Memória (V-HMN)
        |                    |                  |                  |                    |
   - Hopfield (1982)    - Krotov & Hopfield  - Ramsauer et al.  - Millidge et al.    - V-HMN (Wang 2026)
   - Estados binários     (2016, energias      (2020, ICLR 2021)  (UHN, 2022)        - Associative
   - Aprendizagem         polinomiais n > 2) - HopfieldLayer    - Chateau-Laurent &    Transformer (AiT)
     Hebbiana           - Demircigil et al.    diferenciável      Alexandre (2024,   - Memória Hierárquica
   - Capacidade linear    (2017, energia     - Associação com     NeurIPS - DND &      Local + Global
     C = 0.138 d          exponencial)         Transformers       Episodic Control)  - Predictive Coding
```

#### Família 1: Hopfield Networks Clássicas e Modelos de Spin Glass (1982 – 2015)
- **Princípio Operacional:** Redes recorrentes discretas inspiradas nos modelos de vidro de spin (*spin glasses*) da física estatística (modelo de Ising). Neurónios com ativação limiar binária $\{-1, +1\}$ e pesos hebbianos simétricos.
- **Modelos e Autores Canónicos:** Little (1974), Hopfield (1982, 1984), Amit, Gutfreund & Sompolinsky (1985).
- **Características e Limitações:** Comprovação formal de estabilidade via função de Lyapunov; contudo, capacidade mnemónica linear restrita ($C \approx 0.138 d$), incapacidade de lidar com dados contínuos de alta resolução e proliferação de mínimos locais espúrios.

#### Família 2: Dense Associative Memories (DAM) e Energias de Ordem Superior (2016 – 2019)
- **Princípio Operacional:** Substituição da função de energia quadrática por funções de interação sináptica de ordem superior (polinómios de grau $n > 2$ ou funções retificadoras não-lineares), aumentando drasticamente a curvatura dos poços de potencial em redor dos padrões memorizados.
- **Modelos e Autores Canónicos:** Krotov & Hopfield (2016, 2018), Demircigil et al. (2017).
- **Avanço Teórico:** Demonstração empírica e analítica de que potências superiores $F(x) = x^n$ elevam a capacidade de armazenamento para regimes polinomiais e, no limite exponencial $F(x) = \exp(x)$, para capacidades exponenciais em função do número de nós.

#### Família 3: Modern Continuous Hopfield Networks e Camadas Diferenciáveis (2020 – Presente)
- **Princípio Operacional:** Generalização das DAMs para estados contínuos em $\mathbb{R}^d$, integrando funções de energia baseadas em Log-Sum-Exp e provando que a atualização dinâmica em um passo corresponde matematicamente à operação de *softmax attention*.
- **Modelos e Autores Canónicos:** Ramsauer et al. (ICLR 2021 / arXiv 2020), Widrich et al. (NeurIPS 2020 - Hopfield para biosequências e imuno-oncologia), Schäfl et al. (2022 - Hopfield para aprendizagem por poucos exemplos).
- **Avanço Teórico:** Implementação de camadas `HopfieldLayer` perfeitamente diferenciáveis que podem ser inseridas em qualquer arquitetura de aprendizagem profunda (CNNs, MLPs, Grafos), sendo treinadas ponta a ponta via retropropagação do erro (*backpropagation*).

#### Família 4: Universal Hopfield Networks e Controlo Episódico em RL (2022 – Presente)
- **Princípio Operacional:** Formalização unificada que enquadra múltiplos esquemas mnemónicos (incluindo memórias endereçáveis por conteúdo, dicionários neurais e redes de memória) como instâncias de uma única classe teórica denominada *Universal Hopfield Networks (UHN)*.
- **Modelos e Autores Canónicos:** Millidge et al. (2022), Chateau-Laurent & Alexandre (NeurIPS 2024 - *Relating Hopfield Networks to Episodic Control*).
- **Avanço Teórico:** Prova matemática de que o Dicionário Neural Diferenciável (*Differentiable Neural Dictionary - DND*) utilizado no Controlo Episódico (*Neural Episodic Control*) em Aprendizagem por Reforço é, rigorosamente, uma UHN com regularização $L_2$ e kernel métrico customizado (Manhattan vs. Euclidiano), oferecendo base neurobiológica plausível com base no circuito hipocampal CA3-CA1.

#### Família 5: Arquiteturas Visuais Centradas em Memória (*Memory-Centric Vision Backbones*) (2024 – 2026)
- **Princípio Operacional:** Transformação da memória associativa de Hopfield no mecanismo estrutural central de integração espacial e contextual da visão computacional, substituindo a autoatenção quadrática e as convoluções locais.
- **Modelos e Autores Canónicos:** Wang, M'Charrak, Koska, Lukasiewicz et al. (2026 - *Vision Hopfield Memory Networks - V-HMN*), Associative Transformer (AiT, 2023).
- **Avanço Metodológico:** Combinação hierárquica de memórias Hopfield locais (*Local Window Memory* para desruído e textura ao nível de *patches*) com memórias Hopfield globais (*Global Template Path* como memória episódica de cena), integradas por uma dinâmica recorrente de *predictive coding* que refina iterativamente os mapas de ativação.

---

### 1.4 Dicionário de Conceitos-Chave (Pedagógico e Rigoroso)

*   **Memória Associativa / Endereçável por Conteúdo (*Content-Addressable Memory*):**  
    Dispositivo ou algoritmo mnemónico que recupera um padrão ou vetor armazenado na íntegra a partir da apresentação de uma versão parcial, degradada, incompleta ou ruidosa do mesmo, guiando-se pela similaridade intrínseca do conteúdo e não por um endereço físico de memória.
*   **Atrator / Bacia de Atração (*Attractor / Basin of Attraction*):**  
    Ponto de equilíbrio estável no espaço de estados para onde convergem as trajetórias dinâmicas do sistema. A bacia de atração representa a região de vizinhança geométrica no espaço de estados em que qualquer vetor inicial colocado dentro dessa região será atraído para o mesmo padrão mnemónico final.
*   **Função de Energia / Superfície de Lyapunov (*Lyapunov Energy Function*):**  
    Função escalar definida sobre o espaço de estados do sistema dinâmico que decresce estritamente (ou permanece constante) a cada iteração temporal ($E(z_{t+1}) \le E(z_t)$), garantindo matematicamente a convergência assintótica para um estado estacionário e impossibilitando oscilações caóticas infinitas.
*   **Mínimos Espúrios / Estados Metastáveis (*Spurious States / Metastable States*):**  
    Pontos fixos ou vales locais na superfície de energia que não correspondem a nenhum dos padrões originalmente pretendidos para armazenamento, resultantes de interferências e sobreposições lineares entre padrões memorizados. Constituem o principal fator de degradação da fidelidade mnemónica.
*   **Capacidade de Armazenamento (*Storage Capacity*):**  
    Número máximo de padrões independentes de dimensão $d$ que uma rede de memória associativa consegue memorizar e recuperar com uma taxa de erro inferior a um limiar fixo pré-estabelecido. Cresce linearmente ($C \propto d$) em Hopfield clássicas e exponencialmente ($C \propto 2^{d/2}$ ou $C \propto \alpha^d$) em Modern Hopfield Networks.
*   **Equivalência Transformer-Hopfield (*Transformer-Hopfield Equivalence*):**  
    Descoberta matemática que estabelece que o mecanismo de autoatenção ponderada por produto escalar escalado dos Transformers é formalmente idêntico a um passo de atualização de atração de uma Modern Hopfield Network contínua governada por uma função de energia Log-Sum-Exp.
*   **Módulo Hopfield Local (*Local Window Memory*):**  
    Componente arquitetural de visão que restringe a dinâmica associativa de Hopfield a janelas espaciais confinadas ($N \times N$) em redor de cada bloco (*patch*) da imagem, consultando um banco de memória local para filtrar ruído de alta frequência e completar bordas e texturas com fidelidade topológica.
*   **Módulo Hopfield Global (*Global Template Path*):**  
    Mecanismo que agrega representações espaciais completas da imagem (via *mean-pooling* global) para gerar uma consulta holística a nível de cena, recuperando um protótipo temático global a partir de um banco de memória episódica que modula contextualmente todas as camadas subsequentes da rede.
*   **Refinamento Iterativo & Codificação Preditiva (*Iterative Refinement & Predictive Coding*):**  
    Princípio bio-inspirado no qual o processamento visual computacional não é puramente unidirecional (*feedforward*), mas sim um processo recorrente no qual a representação latente do estímulo ($h_t$) é progressivamente ajustada em direção ao protótipo de memória mais compatível ($p$) através da minimização do sinal do erro de predição ($e = p - h_t$) ponderado por uma taxa de atualização $\alpha$.
*   **Buffer Circular Balanceado por Classe (*Class-Balanced Ring Buffer*):**  
    Estrutura mnemónica não-paramétrica que mantém em memória um número fixo de vetores prototípicos latentes extraídos de amostras reais para cada classe de dados. Funciona sob política FIFO (*First-In, First-Out*) durante a fase de treino para acompanhar a evolução das representações e é congelado (*frozen*) durante a inferência.
*   **Protótipos de Memória (*Memory Prototypes*):**  
    Vetores de características representativos armazenados fisicamente na memória da rede que sintetizam características semânticas canónicas de classes ou partes visuais. Fundamentam a **interpretabilidade baseada em protótipos**, permitindo que investigadores humanos inspecionem visualmente os exemplos exatos que ditaram a classificação de uma imagem.
*   **Dicionário Neural Diferenciável (*Differentiable Neural Dictionary - DND*):**  
    Módulo de memória externa utilizado em algoritmos de Aprendizagem por Reforço (como *Neural Episodic Control*) que armazena pares chave-valor (estado latente $\to$ valor $Q$), permitindo gravação rápida e consulta diferenciável via interpolação de vizinhos mais próximos com kernels métricos.
*   **Função de Separação Max (*Max Separation Function*) vs. Softmax:**  
    Mecanismos de normalização da atenção mnemónica. A função Max atribui peso total exclusivamente ao padrão mais similar no espaço latente, enquanto a função Softmax pondera probabilisticamente todos os padrões de acordo com a temperatura $\beta$, demonstrando empiricamente maior capacidade de generalização e suavidade de gradiente.

---

## 2. Protocolo de Pesquisa Sistemática (Search Strategy)

Para orientar agentes autónomos e investigadores na prospeção exaustiva e rigorosa da literatura sobre Redes de Hopfield e Memória Associativa, estabelece-se o seguinte protocolo padronizado.

### 2.1 Venues Científicos Prioritários

A pesquisa bibliográfica deve circunscrever-se rigorosamente a veículos submetidos a arbitragem científica por pares de nível internacional de excelência (*Core A\**, *IEEE*, *ACM*, *Springer*, *Elsevier*):

| Categoria | Sigla / Nome do Venue | Foco Temático e Relevância |
|:---|:---|:---|
| **Conferências Principais de Aprendizagem Automática (Core A\*)** | **NeurIPS** (Neural Information Processing Systems) | Modern Hopfield Networks, Universal Hopfield, DAMs e Controlo Episódico |
| | **ICML** (International Conference on Machine Learning) | Otimização de energia, estabilidade assintótica e algoritmos mnemónicos |
| | **ICLR** (International Conference on Learning Representations) | Conexão Hopfield-Transformer, camadas contínuas e representações latentes |
| | **AAAI** (Association for the Advancement of Artificial Intelligence) | Aplicações abrangentes de memória associativa e modelos híbridos |
| **Conferências de Visão Computacional (Core A\*)** | **CVPR** (Computer Vision and Pattern Recognition) | Backbones centrados em memória, V-HMN, robustez a oclusão e interpretabilidade |
| | **ICCV / ECCV** (Int. / European Conf. on Computer Vision) | Módulos de atenção associativa, protótipos visuais e segmentação |
| **Revistas Científicas de Alto Impacto** | **IEEE T-PAMI** (Trans. Pattern Analysis and Machine Intelligence) | Formulações matemáticas profundas em visão e modelos de memória |
| | **IEEE T-NNLS** (Trans. Neural Networks and Learning Systems) | Teoria de sistemas dinâmicos, funções de Lyapunov e redes recorrentes |
| | **Neural Computation** (MIT Press) | Veículo histórico seminal das redes de Hopfield clássicas e novos modelos biológicos |
| | **Biological Cybernetics / Nature Machine Intelligence** | Modelagem bio-inspirada, predictive coding, hipocampo e biomimetismo cortical |
| | **JMLR** (Journal of Machine Learning Research) | Demonstrações formais de capacidade mnemónica e convergência global |
| **Repositórios de Preprints Verificados** | **arXiv** (`cs.LG`, `cs.NE`, `cs.CV`, `cs.AI`, `q-bio.NC`) | Descobertas de ponta dos últimos 12 a 24 meses (ex.: V-HMN, UHN) |

---

### 2.2 Equações de Pesquisa Booleanas Recomendadas

As equações infra encontram-se desenhadas para interrogação em motores científicos (Google Scholar, Semantic Scholar, IEEE Xplore, ScienceDirect, ACM Digital Library e Scopus):

#### Bloco A: Modern Hopfield Networks, Estados Contínuos e Equivalência com Atenção
```text
("Modern Hopfield Network" OR "Modern Hopfield Networks" OR "HopfieldLayer" OR "Dense Associative Memory") AND ("continuous states" OR "energy function" OR "Lyapunov" OR "concave-convex procedure") AND ("transformer" OR "attention mechanism" OR "softmax" OR "storage capacity")
```

#### Bloco B: Arquiteturas Visuais Centradas em Memória (Memory-Centric Backbones & V-HMN)
```text
("Hopfield" OR "associative memory") AND ("vision backbone" OR "image recognition" OR "token mixing") AND ("local window" OR "global template" OR "predictive coding" OR "V-HMN" OR "prototype")
```

#### Bloco C: Universal Hopfield Networks, Controlo Episódico e Reinforcement Learning
```text
("Universal Hopfield Network" OR "Hopfield Networks") AND ("episodic control" OR "Neural Episodic Control" OR "differentiable neural dictionary" OR "DND") AND ("reinforcement learning" OR "memory retrieval" OR "Manhattan distance")
```

#### Bloco D: Codificação Preditiva, Refinamento Iterativo e Inspiração Biológica
```text
("associative memory" OR "Hopfield network") AND ("predictive coding" OR "iterative refinement" OR "error correction") AND ("hippocampus" OR "cortical microcircuits" OR "brain-inspired")
```

#### Bloco E: Robustez a Ruído, Oclusão, Interpretabilidade e Eficiência com Poucos Dados
```text
("Hopfield" OR "associative memory") AND ("robustness" OR "noise resistance" OR "occlusion" OR "data efficiency" OR "few-shot") AND ("prototype-based interpretability" OR "pattern completion")
```

---

### 2.3 Janela Temporal de Análise

A prospeção da literatura sobre Redes de Hopfield deve obedecer a uma perspetiva temporal estruturada em duas janelas críticas:

1. **Janela Seminal e Fundacional (1982 – 2019):**
   - **Objetivo:** Compreender os fundamentos termodinâmicos, os limites de capacidade e as primeiras quebras de paradigma em direção a memórias densas.
   - **Marcos Históricos Obrigatórios:**
     - Hopfield (1982 - Formulação seminal de redes de memória associativa com função de energia);
     - Amit, Gutfreund & Sompolinsky (1985 - Teoria de spin glasses e capacidade $0.138 d$);
     - Hopfield (1984 - Extensão para neurónios com ativação contínua e dinâmica diferencial);
     - Krotov & Hopfield (2016 - *Dense Associative Memory for Pattern Recognition*, introdução de termos de energia polinomiais de ordem superior);
     - Demircigil et al. (2017 - Prova de capacidade exponencial sob energias exponenciais).

2. **Janela de Avanços Recentes e Estado da Arte (Últimos 3 a 5 anos: 2020 – 2026):**
   - **Objetivo:** Rastrear a integração de Hopfield com Transformers, a formalização das UHN, e a sua aplicação como backbones centrais em visão e robótica.
   - **Marcos Recentes Obrigatórios:**
     - Ramsauer et al. (ICLR 2021 / arXiv 2020 - *Hopfield Networks is All You Need*, formulação contínua e prova da equivalência exata com o mecanismo de atenção de Transformers);
     - Widrich et al. (NeurIPS 2020 - *Modern Hopfield Networks and Attention for Immune Repertoire Classification*);
     - Millidge et al. (2022 - *Universal Hopfield Networks*);
     - Chateau-Laurent & Alexandre (NeurIPS 2024 - *Relating Hopfield Networks to Episodic Control*, conexão formal com DND e controle episódico);
     - Wang, Lukasiewicz et al. (2026 - *Vision Hopfield Memory Networks for Image Recognition - V-HMN*, backbones visuais centrados em memória com *predictive coding*).

---

## 3. Critérios de Elegibilidade (Inclusão e Exclusão)

Para manter a consistência e a excelência científica na seleção de literatura, cada artigo identificado deve ser submetido ao seguinte fluxo decisório e conjunto de critérios formais:

```
                      Artigo Identificado na Pesquisa
                                     |
               +---------------------+---------------------+
               |                                           |
               v                                           v
      Critérios de Inclusão (+)                  Critérios de Exclusão (-)
   - Função de Lyapunov formal                - Sem função de energia matemática
   - Baselines contemporâneos                 - Baselines arcaicos isolados (apenas Hopfield 1982)
   - Avaliação de eficiência / capacidade     - Opacidade nos protótipos / Data Leakage
   - Protocolo de teste aberto                - Artigos puramente opinativos sem validação
               |                                           |
               v                                           v
       ACEITE NO REPOSITÓRIO                       REJEITADO
```

### 3.1 Critérios de Inclusão (+)
O artigo deve preencher cumulativamente pelo menos **três** dos seguintes requisitos:
1. **Rigor e Formalismo Matemático:** Definição explícita da função de energia de Lyapunov subjacente, demonstração de convergência assintótica da regra de atualização proposta e análise da capacidade mnemónica de armazenamento.
2. **Comparação com Baselines Modernos Consolidados:** Em tarefas de visão ou aprendizagem por representação, o modelo deve ser confrontado não apenas contra modelos clássicos, mas contra arquiteturas dominantes no estado da arte: CNNs (ResNet, ConvNeXt), Vision Transformers (ViT, Swin-ViT), Modelos Lineares (MLP-Mixer) e Modelos de Espaço de Estados (Mamba/Vim).
3. **Avaliação Abrangente de Eficiência e Trade-offs:** Relato explícito de métricas computacionais além da acurácia: contagem de parâmetros, FLOPs/MACs, tempo de latência por iteração de refinamento, tamanho dos bancos de memória e consumo de VRAM.
4. **Avaliação em Regimes Críticos (Poucos Dados, Ruído ou Oclusão):** Demonstração empírica das propriedades distintivas da memória associativa, tais como generalização com frações reduzidas de dados de treino (10% a 30%), robustez contra ruído gaussiano/sal-e-pimenta ou oclusão estruturada de imagens.
5. **Transparência e Reprodutibilidade:** Descrição cabal dos hiperparâmetros mnemónicos: temperatura $\beta$, dimensões de chave/valor, tamanho do buffer por classe, esquema de atualização FIFO e disponibilidade de código/pesos.

### 3.2 Critérios de Exclusão (-)
Devem ser prontamente descartados os trabalhos que apresentem qualquer uma das seguintes debilidades:
1. **Heurísticas Ad-Hoc Desprovidas de Função de Energia:** Trabalhos que utilizem o termo "Hopfield" apenas como metáfora discursiva, sem demonstrar a existência de uma função de Lyapunov que governe a dinâmica do sistema.
2. **Avaliações com Baselines Arcaicos:** Estudos contemporâneos que comparam a sua rede exclusivamente com a formulação binária de Hopfield (1982) ou Perceptrons simples, ignorando os avanços de Modern Hopfield Networks e Transformers.
3. **Contaminação de Amostras de Teste no Banco de Memória (*Data Leakage*):** Casos em que amostras pertencentes às partições de validação ou teste foram acidentalmente incorporadas no buffer de protótipos de memória, falseando a generalização.
4. **Publicações Sem Arbitragem Científica Idónea:** Artigos provenientes de editoras predatórias desprovidas de revisão rigorosa por pares, ou notas técnicas conceituais sem qualquer validação empírica ou simulação computacional.

---

## 4. Roteiro de Extração e Síntese Analítica

Para cada artigo selecionado, o curador autónomo ou investigador deve preencher rigorosamente a **Checklist Analítica de 5 Pontos**:

```
+---------------------------------------------------------------------------------+
|               CHECKLIST ANALÍTICA DE EXTRAÇÃO DE ARTIGOS                        |
+---------------------------------------------------------------------------------+
| 1. Problema e Motivação                                                         |
|    - Que estrangulamento mnemónico ou computacional o artigo desafia?           |
|    - Por que motivo falham as redes feedforward ou os Transformers na tarefa?   |
+---------------------------------------------------------------------------------+
| 2. Inovação Metodológica / Arquitetural                                         |
|    - Qual a formulação matemática da função de energia e regra de atualização?  |
|    - Como se estruturam os bancos de memória (locais, globais, ring buffers)?   |
+---------------------------------------------------------------------------------+
| 3. Datasets e Protocolo de Avaliação                                            |
|    - Quais os benchmarks testados (ImageNet, CIFAR, benchmarks de RL)?          |
|    - Quais os regimes de dados (100% vs. few-shot 10-30%, testes sob ruído)?    |
+---------------------------------------------------------------------------------+
| 4. Resultados Empíricos e Trade-offs (Precisão vs. Eficiência)                  |
|    - Quais os ganhos contra baselines dominantes (ViT, ResNet, Swin, DND)?      |
|    - Qual a sobrecarga de parâmetros de memória e tempo de refinamento?         |
+---------------------------------------------------------------------------------+
| 5. Limitações e Desafios em Aberto (Open Challenges)                           |
|    - Como escala a memória face ao aumento linear ou exponencial de classes?    |
|    - Que custos de latência e consumo de VRAM persistem na inferência iterativa?|
+---------------------------------------------------------------------------------+
```

### Detalhe Operacional da Checklist:

1. **Problema e Motivação:**
   - Registar o estrangulamento teórico ou prático: ex., ineficiência de dados dos Vision Transformers que requerem centenas de épocas em bases gigantes; ausência de plausibilidade biológica nas redes puramente feedforward; falta de base teórica unificada entre memória episódica em RL e modelos de memória associativa.
2. **Inovação Metodológica/Arquitetural:**
   - Isolar a formulação matemática exata: explicitar a função de energia, o valor ou calibração da temperatura $\beta$, a estrutura do banco de protótipos (tamanho do *ring buffer*, dimensão latente), a separação em caminhos locais e globais, e a regra de atualização por *predictive coding* ($h_{t+1} = h_t + \alpha (p - h_t)$).
3. **Datasets e Protocolo de Avaliação:**
   - Catalogar as bases de dados: ImageNet-1K, CIFAR-10/100, TinyImageNet, ambientes de RL (Atari, GridWorld) e suites de robustez (ImageNet-C, ImageNet-A para corrupções e ruído).
4. **Resultados Empíricos e Trade-offs:**
   - Quantificar os ganhos absolutos e percentuais: ex., melhoria de precisão Top-1 com 10% de dados de treino; redução de parâmetros comparativamente a ViT-B; métricas sob ruído e oclusão. Reportar os *trade-offs*: se o refinamento iterativo requer 2 a 3 passos adicionais de cálculo, quantificar o impacto nos milissegundos por imagem.
5. **Limitações e Desafios em Aberto:**
   - Assinalar de forma crítica os desafios por resolver: escalabilidade do buffer de protótipos em tarefas de granularidade extrema (milhares de classes); velocidade de atualização do banco em ambientes dinâmicos contínuos; adaptação para tarefas densas de segmentação em tempo real.

---

## 5. Padrão de Formatação para a Tabela de Literatura (`README.md`)

O ficheiro `README.md` localizado neste diretório serve como registo documental consolidado da literatura curada. Todas as adições devem cumprir estritamente a especificação de 6 colunas descrita de seguida.

### 5.1 Especificação Rigorosa das 6 Colunas Obrigatórias

| Coluna | Título | Conteúdo Obrigatório e Regras Formais |
|:---:|:---|:---|
| **1** | **Nome** | Título oficial completo do artigo em língua inglesa, exatamente como registado pelo editor. |
| **2** | **Detalhes** | Metadados catalográficos estruturados verticalmente com `<br><br>`: Autores, Data de publicação, Publisher, Livro/Journal/Conferência, Volume, Número (se houver), Páginas e link oficial do DOI. |
| **3** | **Abstract** | Excertos textuais literais retirados diretamente do resumo original do artigo, colocados estritamente entre aspas duplas (`"..."`). Devem abranger: motivação, proposta metodológica e conclusão preliminar. |
| **4** | **Conclusion** | Excertos textuais literais extraídos da secção final de conclusões do artigo original, colocados estritamente entre aspas duplas (`"..."`). |
| **5** | **Resumo (NotebookLM)** | Síntese analítica em língua portuguesa estruturada impreterivelmente em **quatro parágrafos encadeados** separados por `<br><br>` (conforme especificado na secção 5.2). |
| **6** | **Citação** | Referência bibliográfica completa elaborada segundo a **Norma Vancouver (NLM)**, finalizada com a hiperligação HTTPS ativa para o DOI. |

---

### 5.2 Estrutura Padronizada do "Resumo (NotebookLM)" (4 Parágrafos)

A coluna 5 deve obedecer impreterivelmente à seguinte progressão de 4 parágrafos:

*   **Parágrafo 1 — Contexto e Problema de Investigação:**  
    Apresentação do artigo, contextualização da memória associativa ou redes de Hopfield no desafio em causa e delimitação clara do estrangulamento técnico enfrentado pelas arquiteturas vigentes.
*   **Parágrafo 2 — Mecanismo Proposto e Inovação Arquitetural:**  
    Descrição aprofundada da solução concebida pelos autores (ex.: introdução de módulos Hopfield locais/globais, conexão formal com UHN ou DND, regra de refinamento por predictive coding, formulação de novos kernels métricos ou dinâmica de atualização mnemónica).
*   **Parágrafo 3 — Validação Experimental e Métricas Numéricas:**  
    Detalhe dos conjuntos de dados utilizados, baselines de comparação direta e quantificação rigorosa dos resultados obtidos (valores numéricos de precisão, retenção mnemónica, desempenho em regimes de escassez de dados ou robustez a ruído).
*   **Parágrafo 4 — Significado Teórico, Limitações e Desafios:**  
    Avaliação do impacto científico para o avanço da inteligência artificial, benefícios práticos (interpretabilidade por protótipos, plausibilidade biológica, economia de dados) e limitações abertas identificadas para trabalho futuro.

---

### 5.3 Exemplo Canónico de Registo na Tabela

Para servir de padrão de conformidade e integridade bibliográfica, apresenta-se o registo canónico integral construído a partir do artigo seminal recente de Wang et al. (2026), presente no `README.md` deste diretório:

```markdown
| Vision Hopfield Memory Networks for Image Recognition | **Autores:** Jianfeng Wang, Amine M'Charrak, Luk Koska, Xiangtao Wang, Daniel Petriceanu, Ruizhi Wang, Michael Bumbar, Luca Pinchetti, Thomas Lukasiewicz<br><br>**Data de publicação:** 2026<br><br>**Publisher:** arXiv<br><br>**Livro/Journal:** preprint arXiv:2603.25157<br><br>**DOI:** https://doi.org/10.48550/arXiv.2603.25157 | "In this work, we propose the Vision Hopfield Memory Network (V-HMN), a brain-inspired vision backbone that integrates hierarchical memory mechanisms across layers with iterative refinement updates."<br><br>"Specifically, V-HMN incorporates local Hopfield modules that provide associative memory dynamics at the image patch level, global Hopfield modules that function as episodic memory for contextual modulation, and a predictive-coding-inspired refinement rule for iterative error correction."<br><br>"V-HMN achieves strong performance on small- and medium-scale benchmarks, and remains competitive with widely adopted backbone architectures on ImageNet despite minimal architectural tuning, while offering improved data efficiency and a prototype-based form of interpretability." | "In this work, we introduced V-HMN, a brain-inspired vision backbone that organizes each block around local and global Hopfield-style memory."<br><br>"Through associative retrieval and predictive-coding-inspired refinement, V-HMN moves beyond purely feedforward or self-attention architectures and places memory at the center of feature integration."<br><br>"This yields two key benefits: data efficiency, by reusing stored prototypes as inductive priors, and interpretability, as retrieved prototypes provide an inspectable trace of the patterns involved in each prediction." | Este artigo propõe a Vision Hopfield Memory Network (V-HMN), uma nova arquitetura de visão artificial inspirada no funcionamento do cérebro humano, que afasta as convenções dos modelos *feedforward* ou estritamente baseados em *self-attention*.<br><br>A V-HMN coloca a memória no centro do processamento através de módulos de memória Hopfield: módulos locais para refinar detalhes visuais ao nível dos *patches* da imagem e módulos globais que funcionam como uma memória episódica para modulação do contexto.<br><br>O modelo aplica um processo de refinamento iterativo para correção de erros inspirado na teoria do *predictive-coding*.<br><br>Esta abordagem demonstra ser altamente eficiente na utilização de dados (reutilizando protótipos previamente armazenados) e providencia uma maior interpretabilidade das decisões do modelo, alcançando resultados competitivos com arquiteturas de topo (como no ImageNet) mesmo com poucos dados de treino. | Wang J, M'Charrak A, Koska L, Wang X, Petriceanu D, Wang R, et al. Vision Hopfield Memory Networks for Image Recognition. arXiv preprint arXiv:2603.25157; 2026. https://doi.org/10.48550/arXiv.2603.25157. |
```

---

## 6. Guardrails de Qualidade & Anti-Alucinação

Para assegurar a idoneidade, fidelidade e reprodutibilidade do acervo científico curado, investigadores humanos e agentes de IA devem observar sem exceções os seguintes guardrails:

### 6.1 Validação Rigorosa de DOIs e Metadados
- **Verificação Ativa da Resolução do DOI:** Todas as hiperligações de DOI devem ser testadas e resolver diretamente através do protocolo canónico `https://doi.org/10.xxxx/...`. É estritamente interdito inventar, truncar ou presumir identificadores DOI inexistentes.
- **Auditoria Cruzada de Metadados:** Os nomes dos autores, título da conferência ou periódico, paginação, volume e ano devem ser rigorosamente validados contra bases primárias (CrossRef, DBLP, arXiv, NeurIPS Proceedings, IEEE Xplore).

### 6.2 Proibição Estrita de Interpolação ou Arredondamento Fraudulento de Métricas
- **Exatidão Numérica Inviolável:** Valores de acurácia, capacidade de armazenamento, número de parâmetros, FLOPs ou métricas de erro devem ser transcritos exatamente como reportados no texto original. Se o artigo indica uma acurácia de $83.4\%$ sob 20% dos dados, é proibido arredondar para "cerca de 83%" ou "aproximadamente 84%".
- **Distinção de Configurações:** Registar com rigor sob que condições de teste (ex.: número de épocas, tamanho de lote, parâmetros de temperatura $\beta$) as métricas foram alcançadas.

### 6.3 Fidelidade Literal das Citações Diretas
- Os campos **Abstract** e **Conclusion** destinam-se exclusivamente a transcrições textuais diretas do artigo original, entre aspas duplas (`"..."`). É proibido resumir com palavras próprias, omitir partes essenciais sem reticências ou inserir considerações analíticas nestas colunas.

### 6.4 Sobriedade Terminológica e Neutralidade Epistemológica
- As sínteses e análises críticas devem manter rigor e sobriedade científica, abolindo termos sensacionalistas (ex.: "método milagroso", "precisão extraordinária", "solução perfeita"). O discurso deve focar-se em formulações matemáticas, dinâmicas de convergência de sistemas, capacidades de generalização e compromissos operacionais (*trade-offs* de precisão vs. custo computacional).
