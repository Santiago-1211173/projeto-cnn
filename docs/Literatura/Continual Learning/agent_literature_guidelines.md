# Protocolo de Curadoria e Navegação na Literatura: Continual Learning

> **Ficheiro Central Associado:** `README.md` (no mesmo diretório)  
> **Natureza do Documento:** Guia metodológico de recolha, análise crítica e referenciação científica para agentes autónomos e investigadores.  
> **Domínio Científico:** Inteligência Artificial / Aprendizagem Automática / Aprendizagem Contínua (Lifelong Learning) e Adaptação a Fluxos de Dados (*Data Streams*).

---

## 1. Enquadramento Teórico & Âmbito do Tópico

### 1.1 Definição Canónica na Literatura Internacional de IA
O paradigma da **Aprendizagem Contínua** (*Continual Learning* - CL), frequentemente denominado **Aprendizagem ao Longo da Vida** (*Lifelong Learning* - LL) ou **Aprendizagem Incremental** (*Incremental Learning*), define a capacidade de um sistema de inteligência artificial adquirir, atualizar, consolidar e explorar novos conhecimentos e competências ao longo de uma sequência temporal de dados ou tarefas, sem deteriorar de forma inaceitável o desempenho nas competências previamente aprendidas.

Em termos formais, enquanto a Aprendizagem Supervisionada clássica assume um conjunto estático de dados $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^N$ amostrado de uma distribuição estacionária e invariante $P(X, Y)$, a Aprendizagem Contínua modela um processo sequencial de treino ao longo de um conjunto de tarefas ou períodos de tempo $k = 1, 2, \dots, K$:
$$\mathcal{S} = \langle \mathcal{D}_1, \mathcal{D}_2, \dots, \mathcal{D}_K \rangle$$
onde cada conjunto $\mathcal{D}_k = \{(\mathbf{x}_i^{(k)}, y_i^{(k)})\}_{i=1}^{N_k}$ é governado por uma distribuição de probabilidade potencialmente distinta $P_k(X, Y)$. 

No instante temporal correspondente à tarefa $k$, o algoritmo tem acesso primário apenas aos dados de $\mathcal{D}_k$ (e, eventualmente, a uma memória episódica de capacidade rigorosamente limitada $\mathcal{M}$, onde $|\mathcal{M}| \ll \sum_{j=1}^{k-1} N_j$). O objetivo fundamental do modelo $f(\cdot; \theta)$, parametrizado pelo vetor de pesos $\theta \in \mathbb{R}^P$, é minimizar a perda empírica acumulada em todas as tarefas observadas até ao presente:
$$\min_{\theta} \sum_{j=1}^k \mathbb{E}_{(\mathbf{x}, y) \sim \mathcal{D}_j} \left[ \mathcal{L}(f(\mathbf{x}; \theta), y) \right]$$
sem ter permissão para re-treinar o modelo a partir do zero (*from scratch*) utilizando o histórico integral de todos os dados passados $\bigcup_{j=1}^k \mathcal{D}_j$.

```
+-----------------------------------------------------------------------------------+
|               PARADIGMA CLÁSSICO: BATCH / EPISÓDICO (OFFLINE)                     |
|                                                                                   |
|    Dados Históricos Integrais                                                     |
|    [ D1 + D2 + ... + Dk ] --------> Treino Global Estático --------> Modelo Fixo  |
|                                                                     (Congelado)   |
+-----------------------------------------------------------------------------------+

+-----------------------------------------------------------------------------------+
|               PARADIGMA DE APRENDIZAGEM CONTÍNUA (LIFELONG LEARNING)              |
|                                                                                   |
|    Fluxo Sequencial de Dados       Memória Finita (ERB/Coreset)                   |
|    [ D1 ] -> [ D2 ] -> [ Dk ]             [ M ]                                   |
|       |         |         |                 |                                     |
|       v         v         v                 v                                     |
|    ========================================================                       |
|    Agente Contínuo: Atualização Incremental sob Restrições                        |
|    - Preservação de D1...Dk-1 (Estabilidade)                                      |
|    - Incorporação de Dk (Plasticidade)                                            |
|    - Eficiência Computacional e Energética Sustentável                            |
|    ========================================================                       |
+-----------------------------------------------------------------------------------+
```

---

### 1.2 Problema Fundamental que Aborda
A investigação em *Continual Learning* aborda as fraquezas estruturais das redes neuronais artificiais profundas (*Deep Neural Networks* - DNNs) perante distribuições de dados dinâmicas e não-estacionárias no mundo real.

#### 1. O Fenómeno do Esquecimento Catastrófico (*Catastrophic Forgetting*)
Quando uma rede neuronal treinada com gradiente descendente na tarefa $\mathcal{D}_1$ é posteriormente submetida ao treino na tarefa $\mathcal{D}_2$ sem mecanismos de salvaguarda, os pesos sinápticos $\theta$ sofrem alterações drásticas para minimizar o erro em $\mathcal{D}_2$. Como as representações nas redes profundas são distribuídas e densamente sobrepostas, esta atualização sobrescreve as configurações de parâmetros responsáveis pelo conhecimento de $\mathcal{D}_1$. O resultado é o colapso abrupto e catastrófico da exatidão preditiva nas tarefas anteriores, conhecido na literatura fundacional como *Catastrophic Interference*.

#### 2. O Dilema Estabilidade-Plasticidade (*Stability-Plasticity Dilemma*)
Este dilema constitui o conflito conceptual basilar da neurociência computacional e da aprendizagem contínua:
- **Plasticidade:** A capacidade do sistema neuronal integrar novas informações e adaptar-se velozmente a novos conceitos, classes ou dinâmicas ambientais.
- **Estabilidade:** A capacidade de manter e reter intactas as representações e competências essenciais consolidadas no passado.
O excesso de plasticidade provoca esquecimento imediato; o excesso de estabilidade resulta em rigidez do modelo, bloqueando a assimilação de novos padrões.

#### 3. Desvio de Conceito (*Concept Drift*) em Fluxos Contínuos de Dados
Em sistemas de inferência em tempo real e análise de fluxos (*data streams*), as distribuições estatísticas subjacentes alteram-se com o tempo ($P_{t1}(X, Y) \neq P_{t2}(X, Y)$):
- **Real Concept Drift:** Modificação da relação condicional entre as características e os rótulos ($P(Y \mid X)$ altera-se), provocando a obsolescência das fronteiras de decisão do classificador, mesmo que a distribuição das entradas $P(X)$ permaneça similar.
- **Virtual Concept Drift:** Modificação exclusiva da distribuição marginal das características de entrada ($P(X)$ altera-se), mantendo-se a distribuição condicional $P(Y \mid X)$ inalterada.

#### 4. Cenários Canónicos de Avaliação em Continual Learning
A literatura divide formalmente as tarefas de aprendizagem contínua em três configurações estruturais com diferentes níveis de complexidade:
*   **Task-Incremental Learning (Task-IL):** O modelo é explicitamente informado sobre qual a tarefa que está a resolver através de um identificador (*task-ID* $k$), tanto durante a fase de treino como no momento da inferência/teste. Permite o uso de cabeças de saída isoladas (*multi-head*).
*   **Domain-Incremental Learning (Domain-IL):** O conjunto de rótulos possíveis permanece fixo entre as tarefas, mas a estrutura dos dados de entrada sofre transformações profundas de domínio (ex.: transição de imagens em estilo diurno para estilo noturno). O *task-ID* não é facultado no teste.
*   **Class-Incremental Learning (Class-IL):** Novas classes de saída são adicionadas sequencialmente em cada etapa. Durante o teste, o modelo deve classificar instâncias entre todas as classes observadas até à data, sem qualquer pista sobre a qual tarefa pertence a instância avaliada. Constitui o cenário mais exigente e representativo da cognição no mundo real.
*   **Online Continual Learning / Data Stream Learning:** Os dados chegam num fluxo ininterrupto de instâncias individuais ou pequenos lotes (*mini-batches*), sendo processados numa única passagem (*single-pass*), sem limites artificiais de tarefas (*task-free continual learning*).

---

### 1.3 Taxonomia e Principais Famílias de Abordagens no Estado da Arte

As estratégias de combate ao esquecimento catastrófico e viabilização de aprendizagem contínua agrupam-se em quatro famílias teóricas principais:

```
                          Taxonomia de Continual Learning
                                         |
     +-------------------+---------------+-------------------+-------------------+
     |                   |                                   |                   |
1. Abordagens de    2. Abordagens de                    3. Abordagens de    4. Abordagens para
   Repetição           Regularização                       Isolamento e        Fluxos Contínuos
   (Replay-based)      (Regularization)                    Expansão            (Data Streams &
     |                   |                                 Arquitetural        Edge AI)
     |                   +--> Baseadas em Parâmetros         |                   |
     |                   |    (EWC, SI, MAS)                 +--> Redes Modulares+--> Árvores Adaptativas
     +--> Experience     |                                   |    e Subredes          (ARF, Hoeffding)
     |    Replay (ER)    +--> Baseadas em Dados /            |    (PackNet)      +--> Deteção de Drift
     +--> Prioritized         Destilação (LwF)               +--> Prompt-Based        (ADWIN, DDM)
     |    Replay (PER)                                            (L2P, DualP)   +--> Clustering com
     +--> Selective                                                                   Decaimento
     |    (Distribution Matching / Coverage)                                          (DenStream)
     +--> Coresets & Compressão de Buffers
```

#### Família 1: Abordagens Baseadas em Repetição / Memória (*Replay-Based Methods*)
- **Hipótese Central:** Para prevenir o esquecimento, transições ou representações de tarefas anteriores são armazenadas ou sintetizadas e re-injetadas durante o treino em novas tarefas, aproximando o gradiente local do gradiente global estacionário.
- **Subtipos e Modelos Relevantes:**
  - **Experience Replay (ER) Clássico:** Mantém um reservatório de memória de tamanho fixo utilizando uma política de substituição temporal (ex.: FIFO - *First-In-First-Out* ou *Reservoir Sampling*).
  - **Prioritized Experience Replay (PER):** Rompe com a amostragem uniforme, atribuindo probabilidades de amostragem proporcionais à relevância estatística da experiência, mensurada pela magnitude do erro de diferença temporal (*Temporal-Difference error* - TD error $|\delta|$). Recorre a árvores de soma binárias (*Sum-Trees*) para garantir complexidade de amostragem $\mathcal{O}(\log |\mathcal{M}|)$ e pesos de amostragem por importância (*Importance Sampling*) para corrigir o enviesamento de gradiente.
  - **Selective Experience Replay (SER):** Filtra ativamente que amostras devem residir na memória de longo prazo segundo critérios funcionais:
    * *Distribution Matching:* Preserva uma amostra representativa da distribuição global de dados;
    * *Coverage Maximization:* Maximiza a cobertura espacial das características observadas, garantindo que tarefas sub-representadas não sejam eliminadas.
  - **Compressão de Buffers com Coresets:** Construção de subconjuntos geométricos reduzidos (*coresets*) que preservam propriedades matemáticas chave (como a distribuição de recompensas ou a matriz de covariância latente) através de algoritmos de partição como $k$-means++. Permite comprimir o repositório de experiências entre 10x e 40x sem perda estatística de precisão, tornando viável a alocação em nós com restrições extremas de memória RAM.
  - **Generative Replay (Pseudo-Rehearsal):** Treina redes generativas (VAEs ou GANs) nas distribuições passadas para sintetizar dados artificiais, eliminando o armazenamento físico de dados brutos e mitigando restrições de privacidade.

#### Família 2: Abordagens Baseadas em Regularização (*Regularization-Based Methods*)
- **Hipótese Central:** Não é necessário armazenar dados do passado se a função de perda penalizar modificações nos parâmetros ou representações que eram críticos para as tarefas anteriores.
- **Subtipos e Modelos Relevantes:**
  - **Regularização de Parâmetros (*Prior-focused*):** Calcula a importância relativa de cada peso da rede $\theta_i$ após o término de uma tarefa e penaliza deslocamentos nesses pesos durante o treino subsequente:
    $$\mathcal{L}(\theta) = \mathcal{L}_{\text{nova}}(\theta) + \sum_i \frac{\lambda}{2} \Omega_i (\theta_i - \theta_{i, \text{antigo}}^*)^2$$
    onde $\Omega_i$ é a matriz de informação de Fisher empírica no algoritmo *Elastic Weight Consolidation* (EWC) ou a trajetória acumulada de gradientes em *Synaptic Intelligence* (SI) e *Memory Aware Synapses* (MAS).
  - **Regularização Funcional / Destilação de Conhecimento (*Data-focused*):** Utiliza *Knowledge Distillation* (KD) em que a rede antes de aprender a nova tarefa atua como modelo "professor", forçando a rede atualizada ("aluno") a manter probabilidades ou ativações latentes similares nas saídas antigas, conforme formalizado em *Learning without Forgetting* (LwF).

#### Família 3: Abordagens de Isolamento de Parâmetros e Arquitetura Dinâmica (*Architecture-Based*)
- **Hipótese Central:** Aloca conjuntos dedicados de parâmetros para cada tarefa, impedindo que os gradientes de uma tarefa interfiram diretamente nos circuitos neuronais consolidados de outra.
- **Modelos Relevantes:**
  - **Modelos Modulares e Subredes (ex.: PackNet, HAT):** Aplicam podas (*pruning*) e máscaras binárias sobre os pesos, congelando as conexões vitais para tarefas antigas e alocando os pesos sobrantes para novas tarefas.
  - **Redes Progressivas (*Progressive Neural Networks*):** Expandem a capacidade da rede alocando novas colunas neuronais para cada nova tarefa, preservando ligações laterais fixas para transferir representações já aprendidas sem risco de esquecimento.
  - **Continual Learning Baseado em Prompts (Prompt-Tuning / PEFT):** Emprega modelos fundacionais congelados e aprende pequenas matrizes de parâmetros adaptativos (*prompts* visuais ou textuais organizados num *Prompt Pool*, como no L2P e DualPrompt), selecionados dinamicamente em função da entrada.

#### Família 4: Abordagens para Fluxos Contínuos, Mineração de Streams e Edge AI
- **Hipótese Central:** Projetadas para ambientes onde os dados fluem sem cessar, exigindo atualização rápida em tempo real, deteção ativa de desvios e baixo consumo computacional.
- **Modelos e Técnicas Relevantes:**
  - **Ensembles Incrementais Adaptativos (ex.: Adaptive Random Forest - ARF):** Conjunto de árvores de decisão incrementais (baseadas em *Hoeffding Trees*) combinadas com detetores de *drift* dedicados por árvore, permitindo a substituição autónoma de ramos obsoletos em tempo de execução.
  - **Detetores Ativos de Desvio de Conceito (ADWIN, DDM, EDDM):** Monitorizam estatisticamente o erro do classificador através de janelas deslizantes com limites de corte adaptativos, disparando alertas de adaptação e purga de dados obsoletos.
  - **Clustering Incremental com Decaimento Temporal (ex.: DenStream):** Agrupamento espacial baseado em densidade que aplica uma função de decaimento temporal com fator de esquecimento $\lambda$ ($f(t) = 2^{-\lambda t}$), enfraquecendo padrões temporais inativos e fortalecendo novos aglomerados em formação.
  - **Sustainable Continual Intelligence (Edge AI & TinyML):** Enquadramento que funde a adaptação contínua com a sustentabilidade computacional, energética e de rede, otimizando orçamentos de FLOPs e transmissão sem fios em nós distribuídos e ambientes industriais.

---

### 1.4 Dicionário de Conceitos-Chave (Pedagógico e Rigoroso)

*   **Esquecimento Catastrófico (*Catastrophic Forgetting / Interference*):**  
    A deterioração rápida e substancial da capacidade de uma rede neuronal generalizar ou inferir corretamente sobre dados ou tarefas do passado logo após ter sido otimizada para novas tarefas ou distribuições.
*   **Dilema Estabilidade-Plasticidade (*Stability-Plasticity Dilemma*):**  
    O compromisso fundamental entre a facilidade de assimilar nova informação (*plasticidade*) e a robustez de manter as estruturas e memórias existentes protegidas contra sobrescrita (*estabilidade*).
*   **Buffer de Replay de Experiência (*Experience Replay Buffer* - ERB):**  
    Estrutura de dados em memória que preserva amostras anteriores (tuplos de características-rótulo ou transições de estado-ação-recompensa) para reamostragem periódica durante as fases de otimização estocástica.
*   **Amostragem Ponderada e Correção por Importância (*Importance Sampling Weights*):**  
    Fator de correção $w_i = (N \cdot P(i))^{-\beta}$ aplicado durante o cálculo do gradiente em amostragem prioritária não-uniforme, garantindo que a direção esperada do gradiente não seja viciada artificialmente pelas probabilidades assimétricas de seleção.
*   **Coreset (Subconjunto Núcleo):**  
    Subconjunto ponderado e geometricamente representativo de um conjunto maior de dados que preserva, com garantias matemáticas de aproximação, a função de custo ou as propriedades estruturais da distribuição original de pontos.
*   **Desvio de Conceito (*Concept Drift*):**  
    Alteração temporal da distribuição estocástica dos dados no ambiente operacional. Se modifica a relação entre atributos e classes ($P(Y|X)$), diz-se *Real Drift*; se modifica apenas a frequência ou amplitude das variáveis de entrada ($P(X)$), diz-se *Virtual Drift*.
*   **Avaliação Prequencial (*Prequential Evaluation / Test-Then-Train*):**  
    Protocolo experimental obrigatório em fluxos de dados contínuos em que cada nova amostra observada é primeiramente utilizada para testar o modelo (registando a perda e a predição) e, apenas em seguida, disponibilizada para atualizar os parâmetros do modelo.
*   **Transferência para a Frente (*Forward Transfer* - FWT):**  
    Capacidade de conhecimentos adquiridos em tarefas passadas acelerarem ou melhorarem o desempenho e a velocidade de convergência na aprendizagem de uma nova tarefa futura.
*   **Transferência para Trás (*Backward Transfer* - BWT):**  
    Métrica formal que avalia o impacto que a aprendizagem de uma nova tarefa tem no desempenho das tarefas passadas:
    $$\text{BWT} = \frac{1}{K-1} \sum_{i=1}^{K-1} (R_{K, i} - R_{i, i})$$
    onde $R_{K, i}$ representa a exatidão na tarefa $i$ após treinar a tarefa final $K$. Um BWT positivo reflete melhoria retrospectiva (*retroactive facilitation*); um BWT negativo quantifica a severidade do esquecimento catastrófico.
*   **Destilação de Conhecimento (*Knowledge Distillation* - KD):**  
    Técnica de transferência de representações na qual uma função de divergência (ex.: Kullback-Leibler) penaliza desvios entre as saídas suavizadas por temperatura (*soft targets*) de um modelo de referência consolidado e o modelo corrente.
*   **Esquecimento Catastrófico Espácio-Temporal (*Spatial-Temporal Catastrophic Forgetting*):**  
    Degradação do conhecimento que se manifesta em sistemas distribuídos e federados, ocorrendo simultaneamente ao longo do tempo (sequência histórica de tarefas) e através do espaço físico (heterogeneidade estatística e divergência entre dispositivos clientes descentralizados).
*   **Janela Adaptativa com Garantias Estatísticas (ADWIN - *Adaptive Windowing*):**  
    Algoritmo de controlo de fluxo que preserva estatísticas numa janela deslizante de tamanho variável. Quando a diferença das médias entre subjanelas excede um limiar rigoroso baseado na desigualdade de Hoeffding, o método corta automaticamente a porção antiga da janela, acusando a presença de *drift*.
*   **Fator de Decaimento Temporal ($\lambda$ / *Temporal Decay*):**  
    Parâmetro escalar utilizado em modelos contínuos para reduzir exponencialmente a influência ou o peso de instâncias de dados antigas à medida que o tempo decorre, libertando memória e priorizando a dinâmica mais recente.

---

## 2. Protocolo de Pesquisa Sistemática (Search Strategy)

A recolha bibliográfica em *Continual Learning* deve obedecer a uma metodologia estruturada que capture tanto as bases matemáticas seminais como os avanços computacionais recentes orientados a eficiência e fluxos de dados.

### 2.1 Venues Científicos Prioritários

Os investigadores e agentes autónomos devem restringir a pesquisa a veículos científicos de prestígio reconhecido com arbitragem científica por pares:

| Categoria | Sigla / Nome do Venue | Qualificação / Foco de Investigação |
|:---|:---|:---|
| **Conferências de IA & ML (Core A\*)** | **NeurIPS** (Neural Information Processing Systems) | Fundamentos de otimização, NTK, teoria de regularização e replay |
| | **ICML** (International Conference on Machine Learning) | Algoritmos de aprendizagem contínua, análise de limites de erro e BWT |
| | **ICLR** (International Conference on Learning Representations) | Representações invariantes, PER, prompt-tuning e dinâmica sináptica |
| | **CVPR / ICCV / ECCV** | Class-Incremental Learning, segmentação contínua e visão em aberto |
| | **AAAI** (Conference on Artificial Intelligence) | Selective replay, planeamento em lifelong learning, modelos híbridos |
| | **CoLLAs** (Conf. on Lifelong Learning Agents) | Venue de topo 100% especializado em agentes contínuos e métricas |
| **Revistas Científicas de Referência** | **IEEE T-PAMI** | Trabalhos extensivos de consolidação teórica e representações visuais |
| | **JMLR** (Journal of Machine Learning Research) | Demonstrações matemáticas rigorosas e estruturas teóricas profundas |
| | **IEEE T-NNLS** | Métodos neurais dinâmicos, estabilidade e controlo adaptativo |
| | **IEEE TNSE** | Redes complexas, inteligência contínua sustentável e edge systems |
| | **Neural Networks** (Elsevier) | Fundamentos biológicos do dilema estabilidade-plasticidade |
| **Venues de Sistemas, Borda e Fluxos** | **ACM SenSys / MobiCom / IoTDI** | TinyML contínuo, restrições extremas de energia e nós de borda |
| | **ACM KDD / IEEE TKDE** | Mineração de data streams, deteção de drift e conjuntos adaptativos |
| | **MIDL / IEEE T-MI** | Imagem médica contínua e compressão de buffers de experiência |
| **Repositórios de Preprints Verificados** | **arXiv** (`cs.LG`, `cs.AI`, `cs.CV`, `stat.ML`) | Preprints dos últimos 12 meses com potencial de replicação técnica |

---

### 2.2 Equações de Pesquisa Booleanas Recomendadas

As strings de pesquisa infra estão calibradas para execução direta em bases bibliográficas científicas (Scopus, Web of Science, IEEE Xplore, Google Scholar e ACM Digital Library):

#### Bloco A: Fundamentos Teóricos, Esquecimento Catastrófico e Estabilidade-Plasticidade
```text
("continual learning" OR "lifelong learning" OR "incremental learning") AND ("catastrophic forgetting" OR "catastrophic interference" OR "stability-plasticity dilemma") AND ("class-incremental" OR "backward transfer" OR "elastic weight consolidation")
```

#### Bloco B: Mecanismos de Replay, Compressão de Buffers e Memória Episódica
```text
("continual learning" OR "lifelong learning" OR "reinforcement learning") AND ("experience replay" OR "selective experience replay" OR "prioritized experience replay") AND ("coreset" OR "distribution matching" OR "buffer compression" OR "memory efficiency")
```

#### Bloco C: Fluxos Contínuos (*Data Streams*), Desvio de Conceito e Avaliação Prequencial
```text
("data stream" OR "streaming data" OR "evolving data") AND ("concept drift" OR "adaptive random forest" OR "ADWIN" OR "prequential") AND ("online machine learning" OR "drift detection" OR "DenStream")
```

#### Bloco D: Continual Learning em Borda, Inteligência Sustentável e TinyML
```text
("continual learning" OR "lifelong learning" OR "continual intelligence") AND ("edge AI" OR "edge devices" OR "TinyML" OR "energy efficiency") AND ("sustainable" OR "resource-constrained" OR "federated continual learning")
```

---

### 2.3 Janela Temporal de Análise

A estratificação cronológica divide-se estrategicamente em dois períodos:

1. **Janela Seminal e Fundacional (1989 – 2017):**
   - **Objetivo:** Dominar a origem epistemológica do problema e os pilares matemáticos das três famílias clássicas.
   - **Marcos Históricos Obrigatórios:**
     - McCloskey & Cohen (1989) e Ratcliff (1990) — Descoberta e caracterização formal da interferência catastrófica;
     - Robins (1995) — Formulação do mecanismo de *pseudo-rehearsal*;
     - Schaul et al. (ICLR 2015) — Introdução do *Prioritized Experience Replay* (PER);
     - Kirkpatrick et al. (PNAS 2017) — Otimização bio-inspirada via *Elastic Weight Consolidation* (EWC);
     - Li & Hoiem (IEEE T-PAMI 2017) — Aprendizagem sem esquecimento via destilação (*Learning without Forgetting* - LwF);
     - Lopez-Paz & Ranzato (NeurIPS 2017) — Formalização do *Gradient Episodic Memory* (GEM) e das métricas BWT e FWT.

2. **Janela de Avanços Recentes e Estado da Arte (Últimos 3 a 5 anos):**
   - **Objetivo:** Rastrear a evolução contemporânea focada em sustentabilidade energética, compressão de memória, mitigação de drift em fluxos e adaptação eficiente de modelos de grande escala (*Foundation Models*).
   - **Marcos Recentes Relevantes:**
     - Isele & Cosgun (AAAI 2018) — *Selective Experience Replay* e emparelhamento de distribuição;
     - Haug et al. (2022) — Padronização de métricas e enquadramento de avaliação em fluxos (*float* framework);
     - Wang et al. (CVPR 2022/2023) — *Learning to Prompt* (L2P) e *DualPrompt* para *Continual Learning* sem armazenamento de imagens;
     - Zheng et al. (MIDL 2024) — Compressão assíncrona de buffers ERB via *coresets* e amostragem ponderada $k$-means++;
     - Wu et al. (IEEE TNSE 2026) — Formalização de *Sustainable Continual Intelligence* para *Edge AI* e co-design de redes e comunicação;
     - Reyes et al. e Arkana et al. (2026) — Ensembles adaptativos (ARF), *clustering* com decaimento temporal e resiliência a ruído em fluxos reais.

---

## 3. Critérios de Elegibilidade (Inclusão e Exclusão)

A curadoria da literatura deve aplicar um filtro estrito para separar estudos metodologicamente rigorosos de trabalhos simplistas ou heurísticos desprovidos de garantias empíricas e teóricas.

```
                   Candidato a Artigo Científico de CL
                                     |
                +--------------------+--------------------+
                |                                         |
                v                                         v
       Critérios de Inclusão (+)                 Critérios de Exclusão (-)
    - Isolamento de tarefas/cenário          - Comparação isolada com Fine-Tuning
    - Métricas de retenção (BWT/Acc)         - Omissão de dados de tarefas antigas
    - Baselines canónicos presentes          - Oráculo de tarefa não declarado (Class-IL)
    - Custo computacional e memória          - Treino puramente estático batch
                |                                         |
                v                                         v
        APROVADO PARA TABELA                     REJEITADO / EXCLUÍDO
```

### 3.1 Critérios de Inclusão (+)
O documento avaliado deve preencher cumulativamente os seguintes requisitos:
1. **Especificação Clara do Cenário:** Define explicitamente se o regime avaliado é Task-IL, Domain-IL, Class-IL ou Data Stream Online, clarificando se o modelo tem ou não acesso ao rótulo da tarefa durante a inferência.
2. **Avaliação Quantitativa de Retenção e Esquecimento:** Apresenta relatórios formais sobre o desempenho em tarefas anteriores após o treino das tarefas subsequentes, recorrendo a métricas padronizadas (ex.: Exatidão Média Final $A_K$, *Backward Transfer* BWT, Medida de Esquecimento $F$, ou Exatidão Prequencial ao longo do tempo).
3. **Confronto com Baselines Estabelecidos:** Compara obrigatoriamente a solução contra, no mínimo, dois dos seguintes pontos de calibração clássicos:
   - *Fine-tuning sequencial ingénuo* (limite inferior de desempenho / sem defesas contra o esquecimento);
   - *Joint Training / Multitask Learning* (limite superior de desempenho com acesso simultâneo a todos os dados);
   - Pelo menos um método representativo de regularização (EWC, SI, LwF) ou de replay (ER com *reservoir sampling*, PER).
4. **Transparência de Hiperparâmetros e Recursos:** Detalha rigorosamente a capacidade da memória atribuída ($\mathcal{M}$ em número de amostras ou megabytes), a arquitetura neuronal exata utilizada (ex.: ResNet-18, MLP, Hoeffding Tree) e o número de épocas/passagens por tarefa.
5. **Reprodutibilidade Experimental:** Disponibiliza código-fonte aberto ou providencia a descrição algorítmica passo a passo suficiente para replicação independente.

### 3.2 Critérios de Exclusão (-)
Devem ser rejeitados trabalhos que apresentem qualquer uma das seguintes deficiências:
1. **Ocultação de Desempenho Retrospectivo:** Estudos que reportam apenas o desempenho imediato na última tarefa aprendida, omitindo se o conhecimento das tarefas precedentes colapsou.
2. **Trapaça do Oráculo de Tarefa (*Task-Oracle Cheating*):** Artigos que afirmam resolver o problema difícil de Class-Incremental Learning, mas utilizam discretamente o identificador da tarefa no momento do teste para isolar a cabeça de saída.
3. **Ausência de Restrições de Armazenamento:** Propostas que alegam resolver a aprendizagem contínua armazenando silenciosamente um volume desmedido ou crescente de dados brutos que cresce linearmente sem limite com o número de tarefas ($O(K)$ não controlado).
4. **Validação em Ambientes Triviais Exclusivos:** Testes limitados exclusivamente a versões sintéticas com separabilidade óbvia sem ruído, negligenciando os conjuntos de teste padronizados da área (Split-MNIST, Permuted-MNIST, Split-CIFAR-100, ImageNet-Subset, ou *streams* como Electricity e Covertype).
5. **Documentos de Natureza Puramente Opinativa ou Comercial:** White papers comerciais sem validação experimental independente ou artigos publicados em conferências desprovidas de revisão cega por pares.

---

## 4. Roteiro de Extração e Síntese Analítica

Cada artigo que ultrapasse com sucesso a triagem deve ser submetido à seguinte **Checklist de 5 Dimensões**:

```
+-----------------------------------------------------------------------------------+
|               CHECKLIST ANALÍTICA DE EXTRAÇÃO DE ARTIGOS DE CL                    |
+-----------------------------------------------------------------------------------+
| 1. Problema e Motivação                                                           |
|    - Qual o modo de esquecimento ou restrição ambiental que o artigo combate?    |
|    - Porque falham as abordagens clássicas de regularização ou replay no caso?    |
+-----------------------------------------------------------------------------------+
| 2. Inovação Metodológica / Arquitetural                                           |
|    - Qual o algoritmo exato introduzido (ex.: coreset, prompt pool, ARF)?         |
|    - Como é gerida a alocação de memória e o cálculo da perda de retenção?       |
+-----------------------------------------------------------------------------------+
| 3. Datasets e Protocolo de Avaliação                                              |
|    - Que benchmarks e divisões foram usados (ex.: Split-CIFAR, fluxos com drift)? |
|    - O protocolo foi prequencial, incremental por tarefas ou por classes?         |
+-----------------------------------------------------------------------------------+
| 4. Resultados Empíricos e Trade-offs (Retenção vs. Recursos)                      |
|    - Quais os valores numéricos de exatidão média final, BWT ou compressão?      |
|    - Qual a sobrecarga de tempo de processamento, consumo de RAM e energia?      |
+-----------------------------------------------------------------------------------+
| 5. Limitações e Desafios em Aberto (Open Challenges)                             |
|    - Que restrições foram identificadas (ex.: escalabilidade para milhares de     |
|      tarefas, sensibilidade a ruído, dependência de pré-treino maciço)?         |
+-----------------------------------------------------------------------------------+
```

### Detalhe Operacional de Cada Ponto da Checklist:

1. **Problema e Motivação:**
   - Registar o cenário exato e a dor prática. Exemplo: custos insustentáveis de transmissão e armazenamento do buffer em imagens médicas móveis (Zheng et al., 2024) ou o desperdício de comunicação em Edge AI que ignora a co-otimização de rede e IA (Wu et al., 2026).
2. **Inovação Metodológica/Arquitetural:**
   - Documentar os termos matemáticos centrais: critério de ponderação de amostras, funções de decaimento temporal, algoritmo de seleção geométrica (ex.: $k$-means++ em coresets), ou lógica de árvores incrementais acopladas a ADWIN.
3. **Datasets e Protocolo de Avaliação:**
   - Mapear a sequência de tarefas aplicada (número de tarefas $K$, número de classes por etapa, presença de desvios abruptos vs graduais) e o modo de teste (avaliação prequencial *test-then-train* ou teste matricial cruzado $R_{i,j}$).
4. **Resultados Empíricos e Trade-offs:**
   - Extrair métricas objetivas numéricas exatas: percentagem de exatidão prequencial mantida, razão de compressão do buffer (ex.: 10x sem quebra de performance), percentagem de redução em passos de treino (ex.: aceleração 2x via PER) e consumo em RAM-Hours.
5. **Limitações e Desafios em Aberto:**
   - Registar com honestidade intelectual o que não foi resolvido: necessidade de parâmetros estáticos iniciais, custo computacional adicional da amostragem por importância, ou degradação sob ruído semântico extremo.

---

## 5. Padrão de Formatação para a Tabela de Literatura (`README.md`)

O ficheiro central `README.md` localizado no mesmo diretório organiza o estado da arte numa tabela padronizada de 6 colunas. Toda e qualquer adição deve respeitar integralmente a seguinte especificação.

### 5.1 Especificação Rigorosa das 6 Colunas Obrigatórias

| Coluna | Título | Tipo de Conteúdo e Regras de Formatação |
|:---:|:---|:---|
| **1** | **Nome** | Título oficial e integral do artigo em língua inglesa, sem modificações ou truncagens. |
| **2** | **Detalhes** | Metadados catalográficos estruturados verticalmente utilizando quebras de linha duplas em HTML (`<br><br>`), contendo obrigatoriamente: **Autores**, **Ano**, **Pub** (abreviatura do periódico ou conferência de topo). |
| **3** | **Abstract** | Excertos textuais fiéis e literais transcritos do resumo do artigo original, delimitados individualmente entre aspas duplas (`"..."`) e encadeados por `<br><br>`. Devem capturar a dor do trabalho, a hipótese e o achado central. |
| **4** | **Conclusion** | Excertos textuais fiéis e literais retirados da secção final de conclusões do artigo original, delimitados entre aspas duplas (`"..."`) e encadeados por `<br><br>`. |
| **5** | **Resumo (NotebookLM)** | Síntese analítica densa e aprofundada em língua portuguesa, estruturada de forma estrita em **quatro parágrafos encadeados** separados por `<br><br>` (ver secção 5.2). |
| **6** | **Citação** | Referência bibliográfica completa formatada de acordo com a **Norma Vancouver (NLM)**, finalizada com a hiperligação funcional do DOI (`https://doi.org/...`). |

---

### 5.2 Estrutura Padronizada do "Resumo (NotebookLM)" (4 Parágrafos)

A coluna 5 destina-se a fornecer uma compreensão profunda, conceptual e prática aos investigadores e agentes. Deve obedecer estritamente à seguinte sequência de 4 parágrafos:

*   **Parágrafo 1 — Contexto e Problema de Investigação:**  
    Introduz a classe de problema tratada no âmbito de Continual Learning / Lifelong Learning, estabelecendo a falha prática do treino convencional em lote ou dos modelos estáticos perante a dinâmica sequencial do ambiente (ex.: esquecimento catastrófico, restrições de buffer ou concept drift).
*   **Parágrafo 2 — Mecanismo Proposto e Inovação Arquitetural:**  
    Descreve com rigor conceptual o método ou algoritmo concebido pelos autores. Explica a lógica operacional do mecanismo (ex.: estratégia de seleção do replay, compressão por coreset, preservação da distribuição de recompensas, enquadramento modular com decaimento temporal ou amostragem priorizada estocástica).
*   **Parágrafo 3 — Validação Experimental e Métricas Numéricas:**  
    Sumariza os cenários empíricos de teste (bancos de dados clínicos, jogos Atari, sensores urbanos ou redes dinâmicas), os baselines confrontados e as métricas quantitativas alcançadas (ex.: exatidão prequencial obtida, margem percentual face a modelos estáticos, taxa de compressão sem perda de acuidade e redução no número de iterações para convergência).
*   **Parágrafo 4 — Significado Teórico, Limitações e Desafios:**  
    Avalia o impacto do trabalho na disciplina, a sua viabilidade de aplicação em nós de borda ou produção contínua, identificando de forma transparente as limitações assumidas pelos próprios autores e as direções prioritárias de trabalho futuro.

---

### 5.3 Exemplo Canónico de Registo na Tabela

Para servir de modelo absoluto de calibração para novas adições, apresenta-se a transcrição de um registo exemplar aprovado:

```markdown
| Selective experience replay compression using coresets for lifelong deep reinforcement learning in medical imaging | **Autores:** Zheng et al.<br><br>**Ano:** 2024<br><br>**Pub:** MIDL / PMLR | "Selective experience replay aims to recount selected experiences from previous tasks to avoid catastrophic forgetting."<br><br>"However, storing experiences from all previous tasks make lifelong learning using selective experience replay computationally very expensive and impractical as the number of tasks increase."<br><br>"To that end, we propose a reward distribution-preserving coreset compression technique for compressing experience replay buffers stored for selective experience replay."<br><br>"Our results demonstrate that the potential of the coreset-based ERB compression method for compressing experiences without a significant drop in performance." | "In conclusion, we propose a coreset-based ERB compression technique for increased computational efficiency and scalability of selective experience replay based deep lifelong reinforcement learning with excellent performance."<br><br>"In particular, our weighted sampling-based reward distribution-preserving coreset ERB compression showed excellent performance for shrinking the size of saved experiences from previous tasks for lifelong deep reinforcement learning in medical imaging."<br><br>"Our results demonstrated that experience replay buffers can be compressed up to 10x without any significant drop in performance."<br><br>"In contrast, our proposed coreset-based compression techniques asynchronously compress the ERB, without the need for extra information from the training session where the ERB is produced or the model parameters that came from the training session." | Este artigo aborda o problema do esquecimento catastrófico (*catastrophic forgetting*) em cenários de *Deep Reinforcement Learning* (DRL) aplicados à imagem médica, onde os modelos precisam de se adaptar continuamente a novos cenários e sequências de aquisição sem perder a capacidade de atuar nos ambientes anteriores.<br><br>Para contornar esta limitação, o estudo foca-se no *Selective Experience Replay* (SER), uma estratégia independente do modelo (*model-agnostic*) que armazena e reproduz experiências passadas de tarefas anteriores. Contudo, à medida que o número de tarefas aumenta, o armazenamento e a comunicação do repositório de experiências (*Experience Replay Buffer* - ERB) tornam-se computacionalmente inviáveis.<br><br>Para resolver esta restrição, os autores propõem uma nova técnica de compressão do ERB baseada em *coresets* com amostragem ponderada via $k$-means++, projetada para preservar a distribuição de recompensas sem necessitar dos parâmetros do modelo ou da função Q durante o treino.<br><br>Avaliado em dados de ressonância magnética cerebral (BRATS) para a localização do ventrículo e em dados de RM de corpo inteiro para a localização de múltiplos pontos anatómicos de referência (*landmarks*), o método provou conseguir comprimir o repositório de experiências até 10 vezes sem demonstrar uma degradação estatisticamente significativa no desempenho da localização. | Zheng G, Zhou S, Braverman V, Jacobs MA, Parekh VS. Selective experience replay compression using coresets for lifelong deep reinforcement learning in medical imaging. In: Medical Imaging with Deep Learning. PMLR; 2024. p. 1751–64. |
```

---

## 6. Guardrails de Qualidade & Anti-Alucinação

Para resguardar a integridade, fiabilidade e credibilidade científica da base de conhecimento, os agentes e investigadores devem cumprir impreterivelmente as seguintes diretrizes de controlo:

### 6.1 Validação Estrita de DOIs e Metadados
- **Verificação Ativa de Links:** O link DOI adicionado deve ser real, auditado e direcionar efetivamente para o registo do artigo no servidor da editora ou no repositório de preprints (arXiv). É proibido inventar sufixos numéricos ou combinar autores reais com publicações inexistentes.
- **Conferência em Bases Centrais:** Todos os metadados bibliográficos (ano de publicação, volume, páginas e nome exato do periódico/conferência) devem ser validados no DBLP, CrossRef, Google Scholar ou PubMed.

### 6.2 Proibição Absoluta de Interpolação ou Arredondamento Numérico
- Todos os números reportados na tabela (ex.: ganhos percentuais, taxas de compressão, valores de F1-score, níveis de exatidão prequencial) devem ser **transcritos com precisão literal** a partir das tabelas ou texto do documento original.
- É categoricamente vedado aproximar métricas (ex.: se o artigo reporta 86,4% de exatidão prequencial, o texto deve citar estritamente "86,4%", sendo expressamente proibido escrever "cerca de 86%" ou "aproximadamente 87%").

### 6.3 Fidelidade Textual em Citações Diretas
- Os campos **Abstract** e **Conclusion** destinam-se exclusivamente a **citações literais entre aspas**, extraídas do texto fonte em inglês. Não é permitida a tradução livre, a inserção de resumos subjetivos ou o truncamento malicioso que altere o sentido intencionado pelos autores primários.

### 6.4 Neutralidade Científica e Sobriedade Epistemológica
- O vocabulário utilizado nas sínteses analíticas em português deve manter um tom estritamente académico, factual, pedagógico e neutro.
- Devem ser expurgados quaisquer adjetivos promocionais ou sensacionalistas (tais como "revolucionário", "milagroso", "perfeito", "inovação sem precedentes"), priorizando sempre a descrição técnica clara dos mecanismos, das restrições e dos compromissos de engenharia subjacentes.
