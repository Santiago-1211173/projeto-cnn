# Protocolo de Curadoria e Navegação na Literatura: Early-Exit em Redes Neuronais

> **Ficheiro Central Associado:** `README.md` (no mesmo diretório)  
> **Natureza do Documento:** Guia metodológico de recolha, análise crítica e referenciação científica para agentes autónomos e investigadores.  
> **Domínio Científico:** Inteligência Artificial / Aprendizagem Automática / Redes Neuronais Profundas e Computação Adaptativa Dinâmica (*Dynamic & Multi-Exit Neural Networks*).

---

## 1. Enquadramento Teórico & Âmbito do Tópico

### 1.1 Definição Canónica na Literatura Internacional de IA
O paradigma de **Early-Exit** (também denominado na literatura por *Multi-Exit Neural Networks*, *Early-Exit Deep Neural Networks* - EE-DNNs, ou *Dynamic Depth Neural Networks*) é uma formulação de computação adaptativa em que uma rede neuronal profunda é dotada de ramificações de saída intermediárias (*exit blocks* ou *side branches*) acopladas a diferentes profundidades da arquitetura dorsal (*backbone*).

Formalmente, considere-se uma rede neuronal profunda convencional composta por uma sequência de $L$ blocos de transformação parametrizados:
$$\mathcal{F}(x) = (f_L \circ f_{L-1} \circ \dots \circ f_1)(x)$$
onde $x \in \mathcal{X}$ representa o vetor ou tensor de entrada e $h_l = f_l(h_{l-1})$ denota a ativação intermediária na camada $l \in \{1, \dots, L\}$, com $h_0 = x$.

Numa arquitetura de *Early-Exit*, seleciona-se um subconjunto ordenado de $K$ camadas intermediárias de saída $\mathcal{E} = \{l_1, l_2, \dots, l_K\}$, com $l_1 < l_2 < \dots < l_K = L$. Em cada camada $l_k \in \mathcal{E}$, conecta-se um classificador ou regressor lateral leve $g_k: \mathcal{H}_{l_k} \to \mathcal{Y}$, gerando uma predição intermediária:
$$\hat{y}_k = g_k(h_{l_k})$$

Adicionalmente, cada saída intermediária $k$ é munida de uma função de decisão ou política de paragem $\pi_k(h_{l_k}, \hat{y}_k) \in \{0, 1\}$. Durante a inferência:
- Se $\pi_k(h_{l_k}, \hat{y}_k) = 1$ (ou se for atingida a última saída $k=K$), a propagação para a frente (*forward pass*) é imediatamente interrompida e o modelo retorna $\hat{y}_k$ como resultado final;
- Se $\pi_k(h_{l_k}, \hat{y}_k) = 0$, a propagação continua através das camadas subsequentes $f_{l_k+1}, \dots, f_{l_{k+1}}$ até ao próximo bloco de saída.

```
       +-----------------------------------------------------------------------------------------+
       |                                Fluxo da Rede Dorsal (Backbone)                          |
       +-----------------------------------------------------------------------------------------+
                                                                                    
   Input x ---> [ Bloco 1 ] ---> [ Bloco 2 ] -------------> [ Bloco 3 ] --------> [ Bloco L ]
                     |                 |                          |                    |
                     v                 v                          v                    v
              +-------------+   +-------------+            +-------------+      +-------------+
              | Exit 1 (g1) |   | Exit 2 (g2) |            | Exit 3 (g3) |      | Exit Final  |
              +-------------+   +-------------+            +-------------+      +-------------+
                     |                 |                          |                    |
                     v                 v                          v                    v
              pi1 >= Limiar?    pi2 >= Limiar?             pi3 >= Limiar?          Predição
               /         \       /         \                /         \              Final
             SIM         NÃO   SIM         NÃO            SIM         NÃO              y_L
              |           |     |           |              |           |
              v           +---->+           +------------->+           +-------------->
          Retorna y1           Retorna y2                 Retorna y3
          (Pára aqui)          (Pára aqui)                (Pára aqui)
```

O custo computacional acumulado até à saída $k$, medido em operações de ponto flutuante (FLOPs) ou tempo de latência, satisfaz:
$$\mathcal{C}_k = \sum_{j=1}^{l_k} \text{FLOPs}(f_j) + \text{FLOPs}(g_k) \ll \mathcal{C}_K$$
permitindo modular a computação executada de acordo com as necessidades estocásticas de cada amostra individual.

---

### 1.2 Problema Fundamental que Aborda
O paradigma de *Early-Exit* resolve quatro limitações estruturais severas dos modelos neuronais profundos estáticos contemporâneos:

1. **Inflexibilidade Computacional e Desperdício Energético (Static Inference Bottleneck):**
   - Nas redes profundas estáticas clássicas (ex.: ResNet-152, Vision Transformers, BERT-Large), cada amostra de entrada é forçada a atravessar 100% dos parâmetros e das camadas da rede, independentemente do seu grau de dificuldade.
   - Na prática visual e semântica, uma fração substantiva dos dados é constituída por instâncias triviais (ex.: objetos nítidos, centrados, fundos homogéneos, frases semanticamente inequívocas) que podem ser resolvidas com precisão quase perfeita nas primeiras camadas. Forçar o processamento integral resulta num consumo desnecessário de energia, latência e ciclos de processador.

2. **Mitigação do Fenómeno de "Overthinking" (Superpensamento):**
   - A literatura recente documenta que o processamento em camadas excessivamente profundas pode degradar predições que já se encontravam corretas nas camadas intermediárias.
   - Este fenómeno ocorre porque as camadas mais profundas de redes sobre-parametrizadas tendem a especializar-se em características contextuais altamente abstratas ou subtis, tornando-se mais sensíveis a artefactos espúrios, ruído de alta frequência e perturbações adversariais em amostras simples. A saída precoce blinda o modelo contra esta degradação.

3. **Atenuação do Desvanecimento de Gradientes via Supervisão Profunda (*Deep Supervision*):**
   - Durante o treino de redes com centenas de camadas, a propagação retrógrada do gradiente a partir de uma única função de perda na camada terminal sofre atenuação exponencial (*vanishing gradients*).
   - A incorporação de blocos de saída intermediários atua como uma rede de supervisão distribuída: cada ramo lateral injeta gradientes diretamente nas camadas inferiores da rede dorsal, acelerando a convergência e atuando como um regularizador estrutural robusto contra o sobreajuste (*overfitting*).

4. **Viabilização de Inteligência Artificial na Borda (*Edge AI*) e Particionamento Distribuído:**
   - Dispositivos periféricos (*smartphones*, sensores industriais, drones, sistemas embarcados em veículos) operam sob rigorosas restrições térmicas, de bateria e de latência temporal estrita.
   - O *Early-Exit* fornece uma base natural para o particionamento colaborativo (*Device-Edge-Cloud Hierarchical Offloading*): as primeiras saídas executam no dispositivo local; se a confiança for insuficiente, a representação latente é transmitida a um servidor de borda (*Edge*) ou à nuvem (*Cloud*), otimizando a largura de banda de rede e garantindo respostas em tempo real.

---

### 1.3 Taxonomia e Principais Famílias de Abordagens no Estado da Arte

As metodologias de *Early-Exit* organizam-se em quatro dimensões teóricas e metodológicas fundamentais:

```
                                  Taxonomia de Early-Exit em DNNs
                                                 |
         +--------------------+------------------+------------------+--------------------+
         |                    |                                     |                    |
  1. Topologia e       2. Estratégias de                     3. Mecanismos de     4. Domínios e
     Design de Ramos      Treino e Otimização                   Decisão e Política   Modalidades
         |                    |                                     |                    |
   - BranchyNet         - Treino Conjunto (Joint)             - Limiar de          - Visão (CNNs/ViTs)
   - MSDNet             - Treino em Duas Fases (Freeze)         Entropia/Softmax   - NLP (DeeBERT/PABEE)
   - Shallow-Deep       - Destilação de Conhecimento          - Confidence Gate    - DRL (EEDQN)
   - EENets               (Inter-Exit Distillation)           - Agentes DRL/Bandit - Séries Temporais
```

#### Família 1: Topologia e Design dos Ramos de Saída (*Branch Architecture*)
- **Ramos Leves Convencionais (BranchyNet, Shallow-Deep):** Acoplam uma camada de *pooling* médio global seguida de uma projeção linear (*fully connected*) e ativação softmax. Minimizam a sobrecarga de parâmetros adicionais (*parameter overhead*).
- **Arquiteturas Co-desenhadas para Multi-Exit (MSDNet - Multi-Scale Dense Networks):** Argumentam que conectar saídas precoces a redes clássicas (como ResNet) é subótimo porque as primeiras camadas não têm características de alta resolução semântica. A MSDNet mantém mapas de características multi-escala em todas as profundidades através de conexões densas, permitindo que as saídas precoces operem sobre abstrações ricas.
- **Ramos com Estimadores de Incerteza e Custo (EENets):** Arquiteturas que integram nós duplos de saída em cada ramo: uma cabeça de classificação e uma cabeça de custo/confiança que quantifica o risco de interrupção.

#### Família 2: Estratégias de Treino e Otimização
- **Treino Conjunto Ponta-a-Ponta (*Joint End-to-End Training*):**
  - Otimiza uma função de perda multi-objetivo ponderada:
    $$\mathcal{L}_{\text{total}}(\Theta, \{\theta_k\}) = \sum_{k=1}^K w_k \mathcal{L}_k(y, g_k(h_{l_k}; \theta_k))$$
  - *Desafio do Gradiente Conflitante:* Os gradientes gerados por saídas precoces (que requerem características gerais e invariantes) podem entrar em conflito com os gradientes da saída final (que exigem características altamente especializadas), podendo prejudicar a acuidade da camada final.
- **Treino Desacoplado em Duas Etapas (*Two-Stage / Freeze Backbone*):**
  - Fase 1: A rede dorsal é treinada convencionalmente até à convergência completa e os seus pesos são congelados.
  - Fase 2: Os ramos intermediários são treinados independentemente. Evita interferências na rede base, mas perde o benefício da supervisão profunda e da regularização conjunta.
- **Destilação de Conhecimento Inter-Saídas (*Self-Distillation / Inter-Exit KD*):**
  - A camada final (ou saídas mais profundas), dotada de maior capacidade representacional, atua como "professor" (*teacher*).
  - As saídas precoces ("alunos" / *students*) minimizam uma perda de divergência de Kullback-Leibler em relação às distribuições de probabilidade suavizadas da saída final:
    $$\mathcal{L}_{\text{KD}} = \text{KL}\left(\sigma\left(\frac{z_K}{T}\right) \parallel \sigma\left(\frac{z_k}{T}\right)\right)$$

#### Família 3: Mecanismos e Políticas de Decisão de Saída (*Exit Policies*)
- **Políticas Estáticas Baseadas em Incerteza (Threshold-based):**
  - *Máxima Probabilidade Softmax (MSP):* $\max_c \hat{y}_{k,c} \ge \tau_k$. Simples, mas suscetível a modelos sobre-confiantes mal calibrados.
  - *Entropia de Shannon da Predição:* $\mathcal{H}(\hat{y}_k) = -\sum_{c} \hat{y}_{k,c} \log \hat{y}_{k,c} \le \epsilon_k$. Mede a dispersão da incerteza sobre todas as classes.
  - *Diferença entre os Dois Maiores Logits (Margin):* Avalia a margem de separabilidade da classe vencedora.
- **Políticas Baseadas em Paciência (*Patience-based Exit*):**
  - Utilizadas primariamente em modelos sequenciais e Transformers (ex.: PABEE). A inferência é interrompida se $r$ saídas consecutivas concordarem com a mesma classe predita, evitando saídas erróneas motivadas por picos de confiança espúrios.
- **Políticas Dinâmicas e Adaptativas (DRL, Bandits e Pré-Análise de Input):**
  - Agentes de decisão inteligentes (ex.: redes Deep Q-Networks ou Contextual Bandits) que monitorizam dinamicamente a bateria do sistema, a latência de comunicação e métricas de complexidade da entrada (ex.: contagem e complexidade de contornos da imagem - *Contour Complexity*) para fixar o ponto ótimo de saída antes ou durante a computação.

#### Família 4: Extensões Cross-Domain (NLP, Reinforcement Learning e Além)
- **Modelos de Linguagem Pré-Treinados com Early-Exit (DeeBERT, FastBERT, BERxiT):** Saídas precoces acopladas após cada bloco de *Self-Attention* e *Feed-Forward* do Transformer, reduzindo a latência de geração ou classificação de texto.
- **Deep Q-Networks com Saída Precoce (EEDQN):** Inclusão de rotas duplas em redes de valor Q. Em estados de jogo ou controle onde uma ação é amplamente dominante (baixa incerteza temporal), o agente seleciona a ação através da saída intermediária, reduzindo os FLOPs de interação contínua.

---

### 1.4 Dicionário de Conceitos-Chave (Pedagógico e Rigoroso)

*   **Rede Dorsal (*Backbone Network*):**  
    A arquitetura neuronal primária profunda (ex.: ResNet, VGG, MobileNet, EfficientNet, BERT, Vision Transformer) que transporta o fluxo principal de extração de representações e à qual são anexados os módulos de saída precoce.
*   **Bloco de Saída / Ramo Lateral (*Exit Block / Side Branch*):**  
    Sub-rede auxiliar acoplada a uma camada intermediária do *backbone*. Tipicamente concebida com um número reduzido de operações (ex.: pooling, convoluções pontuais 1x1, camadas densas) para projetar o mapa latente no espaço de decisão sem gerar overhead computacional apreciável.
*   **Supervisão Profunda (*Deep Supervision*):**  
    Princípio de treino em que funções de perda de classificação são calculadas e retropropagadas a partir de múltiplos pontos intermediários da rede, e não exclusivamente da camada final, facilitando a circulação de gradientes em camadas rasas.
*   **Superpensamento (*Overthinking*):**  
    Fenómeno patológico em redes neurais profundas onde amostras de entrada simples, corretamente identificadas em camadas intermediárias, sofrem degradação de acuidade nas camadas finais devido ao excesso de não-linearidades e processamento de ruído irrelevante.
*   **Política de Saída Precoce (*Early-Exit Policy*):**  
    Algoritmo ou critério de gating responsável por determinar, em tempo de inferência, se a computação deve cessar no ramo $k$ ou prosseguir para as camadas seguintes. Classifica-se em estática (limiares fixos pré-computados) ou dinâmica (adaptada a métricas de sistema e input em tempo real).
*   **Banda de Confiança / Limiar de Decisão (*Confidence Band / Threshold*):**  
    Valores escalares de corte (ex.: $\tau \in [0, 1]$ ou intervalo $[\tau_{\min}, \tau_{\max}]$) contra os quais a métrica de certeza do ramo é comparada. O uso de bandas previne flutuações e decisões oscilatórias em regiões de incerteza marginal.
*   **Interferência de Gradientes e "Dead Layers":**  
    Problema que surge no treino conjunto quando amostras que saem precocemente deixam de propagar gradientes para as camadas profundas da rede, ou quando os objetivos dos ramos laterais competem destrutivamente com a especialização da saída final.
*   **Complexidade de Contorno (*Contour Complexity*):**  
    Métrica de pré-processamento visual ultraleve que quantifica a densidade, curvatura e imprevisibilidade das bordas de uma imagem de entrada, fornecendo uma estimativa prévia da sua dificuldade para alimentar controladores de saída em dispositivos de borda.
*   **Intervalo de Atualização de Decisão (*Decision Update Interval*):**  
    Em arquiteturas distribuídas de computação de borda, refere-se à periodicidade temporal ou número de frames entre atualizações da política de saída calculada por um agente externo, equilibrando latência de comunicação de rede e frescura da decisão.
*   **Redução de FLOPs e Aceleração (*Speedup*):**  
    Métricas centrais de avaliação de eficiência. Os FLOPs (*Floating Point Operations*) quantificam o número teórico de operações aritméticas economizadas por amostra, enquanto a latência ou *speedup* real em hardware mede o tempo físico decorrido em milissegundos num processador específico (CPU, GPU, NPU).
*   **EEDQN (*Early-Exit Deep Q-Network*):**  
    Arquitetura de aprendizagem por reforço profundo que integra vias de inferência duplas em redes de valor Q, permitindo a seleção instantânea de ações em estados de baixa entropia de política e reservando a computação completa para estados críticos ou ambíguos.

---

## 2. Protocolo de Pesquisa Sistemática (Search Strategy)

Para assegurar uma cobertura exaustiva, de elevada qualidade epistemológica e isenta de enviesamento na literatura de *Early-Exit*, agentes autónomos e investigadores devem cumprir o protocolo sistemático detalhado infra.

### 2.1 Venues Científicos Prioritários
As pesquisas bibliográficas devem priorizar estritamente publicações que tenham passado por rigoroso processo de revisão por pares (*peer-review*) nas seguintes conferências e revistas de topo internacional:

| Categoria | Sigla / Nome do Venue | Qualificação / Foco Científico |
|:---|:---|:---|
| **Conferências de IA & Aprendizagem Automática (Core A\*)** | **NeurIPS** (Neural Information Processing Systems) | Fundamentos de computação dinâmica, garantias de convergência e calibração de incerteza |
| | **ICML** (International Conference on Machine Learning) | Otimização de funções de perda multi-saída, regularização e teoria de representação |
| | **ICLR** (International Conference on Learning Representations) | Representação profunda em redes adaptativas, Transformers dinâmicos e destilação |
| | **AAAI** (Association for the Advancement of Artificial Intelligence) | Políticas inteligentes de saída, tomada de decisão sob restrições e modelos híbridos |
| **Conferências de Visão Computacional & NLP** | **CVPR** (Computer Vision and Pattern Recognition) | Arquiteturas CNN e ViT multi-exit para tarefas em tempo real de classificação e deteção |
| | **ICCV** / **ECCV** (Int. Conf. on Computer Vision) | Redução de complexidade visual, análise multi-escala e processamento de vídeo adaptativo |
| | **ACL** / **EMNLP** (Computational Linguistics) | Modelos de linguagem adaptativos com saída precoce (Transformers, DeeBERT, PABEE) |
| **Revistas Científicas de Referência** | **ACM Computing Surveys** (CSUR) | Artigos de síntese e taxonomia abrangente do estado da arte em EE-DNNs |
| | **IEEE T-PAMI** (Trans. on Pattern Analysis and Machine Intelligence) | Fundamentos matemáticos e metodológicos de redes dinâmicas em grande escala |
| | **IEEE T-NNLS** (Trans. on Neural Networks and Learning Systems) | Arquiteturas neuronais, estabilidade de gradientes e otimização distribuída |
| | **IEEE T-MC** (Trans. on Mobile Computing) | Execução de modelos na borda (*edge/IoT*), particionamento hierárquico e gestão energética |
| **Conferências de Sistemas, Hardware & Edge AI** | **ACM MobiSys** / **SenSys** | Implementação de EE-DNNs em hardware restrito, sensores IoT e telemóveis |
| | **IEEE INFOCOM** / **SEC** | Particionamento distribuído Device-Edge-Cloud e protocolos de offloading |
| **Repositórios de Preprints Verificados** | **arXiv** (`cs.LG`, `cs.CV`, `cs.AI`, `cs.RO`) | Avanços recentes e dissertações inovadoras nos últimos 12-24 meses |

---

### 2.2 Equações de Pesquisa Booleanas Recomendadas

As seguintes cadeias de pesquisa encontram-se desenhadas para motores de indexação científica (ex.: IEEE Xplore, Google Scholar, Semantic Scholar, ACM Digital Library e Scopus):

#### Bloco A: Fundamentos Teóricos, Arquiteturas e Early-Exit em CNNs
```text
("early-exit" OR "early exit" OR "multi-exit" OR "dynamic depth") AND ("deep neural networks" OR "convolutional neural networks" OR "CNN") AND ("inference acceleration" OR "computational cost" OR "overthinking" OR "deep supervision")
```

#### Bloco B: Mecanismos de Treino, Destilação e Mitigação de Conflitos de Gradiente
```text
("early exit" OR "multi-exit DNN") AND ("joint training" OR "self-distillation" OR "knowledge distillation" OR "gradient conflict" OR "gradient interference" OR "backbone freezing")
```

#### Bloco C: Políticas Dinâmicas de Saída, Edge AI e Aprendizagem por Reforço
```text
("early-exit" OR "multi-exit") AND ("exit policy" OR "reinforcement learning" OR "Deep Q-Network" OR "EEDQN" OR "edge computing" OR "image complexity" OR "contour complexity")
```

#### Bloco D: Modelos Adaptativos em Linguagem e Visão Avançada
```text
("early exit" OR "early-exit") AND ("Transformer" OR "BERT" OR "Vision Transformer" OR "diffusion models") AND ("patience" OR "entropy threshold" OR "adaptive computation")
```

---

### 2.3 Janela Temporal de Análise

A recolha bibliográfica deve ser dividida em duas janelas temporais interligadas:

1. **Janela Seminal e Fundacional (2014 – 2018):**
   - **Objetivo:** Compreender a transição de redes estáticas para dinâmicas e os alicerces da supervisão profunda.
   - **Marcos Fundacionais:**
     - Lee et al. (AISTATS 2015 - *Deeply-Supervised Nets*): Demonstração formal de que perdas intermediárias combatem o desvanecimento de gradientes e regularizam o modelo.
     - Teerapittayanon et al. (ICPR 2016 - *BranchyNet*): Introdução pioneira do conceito moderno de saída precoce em redes neurais para aceleração e computação tolerante a falhas.
     - Huang et al. (ICLR 2018 - *Multi-Scale Dense Networks - MSDNet*): Prova de que arquiteturas dedicadas com representações multi-escala evitam a limitação semântica de saídas em camadas rasas convencionais.
     - Kang et al. (IEEE Trans. Mobile Computing 2017 - *Neurosurgeon*): Balanço do particionamento computacional entre dispositivo e nuvem.

2. **Janela de Avanços Recentes e Estado da Arte (Últimos 3 a 5 anos):**
   - **Objetivo:** Analisar as fronteiras modernas em políticas dinâmicas aprendidas via RL, modelos Transformer, mitigação de interferências de gradiente e aplicações em edge computing.
   - **Marcos Recentes Relevantes:**
     - Kaya et al. (ICCV 2019 - *Shallow-Deep Networks*): Formalização e caracterização empírica do fenómeno de *overthinking*.
     - Demir & Akbas (2019 - *Early-exit CNNs / EENets*): Treino unificado incorporando diretamente o custo computacional na função de perda.
     - Shanmugham & Vidash (2021 - *Impact of Image Complexity*): Introdução de complexidade visual (contornos) para guiar agentes DQN em saídas de borda.
     - Zhou et al. (NeurIPS 2020 - *PABEE*): Saída precoce baseada em paciência para Transformers em NLP.
     - Rahmath et al. (ACM Comput. Surv. 2024 - *A Comprehensive Survey on Early-Exit DNNs*): Taxonomia integradora e síntese exaustiva de benefícios, métodos de treino e desafios em aberto.
     - Cho (2025 - *EEDQN*): Integração de mecanismos de saída precoce em Deep Q-Networks para acelerar aprendizagem por reforço com garantias de desempenho.

---

## 3. Critérios de Elegibilidade (Inclusão e Exclusão)

Para salvaguardar a robustez epistemológica e a utilidade prática do repositório, cada artigo localizado deve ser submetido aos critérios de triagem detalhados no fluxograma e secções seguintes:

```
                            Artigo Identificado na Pesquisa
                                           |
                    +----------------------+----------------------+
                    |                                             |
                    v                                             v
           Critérios de Inclusão (+)                     Critérios de Exclusão (-)
        - Trade-off acurácia vs. FLOPs                - Sem métricas de custo/latência
        - Comparação com baselines sem EE             - Ramo lateral com overhead excessivo
        - Transparência arquitetural/loss             - Sem baselines comparativos
        - Políticas de saída bem formuladas           - Modelos puramente conceituais
        - Reprodutibilidade e dados públicos          - Sem peer-review fiável
                    |                                             |
                    v                                             v
             ACEITE NA TABELA                             REJEITADO
```

### 3.1 Critérios de Inclusão (+)
O artigo deve cumprir cumulativamente pelo menos **três** dos seguintes requisitos para aceitação:
1. **Avaliação Explícita do Binómio Precisão vs. Custo Computacional:** O artigo não reporta apenas acuidade, mas documenta rigorosamente as economias alcançadas (redução percentual de FLOPs, MACs ou latência física em milissegundos) em relação à rede dorsal original não modificada.
2. **Comparação com *Baselines* Estabelecidos:** Compara a abordagem contra redes de referência estáticas (ex.: ResNet, MobileNet, BERT) e contra métodos seminais de early-exit consolidados (ex.: BranchyNet, MSDNet, Shallow-Deep Networks ou saídas com limiares de entropia).
3. **Transparência Arquitetural e de Treino:** Especifica com precisão matemática a localização dos ramos de saída, a arquitetura interna de cada *side branch* (número de parâmetros adicionados) e a formulação da função de perda multi-saída e fatores de ponderação.
4. **Definição Clara da Política de Decisão:** Apresenta a equação ou algoritmo que rege a interrupção da inferência (ex.: fórmula de entropia, limiares $\tau$, mecanismo de atenção/paciência ou rede de gating).
5. **Validação em Conjuntos de Dados e Tarefas Canónicas:** Demonstra a eficácia em benchmarks reconhecidos pela comunidade científica (ex.: CIFAR-10/100, ImageNet, Tiny-ImageNet, GLUE, OpenAI Gym/Atari, benchmarks de séries temporais).

### 3.2 Critérios de Exclusão (-)
Devem ser sumariamente rejeitados os documentos que apresentem qualquer uma das seguintes lacunas:
1. **Omissão de Métricas de Eficiência:** Trabalhos que alegam acelerar o modelo mas não fornecem valores numéricos concretos de FLOPs, tempo de inferência ou consumo de energia.
2. **Sobrecarga Arquitetural Irrealista (*Negative Speedup*):** Métodos que introduzem ramos intermediários tão volumosos e complexos que o custo de avaliar a primeira saída supera o custo de executar várias camadas da rede dorsal.
3. **Ausência de Comparação Justa:** Ensaios que comparam apenas contra versões deliberadamente empobrecidas ou que não reportam a acuidade do modelo base completo de topo.
4. **Artigos Puramente Conceituais ou sem Validação Empírica:** Textos de opinião, relatórios brutos de blogues, resumos alargados sem secção experimental rigorosa e artigos em revistas predatórias sem processo idóneo de revisão por pares.

---

## 4. Roteiro de Extração e Síntese Analítica

Ao analisar cada publicação validada na triagem de elegibilidade, o curador ou agente de IA deve preencher exaustivamente a seguinte **Checklist de 5 Pontos**:

```
+-----------------------------------------------------------------------------------------+
|                  CHECKLIST ANALÍTICA DE EXTRAÇÃO DE ARTIGOS DE EARLY-EXIT               |
+-----------------------------------------------------------------------------------------+
| 1. Problema e Motivação                                                                 |
|    - Que estrangulamento aborda (latência, overthinking, desvanecimento, edge AI)?      |
|    - Porque falham os modelos estáticos convencionais ou métodos de saída anteriores?   |
+-----------------------------------------------------------------------------------------+
| 2. Inovação Metodológica / Arquitetural                                                 |
|    - Como é desenhado o bloco de saída e onde se conecta no backbone?                   |
|    - Qual a estratégia de treino (Joint, Freeze, Distillation) e a função de perda?     |
|    - Que política de saída é empregue (limiar de entropia, confiança, DRL, paciência)?   |
+-----------------------------------------------------------------------------------------+
| 3. Datasets e Protocolo de Avaliação                                                    |
|    - Quais os conjuntos de dados utilizados e classes de teste?                         |
|    - Em que hardware foram medidos os tempos de latência e consumo energético?          |
+-----------------------------------------------------------------------------------------+
| 4. Resultados Empíricos e Trade-offs (Precisão vs. Eficiência)                          |
|    - Qual a redução percentual de FLOPs e o speedup real em hardware?                   |
|    - Houve degradação, manutenção ou ganho de acuidade face ao modelo estático?         |
+-----------------------------------------------------------------------------------------+
| 5. Limitações e Desafios em Aberto (Open Challenges)                                   |
|    - Sob que condições o método degrada (ruído extremo, desequilíbrio de classes)?      |
|    - Quais os desafios identificados (calibração de saída, conflito de gradientes)?     |
+-----------------------------------------------------------------------------------------+
```

### Detalhe Operacional de Cada Ponto da Checklist:

1. **Problema e Motivação:**  
   Identificar a premissa central contestada pelo artigo (ex.: o custo de avaliar camadas convolucionais terminais em imagens simples, a instabilidade de limiares estáticos sob variações de ruído ou a latência insustentável de Transformers em inferência contínua).
2. **Inovação Metodológica/Arquitetural:**  
   Registar as equações matemáticas chave: formulação da perda conjunta $\mathcal{L}_{total}$, termos de penalização por complexidade computacional, arquitetura dos blocos laterais (*side branches*) e lógica da política de interrupção $\pi_k$.
3. **Datasets e Protocolo de Avaliação:**  
   Documentar os modelos base utilizados (ex.: ResNet-56, MobileNetV2, ViT-B/16, BERT-Base), os conjuntos de dados (ex.: CIFAR-100, ImageNet, GLUE SST-2), e as condições do teste de inferência (batch size = 1 para latência estrita de borda vs. processamento por lotes).
4. **Resultados Empíricos e Trade-offs:**  
   Extrair números exatos: percentagem de amostras que abandonam a inferência em cada ramo (*exit distribution*), redução de FLOPs (ex.: "redução para 20% do custo original"), variação na acuidade Top-1 (ex.: "+0.2%" ou "-0.5%") e aceleração física observada.
5. **Limitações e Desafios em Aberto:**  
   Registar vulnerabilidades admitidas pelos autores (ex.: sobrecarga de memória RAM ao carregar parâmetros adicionais dos ramos, sensibilidade à calibração de temperaturas softmax, ou saturação em batches concorrentes).

---

## 5. Padrão de Formatação para a Tabela de Literatura (`README.md`)

O ficheiro central `README.md` localizado no mesmo diretório organiza o estado da arte numa tabela padronizada de 6 colunas. Quaisquer novas adições devem cumprir escrupulosamente as normas infra.

### 5.1 Especificação Rigorosa das 6 Colunas Obrigatórias

| Coluna | Título | Tipo de Conteúdo e Regras de Formatação |
|:---:|:---|:---|
| **1** | **Nome** | Título oficial e integral do artigo em língua inglesa, sem alterações de grafia ou omissões. |
| **2** | **Detalhes** | Metadados catalográficos estruturados verticalmente com tags HTML `<br><br>`, contendo: Autores, Data de publicação, Livro/Journal/Conferência, Volume, Número, Páginas e hiperligação DOI ativa. |
| **3** | **Abstract** | Citações textuais integrais e fiéis dos excertos mais relevantes do resumo original da publicação, delimitadas obrigatoriamente entre aspas duplas (`"..."`). |
| **4** | **Conclusion** | Citações textuais diretas e exatas retiradas da secção de conclusões ou considerações finais do artigo original, delimitadas entre aspas duplas (`"..."`). |
| **5** | **Resumo (NotebookLM)** | Síntese analítica em língua portuguesa estruturada impreterivelmente em **quatro parágrafos encadeados** separados por tags `<br><br>` (ver secção 5.2). |
| **6** | **Citação** | Referência bibliográfica formatada em estrita conformidade com a **Norma Vancouver (NLM)**, finalizada com a hiperligação DOI funcional. |

---

### 5.2 Estrutura Padronizada do "Resumo (NotebookLM)" (4 Parágrafos)

A quinta coluna deve obedecer rigidamente à seguinte progressão lógica e conceptual:

*   **Parágrafo 1 — Contexto e Problema de Investigação:**  
    Apresentação do trabalho, enquadramento da classe de arquiteturas de *Early-Exit* ou computação adaptativa em causa e explicitação clara do problema, lacuna ou ineficiência que a publicação visa resolver.
*   **Parágrafo 2 — Mecanismo Proposto e Inovação Arquitetural:**  
    Descrição detalhada da metodologia concebida pelos investigadores (desenho dos ramos de saída, formulação da função de perda multi-objetivo, estratégias de destilação ou políticas de decisão estáticas/dinâmicas).
*   **Parágrafo 3 — Validação Experimental e Métricas Numéricas:**  
    Resumo dos protocolos de experimentação (redes base utilizadas, conjuntos de dados de teste, plataformas de hardware) e quantificação rigorosa dos resultados alcançados (reduções percentuais de FLOPs, poupança temporal e manutenção de acuidade).
*   **Parágrafo 4 — Significado Teórico, Limitações e Desafios:**  
    Avaliação do impacto científico da contribuição para a comunidade de IA e computação na borda, acompanhada pela enumeração honesta das limitações teóricas ou práticas e rumos de investigação sugeridos pelos autores.

---

### 5.3 Exemplo Canónico de Registo na Tabela

Para servir de padrão de conformidade e calibração de estilo, apresenta-se de seguida o modelo de preenchimento integral baseado num artigo seminal do repositório:

```markdown
| Early-Exit Deep Neural Network - A Comprehensive Survey | **Autores:** Rahmath P H, Vivek Srivastava, Kuldeep Chaurasia, Roberto G. Pacheco, Rodrigo S. Couto<br><br>**Data de publicação:** 2024<br><br>**Livro/Journal:** ACM Computing Surveys<br><br>**Volume:** 57<br><br>**Número:** 3<br><br>**Páginas:** 1–37<br><br>**DOI:** https://doi.org/10.1145/3698767 | "Early-exit DNNs are multi-exit neural networks that attach many side branches to the conventional DNN, enabling inference to stop early at intermediate points."<br><br>"This approach offers several advantages, including speeding up the inference process, mitigating the vanishing gradients problems, reducing overfitting and overthinking tendencies."<br><br>"This article decomposes the early-exit DNN architecture and reviews the recent advances in the field. The study explores its benefits, designs, training strategies, and adaptive inference mechanisms." | "Early-exit DNNs allow inference to stop early via side branches connected to the conventional DNN architecture. Besides speeding up inferences, they mitigate issues like vanishing gradients, overfitting, and overthinking."<br><br>"This survey underscores significant design issues and research challenges, aiming to stimulate further research in this exciting area."<br><br>"In the coming years, early-exit techniques are poised to become fundamental computational components rather than just nice-to-have features." | Este artigo apresenta uma revisão exaustiva e estruturada sobre o estado da arte em Redes Neuronais Profundas com Saída Antecipada (*Early-Exit Deep Neural Networks* - EE-DNNs), uma das principais abordagens para contornar a rigidez computacional das redes estáticas.<br><br>Os autores decompõem sistematicamente a arquitetura destas redes, detalhando a topologia dos ramos laterais (*side branches*), as estratégias de treino (treino conjunto ponta-a-ponta, treino desacoplado e destilação de conhecimento inter-saídas) e as políticas de inferência adaptativa que regulam a interrupção precoce da computação em função de limiares de incerteza.<br><br>O levantamento demonstra quantitativamente que as EE-DNNs conseguem reduções substanciais no tempo de inferência e no consumo de operações matemáticas (FLOPs), mitigando em simultâneo estrangulamentos clássicos como o desvanecimento de gradientes (*vanishing gradients*), o sobreajuste (*overfitting*) e o fenómeno de superpensamento (*overthinking*), tornando viável a execução de modelos densos em dispositivos de borda (*edge computing*).<br><br>Como orientações futuras, o estudo destaca a necessidade premente de aprimorar a calibração de probabilidade nos ramos intermediários, resolver conflitos de gradiente entre saídas concorrentes durante a otimização conjunta e criar mecanismos de decisão mais resilientes a desvios de distribuição (*out-of-distribution*). | Rahmath P H, Srivastava V, Chaurasia K, Pacheco RG, Couto RS. Early-Exit Deep Neural Network - A Comprehensive Survey. ACM Comput Surv. 2024;57(3):1–37. https://doi.org/10.1145/3698767 |
```

---

## 6. Guardrails de Qualidade & Anti-Alucinação

Para assegurar que o repositório mantém integridade científica inatacável e padrões de publicações de referência internacional, todos os agentes e investigadores devem aderir estritamente aos seguintes guardrails:

### 6.1 Verificação Rigorosa de DOIs e Metadados
- **Resolução Funcional:** Todo o DOI inserido na base documental deve ser verificado em tempo real através do prefixo oficial `https://doi.org/...`. Não é permitida a inclusão de referências com hiperligações quebradas ou sufixos inventados.
- **Auditoria de Catalogação:** Os dados bibliográficos (nomes de autores, título oficial, conferência ou periódico, volume, páginas e ano de publicação) devem ser validados por cruzamento direto com bases bibliográficas fidedignas (CrossRef, DBLP, ACM Digital Library ou IEEE Xplore).

### 6.2 Proibição Estrita de Interpolação e Fabricação de Métricas
- Os ganhos numéricos de eficiência computacional, reduções de FLOPs e taxas de acuidade devem espelhar fielmente o que consta do documento original. Se um artigo reporta "redução de computação para 20% do original no ResNet-44 em CIFAR-10", o curador está **proibido** de generalizar para "reduz cerca de 80% em todas as redes".
- É estritamente vedada a extrapolação de desempenhos empíricos para datasets ou hardwares que não tenham sido testados no artigo fonte.

### 6.3 Fidelidade Literal das Citações Textuais
- Os campos **Abstract** e **Conclusion** destinam-se exclusivamente a **transcrições literais de excertos da obra**. Não é autorizada a introdução de paráfrases, resumos subjetivos ou alteração das palavras originais no interior das aspas delimitadoras.

### 6.4 Sobriedade Epistemológica e Linguagem Científica Neutra
- O texto das sínteses analíticas deve adotar um tom impessoal, académico, equilibrado e rigoroso.
- São expressamente banidos vocábulos promocionais, informais ou sensacionalistas (tais como "revolução paradigmática", "desempenho milagroso", "algoritmo imbatível"). O foco analítico deve recair estritamente sobre as premissas matemáticas, arquitetura, evidências empíricas e limitações metodológicas.
