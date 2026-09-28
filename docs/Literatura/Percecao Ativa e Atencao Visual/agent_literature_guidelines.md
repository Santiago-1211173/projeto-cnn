# Protocolo de Curadoria e Navegação na Literatura: Perceção Ativa e Atenção Visual

> **Ficheiro Central Associado:** `README.md` (no mesmo diretório)  
> **Natureza do Documento:** Guia metodológico de recolha, análise crítica e referenciação científica para agentes autónomos e investigadores.  
> **Domínio Científico:** Inteligência Artificial / Visão por Computador / Perceção Ativa, Atenção Visual e Reconstrução Tridimensional Robótica (*Active Perception, Visual Attention, Uncertainty-Driven Geometric Reconstruction & 3D Scene Understanding*).

---

## 1. Enquadramento Teórico & Âmbito do Tópico

### 1.1 Definição Canónica na Literatura Internacional de IA
O paradigma da **Perceção Ativa** (*Active Perception* ou *Active Vision*) e dos **Mecanismos de Atenção Visual** (*Visual Attention Mechanisms*) define uma das transições metodológicas mais profundas na Visão por Computador contemporânea e na Robótica Autónoma: a passagem de uma observação puramente estática e passiva para uma recolha e processamento seletivo de informação orientado por objetivos, geometria espacial e incerteza.

Formalizado seminalmente por Ruzena Bajcsy (1988), o axioma fundamental da Perceção Ativa estabelece que:
> *"Um agente inteligente vê para agir e age para ver."*

Em contraste com a visão passiva convencional — em que um algoritmo processa passivamente uma matriz de píxeis estática sem capacidade de intervenção, direcionamento ou discernimento focal —, um agente ativo modula as suas estratégias sensoriais e o seu foco computacional de acordo com o estado do ambiente, as incertezas observacionais e os requisitos estritos da tarefa a jusante.

Complementarmente, a **Atenção Visual** em Inteligência Artificial constitui a formalização computacional de alocação de recursos finitos: face à imensidão de dados sensoriais brutos recebidos continuamente pelo sistema percetivo, o modelo calcula dinamicamente uma distribuição de saliência ou ponderação de relevância, priorizando regiões informativas e suprimindo o ruído irrelevante do fundo ou de zonas ocluídas.

Na confluência entre a visão por computador e a robótica móvel/condução autónoma, este domínio aborda criticamente o desafio da **Deteção Monocular de Objetos em 3D** (*Monocular 3D Object Detection*) e da **Estimação de Pose 6DoF** (*6 Degrees of Freedom Pose Estimation*), em que um agente recupera a geometria métrica tridimensional e a orientação espacial de entidades no mundo físico a partir de uma única perspetiva visual bidimensional (RGB).

#### Formulação Matemática Canónica
Considere-se uma imagem monocular calibrada $I \in \mathbb{R}^{H \times W \times 3}$, capturada por uma câmara estenopeica (*pinhole*) com matriz de parâmetros intrínsecos conhecida:

$$K = \begin{bmatrix} f_x & 0 & u_0 \\ 0 & f_y & v_0 \\ 0 & 0 & 1 \end{bmatrix}$$

onde $(f_x, f_y)$ representam as distâncias focais e $(u_0, v_0)$ denotam as coordenadas do ponto principal na grelha de píxeis.

Para cada objeto de interesse presente na cena, a sua pose tridimensional rígida no referencial da câmara é parametrizada pelo vetor $p = [R \mid t]$, onde $R \in SO(3)$ define a matriz de rotação tridimensional (frequentemente simplificada em robótica terrestre pelo ângulo de guinada /*yaw* $\beta$) e $t = [t_x, t_y, t_z]^T \in \mathbb{R}^3$ representa a translação euclidiana do centro do objeto. As dimensões métricas da caixa delimitadora (*bounding box*) 3D são dadas por $s = [l, h, w]^T \in \mathbb{R}^3$ (comprimento, altura e largura).

A projeção de um ponto tridimensional $X = [X, Y, Z]^T$ no espaço da câmara para o plano de imagem $x = [u, v]^T$ obedece à equação de projeção perspectiva:

$$z \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = K \cdot X = K \cdot (R \cdot X_{\text{obj}} + t)$$

onde $X_{\text{obj}}$ representa as coordenadas no referencial próprio do objeto e $z = t_z$ denota a profundidade ortogonal.

```
       +-------------------------------------------------------------------------------------------------+
       |                  FLUXO GERAL DE PERCEÇÃO ATIVA E ATENÇÃO ORIENTADA POR INCERTEZA                 |
       +-------------------------------------------------------------------------------------------------+

                                                 +-----------------------------------+
                                                 | Imagem Monocular 2D de Entrada    |
                                                 |  - Ausência de profundidade z     |
                                                 |  - Oclusões, fundos complexos     |
                                                 +-----------------+-----------------+
                                                                   |
                                                                   v
                                                 +-----------------------------------+
                                                 | Detetor Regional 2D (FPN / RoI)   |
                                                 |  - Delimitação de Propostas (RoI) |
                                                 +-----------------+-----------------+
                                                                   |
                                        +--------------------------+--------------------------+
                                        |                                                     |
                                        v                                                     v
                         +------------------------------+                      +------------------------------+
                         | Extrator Global de Atributos |                      | Decodificador NOC Denso      |
                         |  - Dimensões 3D (s = [l,h,w])|                      |  - Coordenadas 3D (X_noc)    |
                         |  - Vetor latente de forma/ocl|                      |  - Incerteza Aleatória (u_i) |
                         +--------------+---------------+                      +--------------+---------------+
                                        |                                                     |
                                        |                                                     v
                                        |                                      +------------------------------+
                                        |                                      | Mecanismo de Atenção Visual: |
                                        |                                      | Ponderação pela Incerteza    |
                                        |                                      | (Foco no 1º plano fiável)    |
                                        |                                      +--------------+---------------+
                                        |                                                     |
                                        +--------------------------+--------------------------+
                                                                   |
                                                                   v
                                                 +-----------------------------------+
                                                 | Otimizador PnP Guiado por Incerteza|
                                                 |  - Minimização robusta de erro    |
                                                 |  - Estimação de Pose [R | t]      |
                                                 +-----------------+-----------------+
                                                                   |
                                                                   v
                                                 +-----------------------------------+
                                                 | Propagação Analítica de Covariância|
                                                 |  - Matriz Sigma_pose (6x6)        |
                                                 |  - Calibração de Confiança 3D     |
                                                 +-----------------+-----------------+
                                                                   |
                                                                   v
                                                 +-----------------------------------+
                                                 | Módulos Robóticos a Jusante       |
                                                 |  - Rastreamento Estocástico       |
                                                 |  - Planeamento de Trajetórias     |
                                                 +-----------------------------------+
```

Para contornar a ausência de profundidade explícita, a arquitetura moderna aprende **correspondências densas 2D-3D** mapeando cada pixel visível $x_i = [u_i, v_i]^T$ para um ponto em Coordenadas Normalizadas do Objeto (**NOC** - *Normalized Object Coordinates*) $X_{\text{noc}, i} \in [0, 1]^3$, invariantes à escala. A relação geométrica fundamental satisfaz:

$$X_i = R \cdot (X_{\text{noc}, i} \odot s) + t$$

onde $\odot$ representa o produto de Hadamard (multiplicação elemento a elemento).

A atenção visual do sistema manifesta-se através da estimação de uma distribuição heterocedástica de incerteza aleatória para cada pixel:

$$p(X_{\text{noc}, i} \mid I) = \mathcal{N}\left(\hat{X}_{\text{noc}, i}, \text{diag}(\sigma_i^2)\right)$$

onde $\sigma_i^2 \in \mathbb{R}^3$ reflete o ruído observacional, a oclusão e a probabilidade de o pixel pertencer ao fundo. 

O cálculo final da pose tridimensional é executado através de um algoritmo de **Perspective-n-Point (PnP) guiado por incerteza**, que resolve o problema de Máxima Verosimilhança (*Maximum Likelihood Estimation* - MLE) ponderando cada restrição pelo inverso da sua incerteza:

$$\hat{p} = \arg\min_{p} \sum_{i=1}^{N} \left\| \frac{\pi_K(R \cdot (X_{\text{noc}, i} \odot s) + t) - x_i}{\sigma_{\text{proj}, i}} \right\|_{\rho}$$

onde $\pi_K(\cdot)$ denota a projeção perspetiva pela matriz $K$ e $\|\cdot\|_{\rho}$ é uma norma robusta contra *outliers*. 

A incerteza interna da rede é subsequentemente propagada de forma diferenciável através do teorema da função implícita e de matrizes Jacobianas, produzindo uma **matriz de covariância completa $\Sigma_{p} \in \mathbb{R}^{6 \times 6}$**, dotando o agente robótico de um modelo probabilístico calibrado para navegação e segurança.

---

### 1.2 Problema Fundamental que Aborda
A Perceção Ativa e a Atenção Visual orientada por geometria abordam limitações severas e intransponíveis dos modelos neuronais convolucionais e transformadores convencionais:

1. **A Ambiguidade Inerente da Projeção Monocular (O Problema Inverso Mal-Posto):**
   - Na geometria projetiva, uma infinidade de combinações entre tamanho físico tridimensional e distância métrica produz exatamente a mesma projeção 2D na retina do sensor (um objeto pequeno e próximo projeta os mesmos píxeis que um objeto grande e distante).
   - Abordagens passivas convencionais que regridem coordenadas 3D globais diretamente a partir de vetores de atributos globais sofrem de extrema instabilidade numérica e falham catastroficamente perante desvios de distribuição (*domain shift*). A perceção ativa baseada em correspondências densas desacopla a geometria local da pose global, estabelecendo restrições físicas invariantes.

2. **Desperdício Computacional e Interferência de Fundo (The Cluttered Background Dilemma):**
   - Redes neuronais densas padrão processam todas as regiões da imagem com o mesmo orçamento computacional. Em cenários reais exteriores (condução urbana, exploração florestal ou industrial), mais de 80% dos píxeis contêm elementos de fundo irrelevantes (asfalto, céu, vegetação).
   - Forçar o modelo a reconstruir coordenadas em regiões não estruturadas introduz ruído que desestabiliza a convergência dos gradientes. A atenção visual atua como um modulador atencional: as regiões com alta incerteza aleatória são dinamicamente desvalorizadas ou filtradas durante a otimização geométrica.

3. **O Colapso sob Oclusões Severas e Truncamentos:**
   - Métodos clássicos baseados em pontos-chave esparsos (*sparse keypoints*, como os 8 vértices da caixa 3D) colapsam quando um ou mais cantos são ocluídos por outro veículo ou cortados pela margem do sensor.
   - A modelação por correspondência densa assegura redundância maciça: se apenas uma pequena fração da superfície lateral ou frontal estiver visível, os milhares de píxeis sobreviventes fornecem restrições suficientes para que o solucionador PnP reconstrua com precisão a pose tridimensional.

4. **Escassez de Modelos CAD 3D Rígidos e Dependência de Supervisão Manual:**
   - Métodos tradicionais de correspondência 2D-3D exigem modelos CAD tridimensionais perfeitos e pré-fabricados para cada categoria de objeto, ou máscaras de segmentação com anotação manual ao nível do pixel — requisitos inviáveis em grande escala no mundo real.
   - O paradigma auto-supervisionado orientado por incerteza (como no MonoRUn) aprende a geometria e a segmentação implícita apenas com caixas 3D anotadas superficialmente, aproveitando a reprojeção diferencial para separar autonomamente o objeto do fundo sem intervenção humana.

5. **Ausência de Calibração de Confiança para Decisão Crítica em Robótica:**
   - Redes neuronais clássicas são notoriamente sobre-confiantes nas suas predições erradas. Em sistemas autónomos ciber-físicos, alimentar um módulo de controlo e travagem de emergência com coordenadas determinísticas sem margem de erro estimada constitui um perigo existencial.
   - A propagação estocástica de incerteza gera matrizes de covariância formais que alimentam diretamente filtros de Kalman estendidos (EKF) e algoritmos de planeamento de trajetória sob incerteza.

---

### 1.3 Taxonomia e Principais Famílias de Abordagens no Estado da Arte

O espetro contemporâneo de Perceção Ativa e Atenção Visual estrutura-se em quatro famílias metodológicas interconectadas:

```
                                 Taxonomia de Perceção Ativa & Atenção Visual
                                                      |
         +-----------------------+--------------------+-----------------------+-----------------------+
         |                       |                                            |                       |
   1. Mecanismos de        2. Deteção Monocular 3D      3. Perceção Ativa Interativa    4. Modelos Probabilísticos
      Atenção Visual          Orientada por Incerteza      e Robótica Embodied             e Auto-Supervisão
         |                       |                                            |                       |
   - Hard Attention        - Baseados em Âncoras 2D     - Next-Best-View (NBV)          - Incerteza Aleatória
     (Sacadelo/Fóvea/RL)     e Geometria (MonoRUn)      - Active SLAM & Exploração        Heterocedástica
   - Soft Attention        - Baseados em Pseudo-LiDAR   - Visual Saccading Dinâmico     - Perda KL Robusta
     (Diferenciável/STN)     (Weng et al., Wang et al.) - Active Object Search via        (Gaussian/Laplace)
   - Self-Attention        - Baseados em Centros/Pontos   Deep Q-Networks / PPO         - Propagação Analítica
     (Transformers/ViT)      (CenterNet3D, SMOKE)                                         via Jacobiana PnP
```

#### Família 1: Mecanismos de Atenção Visual (Hard, Soft e Foveated Attention)
- **Hard Attention (Atenção Discreta Estocástica):**
  - Inspirada no sistema visual humano, em que o olho executa movimentos sacádicos (*saccades*) orientando a fóvea (zona retiniana de alta resolução angular) para regiões críticas, enquanto a periferia é processada em baixa resolução.
  - Formalizada em modelos como o **Recurrent Models of Visual Attention (RAM)** (Mnih et al., 2014): o agente seleciona iterativamente coordenadas de fixação $l_t$, extrai um recorte sensorial multi-escala (*glimpse*) e atualiza um estado recorrente LSTM. Como o mecanismo de amostragem de coordenadas não é diferenciável, a otimização é conduzida via Aprendizagem por Reforço com o algoritmo REINFORCE.
- **Soft Attention (Atenção Contínua Diferenciável):**
  - Aplica matrizes de ponderação normalizadas via softmax sobre mapas de características latentes, permitindo a propagação retrógrada do gradiente de ponta a ponta.
  - Notabilizada por redes como Spatial Transformer Networks (STN), Squeeze-and-Excitation (SE-Nets), CBAM (Convolutional Block Attention Module) e os modernos Vision Transformers (ViT), que empregam autoatenção de produto escalar escalonado para capturar dependências visuais globais.

#### Família 2: Deteção Monocular 3D Orientada por Incerteza e Correspondências Densas
- **Abordagens de Correspondência Densa 2D-3D e Coordenadas NOC (ex.: MonoRUn):**
  - Substituem a regressão direta de pose por um mapeamento denso pixel-a-ponto no espaço do objeto normalizado. A geometria tridimensional é recuperada analiticamente através de PnP ponderado por incerteza heterocedástica.
  - Vantagens: Altamente robusta a oclusões parciais e generalizável entre instâncias distintas da mesma categoria de veículos/pedestres.
- **Abordagens Baseadas em Pseudo-LiDAR:**
  - Convertem estimativas monoculares de profundidade em nuvens de pontos densas no espaço 3D, aplicando subsequentemente arquiteturas de processamento de nuvens de pontos concebidas para LiDAR (como PointNet++ ou VoxelNet).
  - Limitações: Elevadíssimo custo computacional, latência incompatível com sistemas de borda e forte acumulação de artefactos quando a profundidade monocular comete erros sistemáticos.
- **Abordagens Diretas Baseadas em Pontos-Chave e Centros (ex.: SMOKE, FCOS3D, CenterNet3D):**
  - Tratam objetos como pontos centrais no plano 2D projetado e regridem diretamente offsets 3D, dimensões e ângulos de rotação.
  - Limitações: Suscetíveis a grandes erros de translação longitudinal ($t_z$), uma vez que pequenas imprecisões no plano da imagem traduzem-se em metros de erro no espaço euclidiano real.

#### Família 3: Perceção Ativa Interativa e Embodied AI (Active Exploration & SLAM)
- **Seleção Ativa de Ponto de Vista (*Next-Best-View* - NBV):**
  - O agente robótico com câmara móvel computa ativamente para onde deve deslocar o sensor de forma a maximizar o **Ganho de Informação** (*Information Gain*), minimizando a entropia condicional $H(S \mid A)$ do mapa de reconstrução 3D ou do classificador de objetos.
- **Active SLAM e Exploração Autónoma:**
  - O controlo do veículo autónomo ou braço manipulador é integrado no ciclo percetivo: o robô desvia-se propositadamente da sua rota para reduzir incertezas em marcos visuais (*landmarks*) ou inspecionar superfícies ambíguas.

#### Família 4: Formulações Probabilísticas e Aprendizagem Auto-Supervisionada
- **Incerteza Aleatória vs. Epistémica:**
  - *Incerteza Aleatória (Dados):* Modelada pela rede neural como uma saída direta $\sigma(x)$, capturando o ruído nos sensores, reflexos especulares e oclusões parciais.
  - *Incerteza Epistémica (Modelo):* Capturada através de aproximações Bayesianas, tais como *Monte Carlo Dropout* em tempo de inferência ou *ensembles* neurais.
- **Perda KL Robusta (*Robust KL Loss*):**
  - Funções de perda de regressão probabilística formuladas para mitigar o colapso numérico perante *outliers*. Combinam características Gaussianas e Laplacianas com normalização por lotes de pesos para evitar a explosão de gradientes quando $\sigma \to 0$.

---

### 1.4 Dicionário de Conceitos-Chave Fundamentais

| Conceito | Definição Canónica & Papel na Arquitetura | Analogia / Significado Operacional |
|:---|:---|:---|
| **Perceção Ativa** (*Active Perception*) | Paradigma computacional em que o agente controla ativamente os parâmetros sensoriais (foco, ponto de vista, atenção computacional) para recolher informação relevante. | Ao invés de olhar passivamente para uma fotografia estática, o sistema age como um explorador que move a cabeça e examina detalhes críticos. |
| **Atenção Visual** (*Visual Attention*) | Mecanismo dinâmico de filtragem que atribui diferentes pesos de saliência e recursos computacionais a distintas regiões sensoriais. | Um filtro cognitivo que ignora o ruído do asfalto circundante para concentrar o processamento exclusivamente na forma do peão em travessia. |
| **Deteção Monocular 3D** (*Monocular 3D Detection*) | Tarefa de identificar classes e estimar caixas delimitadoras métricas tridimensionais (dimensões e pose 6DoF) a partir de uma única câmara RGB 2D. | Reconstruir mentalmente a posição e volume real de um camião no mundo tridimensional através de uma única fotografia plana. |
| **Correspondências Densas 2D-3D** | Mapeamento no qual cada pixel individual de uma região de interesse é associado a uma coordenada tridimensional contínua na superfície do objeto. | Uma malha elástica de milhares de fios invisíveis ligando cada pixel da imagem da porta de um carro ao seu ponto 3D na lataria física. |
| **Coordenadas NOC** (*Normalized Object Coordinates*) | Sistema de coordenadas cúbico normalizado no intervalo $[0, 1]^3$ ou $[-0.5, 0.5]^3$, centrado no objeto e independente da sua escala real. | Uma representação universal de "carro canónico" que permite aprender a geometria de sedans e camiões numa mesma escala normalizada. |
| **PnP Guiado por Incerteza** (*Uncertainty-Driven PnP*) | Algoritmo de resolução geométrica que estima a pose ótima de um objeto minimizando erros de reprojeção ponderados pela incerteza de cada pixel. | Um tribunal geométrico onde os píxeis mais fiáveis e nítidos têm voto decisivo, enquanto os píxeis duvidosos ou de fundo têm voto quase nulo. |
| **Incerteza Aleatória** (*Aleatoric Uncertainty*) | Incerteza estocástica inerente aos dados e observações visuais, decorrente de oclusões, reflexos, sombras e ruído eletrónico do sensor. | O ruído de uma imagem tirada com nevoeiro denso: mesmo com a melhor rede do mundo, a informação daquele pixel é intrinsecamente ruidosa. |
| **Incerteza Epistémica** (*Epistemic Uncertainty*) | Incerteza decorrente da ignorância ou falta de dados de treino do próprio modelo sobre uma determinada região ou classe de objetos. | A dúvida do modelo ao encontrar um tipo de veículo exótico ou ambulância que nunca observou durante o seu treino de base. |
| **Perda KL Robusta** (*Robust KL Loss*) | Formulação matemática de função de perda baseada na divergência de Kullback-Leibler, estabilizada contra *outliers* na estimação conjunta de pose e incerteza. | Um amortecedor matemático que impede que píxeis de fundo aberrantes façam os gradientes explodir para infinito durante o treino. |
| **Propagação de Covariância** | Mecanismo analítico baseado em matrizes Jacobianas que transmite as incertezas dos píxeis através do PnP até gerar a matriz $\Sigma_{p}$ da pose final. | Um sistema de cálculo rigoroso que informa os módulos de condução se a distância do carro é de 20 metros $\pm 10$ centímetros ou $\pm 4$ metros. |
| **Movimentos Sacádicos e Fóvea** (*Saccades & Fovea*) | Mecanismo biológico em que a visão periférica de baixa resolução deteta zonas de interesse e dispara movimentos rápidos para focar a fóvea de alta resolução. | Olhar pelo canto do olho para um vulto na estrada e imediatamente focar os olhos com detalhe máximo para reconhecer o perigo. |
| **Erro de Reprojeção** (*Reprojection Error*) | Distância euclidiana no plano da imagem 2D entre um ponto de dados observado e o ponto 3D projetado através da câmara calibrada. | A discrepância visual entre onde a rede pensa que a roda do carro deveria aparecer na fotografia e onde ela realmente surge na imagem. |

---

## 2. Protocolo de Pesquisa Sistemática (Search Strategy)

Para constituir uma base bibliográfica sólida, contemporânea e epistemologicamente inatacável, os investigadores e agentes autónomos devem seguir o protocolo sistemático especificado infra.

### 2.1 Fontes e Venues Prioritários

| Categoria de Publicação | Fóruns Científicos Primários (Venues) | Foco Temático e Relevância |
|:---|:---|:---|
| **Conferências Principais de Visão por Computador** | **CVPR** (IEEE Conf. on Computer Vision and Pattern Recognition)<br>**ICCV** (Int. Conf. on Computer Vision)<br>**ECCV** (European Conf. on Computer Vision) | Artigos pioneiros em deteção 3D monocular, correspondências densas, arquiteturas de atenção visual e auto-supervisão geométrica. |
| **Conferências de Topo em Inteligência Artificial e ML** | **NeurIPS** (Neural Information Processing Systems)<br>**ICML** (Int. Conf. on Machine Learning)<br>**ICLR** (Int. Conf. on Learning Representations)<br>**AAAI** (AAAI Conf. on Artificial Intelligence) | Fundamentos de atenção biológica (RAM), redes bayesianas de incerteza, formulações de perda robusta e aprendizagem por reforço para perceção ativa. |
| **Conferências Principais de Robótica e Sistemas Autónomos** | **IEEE ICRA** (Int. Conf. on Robotics and Automation)<br>**IROS** (IEEE/RSJ Int. Conf. on Intelligent Robots and Systems)<br>**RSS** (Robotics: Science and Systems) | Aplicações reais em navegação autónoma, algoritmos PnP eficientes em robôs, Active SLAM e seleção ativa de pontos de vista (Next-Best-View). |
| **Periódicos Científicos de Alto Impacto** | **IEEE T-PAMI** (Trans. on Pattern Analysis and Machine Intelligence)<br>**IJCV** (International Journal of Computer Vision)<br>**IEEE T-RO** (Transactions on Robotics)<br>**IEEE T-ITS** (Trans. on Intelligent Transportation Systems) | Tratados teóricos aprofundados, validações exaustivas em benchmarks complexos e garantias analíticas de propagação de erro e convergência geométrica. |
| **Repositórios de Preprints Verificados** | **arXiv** (`cs.CV`, `cs.RO`, `cs.AI`, `cs.LG`) | Avanços rápidos de fronteira nos últimos 12 a 24 meses (atenção adaptativa em tempo real, modelos fundacionais de visão 3D). |

---

### 2.2 Equações de Pesquisa Booleanas Recomendadas

As equações seguintes encontram-se estruturadas e calibradas para introdução direta em motores de busca bibliográfica especializados (IEEE Xplore, Google Scholar, Scopus, ACM Digital Library, DBLP e Semantic Scholar):

#### Bloco A: Deteção Monocular 3D, Correspondências Densas e NOC
```text
("monocular 3D object detection" OR "monocular 3D pose estimation") AND ("dense correspondences" OR "normalized object coordinates" OR "NOC" OR "regional reconstruction") AND ("perspective-n-point" OR "PnP")
```

#### Bloco B: Atenção Visual Guiada por Incerteza e Propagação de Covariância
```text
("uncertainty-driven" OR "uncertainty-aware" OR "probabilistic pose estimation") AND ("aleatoric uncertainty" OR "covariance propagation" OR "robust KL loss") AND ("monocular 3D detection" OR "robot perception")
```

#### Bloco C: Atenção Visual Biológica, Foveação e Hard Attention
```text
("visual attention" OR "foveated vision" OR "recurrent attention model" OR "saccadic eye movements") AND ("hard attention" OR "reinforcement learning" OR "active vision") AND ("deep neural networks" OR "CNN")
```

#### Bloco D: Perceção Ativa Interativa, Next-Best-View e Robótica Embodied
```text
("active perception" OR "active visual perception" OR "next-best-view" OR "active SLAM") AND ("information gain" OR "entropy reduction" OR "autonomous navigation" OR "mobile robot")
```

---

### 2.3 Janela Temporal de Análise

A pesquisa e estruturação bibliográfica deve respeitar a bifurcação entre alicerces históricos e fronteira recente:

1. **Janela Seminal e Fundacional (1988 – 2017):**
   - **Objetivo:** Compreender a evolução conceptual dos paradigmas da perceção ativa, os primeiros modelos formais de atenção seletiva e os alicerces matemáticos dos algoritmos de visão geométrica.
   - **Marcos Fundacionais Indispensáveis:**
     - Bajcsy (1988 - *Active Perception*): Formalização original do paradigma agente-ambiente na recolha de dados sensoriais.
     - Itti, Koch & Niebur (IEEE T-PAMI 1998 - *A Model of Saliency-Based Visual Attention for Rapid Scene Analysis*): O modelo computacional seminal de saliência visual biológica baseada em mapas de cor, intensidade e orientação.
     - Lepetit, Moreno-Noguer & Fua (IJCV 2009 - *EPnP: An Accurate O(n) Solution to the PnP Problem*): O algoritmo de complexidade linear fundamental para resolver a correspondência 2D-3D.
     - Mnih et al. (NeurIPS 2014 - *Recurrent Models of Visual Attention - RAM*): A introdução da atenção dura foveada (*hard attention*) treinada via reforço com redes profundas.
     - Kendall & Gal (NeurIPS 2017 - *What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?*): Decomposição matemática rigorosa de incerteza aleatória heterocedástica e incerteza epistémica em redes neuronais de visão.

2. **Janela de Avanços Recentes e Estado da Arte (Últimos 3 a 5 anos):**
   - **Objetivo:** Capturar as metodologias de ponta em representação 3D auto-supervisionada, propagação probabilística para robótica crítica e novos mecanismos de atenção adaptativa.
   - **Marcos Recentes Relevantes:**
     - Wang et al. (CVPR 2019 - *Normalized Object Coordinate Space for Category-Level 6D Object Pose and Size Estimation - NOCS*): Generalização categorial sem dependência de modelos CAD individuais.
     - Chen et al. (CVPR 2021 - *MonoRUn: Monocular 3D Object Detection by Reconstruction and Uncertainty Propagation*): Eliminação da supervisão densa geométrica através de reconstrução regional auto-supervisionada guiada por incerteza e propagação de covariância PnP.
     - Lu et al. (CVPR 2021 - *Geometry Uncertainty Projection Network for Monocular 3D Object Detection - GUPNet*): Modelação formal da amplificação de erro geométrico na profundidade.
     - Liu et al. (ECCV 2022 - *PETR: Position Embedding Transformation for Multi-View 3D Object Detection*): Transformação implícita de coordenadas baseada em transformadores e atenção cruzada (*cross-attention*).
     - Huang et al. (CVPR 2023 - *Tri-Perspective View / BEVDepth*): Fusão de representações atencionais foveadas em visões de pássaro (*Bird's-Eye-View* - BEV) com calibração explícita de profundidade para robótica móvel.

---

## 3. Critérios de Elegibilidade (Inclusão e Exclusão)

Para blindar o repositório contra publicações metodologicamente frágeis ou irrelevantes para sistemas reais, os artigos identificados na estratégia de pesquisa devem ser submetidos à grelha de triagem rigorosa infra:

```
                            Artigo Identificado na Pesquisa
                                           |
                    +----------------------+----------------------+
                    |                                             |
                    v                                             v
           Critérios de Inclusão (+)                     Critérios de Exclusão (-)
        - Reconstrução métrica 3D ou atenção ativa    - Apenas caixas delimitadoras 2D planas
        - Modelação explícita de incerteza/geometria  - Heurísticas sem formulação matemática
        - Avaliação em benchmarks canónicos (KITTI)   - Sem métricas oficiais (AP_3D, IoU 3D)
        - Métricas de latência e custo computacional  - Dependência estrita de CADs perfeitos
        - Reprodutibilidade e transparência de treino - Artigos de opinião sem testes empíricos
                    |                                             |
                    v                                             v
             ACEITE NA TABELA                             REJEITADO
```

### 3.1 Critérios de Inclusão (+)
O documento deve satisfazer cumulativamente pelo menos **três** dos seguintes requisitos para ser admitido na base documental:

1. **Reconstrução Tridimensional Métrica ou Mecanismo de Atenção Seletiva Explícito:**  
   O trabalho aborda a estimativa de geometria e pose no espaço euclidiano tridimensional real ($SE(3)$) a partir de modalidades visuais ou propõe um mecanismo comprovado de alocação dinâmica de atenção (foveação, saliência geométrica ou filtros de incerteza).
2. **Tratamento Rigoroso e Formal da Incerteza e Restrições Físicas:**  
   A metodologia incorpora modelação matemática de incerteza observacional (aleatória/epistémica), funções de perda estocásticas robustas ou mecanismos de propagação analítica de erro através de solvers geométricos (ex.: PnP, Bundle Adjustment, EKF).
3. **Validação Empírica em Benchmarks Canónicos da Comunidade:**  
   Avaliação obrigatória em conjuntos de dados de referência reconhecidos pela literatura internacional (ex.: **KITTI 3D Object Detection Benchmark**, **nuScenes**, **Waymo Open Dataset**, **ScanNet**, ou **SUN RGB-D**).
4. **Reporte Formal de Métricas Oficiais do Domínio:**  
   Apresentação clara de métricas padronizadas: Precisão Média Tridimensional ($AP_{3D}$) e Precisão Média em Vista Superior (*Bird's Eye View* - $AP_{BEV}$) calculadas sob limiares estritos de intersecção sobre união ($\text{IoU} \ge 0.7$ para carros e $\text{IoU} \ge 0.5$ para peões/ciclistas), divididas nas dificuldades oficiais (*Easy*, *Moderate*, *Hard*).
5. **Transparência Arquitetural, Latência e Viabilidade Robótica:**  
   Documentação explícita do tempo de inferência por imagem (latência em milissegundos ou fotogramas por segundo - FPS) e do hardware utilizado, demonstrando compatibilidade ou aproximação a requisitos operacionais de tempo real para robótica autónoma ($\le 100\text{ ms}$).

### 3.2 Critérios de Exclusão (-)
Devem ser prontamente rejeitados os artigos que incorram em qualquer uma das seguintes deficiências:

1. **Confinamento Exclusivo ao Espaço 2D:**  
   Trabalhos de deteção de objetos puramente bidimensionais que geram caixas 2D sem qualquer estimativa de profundidade métrica, orientação tridimensional ou pose espacial no mundo real.
2. **Dependência Irrealista de Modelos CAD 3D Pré-Fabricados para Todas as Classes:**  
   Métodos que requerem um ficheiro CAD tridimensional exato da geometria de cada objeto para funcionar, inviabilizando a sua operação em ambientes abertos e dinâmicos com objetos desconhecidos ou deformáveis.
3. **Ausência de Avaliação nos Protocolos Canónicos de IoU 3D:**  
   Trabalhos que contornam os benchmarks oficiais ou utilizam limiares arbitrários excessivamente brandos (ex.: $\text{IoU} \ge 0.1$) para ocultar desempenhos insuficientes na geometria tridimensional.
4. **Omissão de Métricas de Latência ou Custos Computacionais Inviáveis:**  
   Arquiteturas que alcançam acuidade teórica à custa de pipelines estáticos com múltiplos segundos por imagem (ex.: horas de otimização de feixes sem garantias de tempo real), sem discussão de aplicabilidade robótica.
5. **Publicações Sem Revisão por Pares Idónea ou Relatórios Puramente Opinativos:**  
   Textos de blogues, comunicações comerciais de empresas sem metodologia revelada, ou artigos em publicações sem escrutínio por pares qualificado.

---

## 4. Roteiro de Extração e Síntese Analítica

Para cada publicação admitida após a triagem de elegibilidade, o curador humano ou agente inteligente deve preencher exaustivamente a **Checklist Analítica de 5 Pontos**:

```
+-------------------------------------------------------------------------------------------------+
|               CHECKLIST ANALÍTICA DE EXTRAÇÃO: PERCEÇÃO ATIVA E ATENÇÃO VISUAL                  |
+-------------------------------------------------------------------------------------------------+
| 1. Problema e Motivação                                                                         |
|    - Que lacuna específica aborda (ambiguidade 2D-3D, escassez de CAD/LiDAR, oclusões severas)? |
|    - Porque falham os detetores diretos, abordagens passivas ou estimadores de profundidade?    |
+-------------------------------------------------------------------------------------------------+
| 2. Inovação Metodológica / Arquitetural                                                         |
|    - Como é estruturado o fluxo percetivo (RoI 3D, NOC, PnP diferenciável, Hard/Soft Attention)?|
|    - Que formulação de incerteza é adotada e como é treinada (Robust KL, Laplace, Gaussian)?    |
|    - Como é propagada a incerteza para a pose final (Jacobiana implícita, Covariância)?        |
+-------------------------------------------------------------------------------------------------+
| 3. Datasets e Protocolo de Avaliação                                                            |
|    - Quais os conjuntos de dados (KITTI, nuScenes, Waymo) e classes avaliadas?                  |
|    - Em que condições de hardware foram medidas a latência física e as métricas de inferência?  |
+-------------------------------------------------------------------------------------------------+
| 4. Resultados Empíricos e Trade-offs (Precisão vs. Eficiência)                                  |
|    - Quais os valores numéricos exatos de AP_3D e AP_BEV (Easy, Moderate, Hard)?                |
|    - Qual o tempo de execução (latência em ms / FPS) e o compromisso face a baselines LiDAR?    |
+-------------------------------------------------------------------------------------------------+
| 5. Limitações e Desafios em Aberto (Open Challenges)                                            |
|    - Sob que condições o método degrada (longas distâncias, iluminação adversa, chuva)?         |
|    - Quais as dependências críticas (calibração intrínseca perfeita, acumulação de erro temporal)?|
+-------------------------------------------------------------------------------------------------+
```

### Detalhe Operacional de Cada Ponto da Checklist:

1. **Problema e Motivação:**  
   Isolar a hipótese central contestada pelos autores. Exemplo: *"A regressão direta de pose 3D é excessivamente sensível a pequenos desvios de centro 2D, enquanto métodos baseados em Pseudo-LiDAR são demasiado lentos para robótica em tempo real e os métodos com malhas CAD não generalizam para ambientes não rotulados."*
2. **Inovação Metodológica/Arquitetural:**  
   Registar as formulações matemáticas nucleares: função de perda (ex.: equação da *Robust KL loss*), arquitetura do decodificador de coordenadas (ex.: camadas convolucionais com *pixel-wise uncertainty*), e o mecanismo atencional de ponderação matricial no solucionador PnP.
3. **Datasets e Protocolo de Avaliação:**  
   Registar as divisões de dados exatas (ex.: split oficial de treino/validação de Chen et al. no KITTI, contendo 3.712 imagens de treino e 3.769 de validação) e as especificações de hardware (ex.: GPU NVIDIA TITAN RTX, PyTorch 1.7).
4. **Resultados Empíricos e Trade-offs:**  
   Extrair tabelas quantitativas sem arredondamentos ou extrapolações: relatar os ganhos percentuais de $AP_{3D}$ em relação a concorrentes contemporâneos (ex.: D4LCN, Kinematic3D, MonoDIS) e a latência exata registada (ex.: 0,070 segundos por imagem).
5. **Limitações e Desafios em Aberto:**  
   Registar com honestidade científica as deficiências admitidas pelos autores e identificadas pela comunidade: perda de densidade geométrica em alvos além dos 40 metros de distância, degradação em condições noturnas com faróis ofuscantes e sensibilidade a desalinhamentos na calibração estática da câmara.

---

## 5. Padrão de Formatação para a Tabela de Literatura (`README.md`)

O ficheiro central `README.md` no mesmo diretório reúne e apresenta o estado da arte numa tabela padronizada de 6 colunas. Qualquer artigo adicionado deve cumprir escrupulosamente as normas estabelecidas infra.

### 5.1 Especificação Rigorosa das 6 Colunas Obrigatórias

| Coluna | Título | Tipo de Conteúdo e Regras de Formatação |
|:---:|:---|:---|
| **1** | **Nome** | Título oficial integral da publicação em língua inglesa, exatamente idêntico ao original registado no DOI. |
| **2** | **Detalhes** | Metadados catalográficos estruturados verticalmente por tags HTML `<br><br>`:<br>`**Autores:** [Lista de autores]`<br>`**Data de publicação:** [Ano]`<br>`**Livro/Journal:** [Conferência ou Revista]`<br>`**Volume:** [Se aplicável]`<br>`**Número:** [Se aplicável]`<br>`**Páginas:** [Intervalo de páginas]`<br>`**DOI:** [Hiperligação funcional https://doi.org/...]` |
| **3** | **Abstract** | Transcrições textuais integrais e fiéis dos excertos mais relevantes do resumo original da publicação, delimitadas obrigatoriamente entre aspas duplas (`"..."`) e separadas por `<br><br>`. |
| **4** | **Conclusion** | Transcrições literais diretas e exatas retiradas da secção de conclusões ou considerações finais do artigo original, delimitadas entre aspas duplas (`"..."`) e separadas por `<br><br>`. |
| **5** | **Resumo (NotebookLM)** | Síntese analítica crítica redigida em língua portuguesa, organizada obrigatoriamente em **quatro parágrafos encadeados** separados por tags `<br><br>` (ver secção 5.2). |
| **6** | **Citação** | Referência bibliográfica académica formal formatada em estrita conformidade com a **Norma Vancouver (NLM)**, finalizada com a hiperligação DOI ativa. |

---

### 5.2 Estrutura Padronizada do "Resumo (NotebookLM)" (4 Parágrafos)

A quinta coluna deve obedecer rigidamente à seguinte progressão conceptual e lógica em quatro parágrafos:

*   **Parágrafo 1 — Contexto e Problema de Investigação:**  
    Apresentação do artigo, contextualização da tarefa específica de perceção ativa, atenção visual ou reconstrução tridimensional robótica em causa e formulação inequívoca da limitação ou estrangulamento físico/computacional que o trabalho combate.
*   **Parágrafo 2 — Mecanismo Proposto e Inovação Arquitetural:**  
    Descrição minuciosa da abordagem metodológica concebida pelos autores (arquitetura dos ramos de rede, representação em coordenadas normalizadas NOC, modelação matemática de incerteza, perda robusta de reprojeção e solver PnP ponderado).
*   **Parágrafo 3 — Validação Experimental e Métricas Numéricas:**  
    Resumo rigoroso do protocolo de testes (benchmarks empregues, classes avaliadas, condições de experimentação) com relato explícito e exato das métricas de desempenho tridimensional ($AP_{3D}$, $AP_{BEV}$) e dos tempos de execução físicos obtidos em hardware real.
*   **Parágrafo 4 — Significado Teórico, Limitações e Desafios em Aberto:**  
    Apreciação do impacto da contribuição na comunidade científica de inteligência artificial e robótica, acompanhada pela identificação honesta de vulnerabilidades práticas, limites de generalização e rumos de investigação futura delineados.

---

### 5.3 Exemplo Canónico de Registo na Tabela

Para calibração de estilo e verificação de conformidade, apresenta-se de seguida o modelo de preenchimento integral com base no artigo seminal que encabeça o repositório da Fase 9:

```markdown
| MonoRUn: Monocular 3D Object Detection by Reconstruction and Uncertainty Propagation | **Autores:** Chen et al.<br><br>**Data de publicação:** 2021<br><br>**Livro/Journal:** IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)<br><br>**Páginas:** 7879–7888<br><br>**DOI:** https://doi.org/10.48550/arXiv.2103.12605 | "Object localization in 3D space is a challenging aspect in monocular 3D object detection."<br><br>"To address this issue, we propose MonoRUn, a novel detection framework that learns dense correspondences and geometry in a self-supervised manner, with simple 3D bounding box annotations."<br><br>"To regress the pixel-related 3D object coordinates, we employ a regional reconstruction network with uncertainty awareness. For self-supervised training, the predicted 3D coordinates are projected back to the image plane. A Robust KL loss is proposed to minimize the uncertainty-weighted reprojection error."<br><br>"During testing phase, we exploit the network uncertainty by propagating it through all downstream modules. More specifically, the uncertainty-driven PnP algorithm is leveraged to estimate object pose and its covariance."<br><br>"Extensive experiments demonstrate that our proposed approach outperforms current state-of-the-art methods on KITTI benchmark." | "We presented the MonoRUn framework, a novel monocular 3D object detector with state-of-the-art performance and high practicality."<br><br>"To employ dense correspondence method for 3D detection in real driving scenes, we overcame the deficiency of geometry supervision by self-supervised reconstruction with uncertainty awareness."<br><br>"Meanwhile, we made uncertainty-aware deep regression networks easier to optimize by proposing the Robust KL loss."<br><br>"Finally, we are among the first to explore probabilistic 3D object localization by uncertainty propagation through PnP, which may open up new possibilities for downstream tasks such as robust tracking and motion prediction." | Este artigo introduz o MonoRUn, uma nova estrutura de deteção de objetos 3D a partir de imagens monoculares concebida para cenários reais de condução autónoma. A localização de objetos no espaço 3D representa um grande desafio na visão por computador pela ausência de informação explícita de profundidade em imagens 2D.<br><br>Para ultrapassar a necessidade de modelos 3D exatos de objetos ou anotações geométricas detalhadas (difíceis de obter em grande escala em ambientes exteriores), o MonoRUn aprende correspondências 2D-3D densas e a geometria dos objetos de forma auto-supervisionada (*self-supervised manner*), utilizando apenas anotações simples de caixas delimitadoras 3D (*3D bounding boxes*).<br><br>O modelo estima coordenadas 3D normalizadas (*Normalized Object Coordinates* - NOC) por meio de uma rede de reconstrução regional sensível à incerteza. Na fase de treino, as coordenadas 3D previstas são re-projetadas no plano da imagem utilizando a pose real (*ground truth*) e os parâmetros intrínsecos da câmara. Para lidar com *pixels* de fundo e oclusões sem necessitar de anotações de segmentação, os autores introduzem a perda *Robust KL loss*, que minimiza o erro de reprojeção ponderado pela incerteza aleatória.<br><br>Durante a inferência, a incerteza da rede é propagada através de um algoritmo PnP (*Perspective-n-Point*) orientado pela incerteza, permitindo estimar a pose do objeto e a respetiva matriz de covariância. Nos testes realizados no conjunto de dados de referência KITTI, o MonoRUn alcançou resultados *state-of-the-art* na deteção 3D com um tempo de execução médio de apenas 0,070 segundos por imagem. | Chen H, Huang Y, Tian W, Gao Z, Xiong L. MonoRUn: Monocular 3D Object Detection by Reconstruction and Uncertainty Propagation. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). arXiv preprint arXiv:2103.12605; 2021. https://doi.org/10.48550/arXiv.2103.12605. |
```

---

## 6. Guardrails de Qualidade & Anti-Alucinação

Para garantir que a base bibliográfica mantém a máxima idoneidade académica, reprodutibilidade científica e precisão factual, todos os agentes autónomos e investigadores devem obedecer categoricamente aos seguintes guardrails operacionais:

### 6.1 Validação Estrita de DOIs e Resolução Web em Tempo Real
- **Resolução Funcional:** Todo o DOI registado na tabela deve ser verificado e resolvido via servidor oficial através do prefixo canónico `https://doi.org/...` ou no arquivo de preprints `https://doi.org/10.48550/arXiv...`. É terminantemente proibido registar DOIs hipotéticos, URLs truncados ou ligações quebradas.
- **Auditoria Cruzada de Metadados:** Os nomes dos autores, título oficial em inglês, veículo de publicação, ano e paginação devem ser conferidos por cruzamento direto com bases de dados fiáveis (CrossRef, DBLP, Google Scholar ou IEEE Xplore).

### 6.2 Proibição Estrita de Fabricação ou Arredondamento de Métricas Numéricas
- As métricas de desempenho físico e quantitativo reportadas no resumo analítico devem reproduzir estritamente os valores expressos no artigo original.
- Se o artigo reporta um tempo de inferência de **0,070 segundos por imagem** num teste com GPU NVIDIA TITAN RTX, o curador está **proibido** de generalizar de forma vaga para *"executa em cerca de cem milissegundos"* ou *"ultrapassa os 20 FPS em qualquer computador"*.
- É formalmente vedada a extrapolação de taxas de precisão ($AP_{3D}$) para condições ou conjuntos de dados que não tenham sido objeto de experimentação empírica pelos autores originais.

### 6.3 Fidelidade Literal das Citações Textuais
- Os campos **Abstract** e **Conclusion** destinam-se exclusivamente a **citações literais e fiéis do texto publicado em inglês**.
- É terminantemente proibido inserir paráfrases, textos gerados por inteligência artificial, supressões que alterem o sentido original ou traduções nos campos de citação. Omissões pontuais devem ser assinaladas pela notação convencional entre colchetes `[...]`.

### 6.4 Sobriedade Epistemológica e Linguagem Científica Neutra
- O texto em língua portuguesa dos resumos analíticos deve obedecer aos mais estritos padrões da redação académica e científica: tom impessoal, rigor concetual, precisão terminológica e sobriedade de juízo.
- São expressamente banidos adjetivos hiperbólicos, vocabulário sensacionalista e alegações promocionais não fundamentadas (ex.: *"método revolucionário milagroso"*, *"precisão infalível"*, *"algoritmo perfeito para qualquer robô"*). O foco deve concentrar-se exclusivamente nas premissas matemáticas, na geometria demonstrada, nas métricas empíricas aferidas e nas limitações reais do sistema.
