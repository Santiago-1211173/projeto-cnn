# Protocolo de Curadoria e Navegação na Literatura: Deep Learning

> **Ficheiro Central Associado:** `README.md` (no mesmo diretório)  
> **Natureza do Documento:** Guia metodológico de recolha, análise crítica e referenciação científica para agentes autónomos e investigadores.  
> **Domínio Científico:** Inteligência Artificial / Aprendizagem Automática / Deep Learning (Aprendizagem Profunda) e Arquiteturas Neuronais.

---

## 1. Enquadramento Teórico & Âmbito do Tópico

### 1.1 Definição Canónica na Literatura Internacional de IA
O **Deep Learning (DL)**, ou Aprendizagem Profunda, é uma subárea da Inteligência Artificial e da Aprendizagem Automática (*Machine Learning*) baseada no desenvolvimento e treino de redes neuronais artificiais estruturadas em múltiplas camadas de processamento não-linear.

A sua premissa matemática e concetual basilar consiste na **aprendizagem de representações hierárquicas** (*hierarchical representation learning*). Ao invés de depender de processos manuais de engenharia de atributos (*feature engineering*), concebidos heuristicamente por especialistas humanos, um modelo de *Deep Learning* recebe os dados em formato bruto (como valores brutos de píxeis de imagens, formas de onda acústicas ou séries temporais contínuas de sensores) e transforma-os sucessivamente através de uma composição de funções matemáticas parametrizadas:

$$f(x; \theta) = f^{(L)}\left(f^{(L-1)}\left(\dots f^{(1)}(x; W^{(1)}, b^{(1)})\dots; W^{(L-1)}, b^{(L-1)}\right); W^{(L)}, b^{(L)}\right)$$

onde $L \in \mathbb{N}$ denota a profundidade da arquitetura (número total de camadas), $\theta = \{W^{(l)}, b^{(l)}\}_{l=1}^L$ representa o conjunto global de matrizes de pesos e vetores de enviesamento (*biases*), e $f^{(l)}(z) = \sigma^{(l)}(W^{(l)} z + b^{(l)})$ representa a operação linear seguida da aplicação de uma função de ativação não-linear elementar $\sigma^{(l)}(\cdot)$.

Ao longo deste grafo computacional profundo, o modelo aprende progressivamente um espetro de características distribuídas:
1. **Camadas Iniciais:** Extraem primitivas locais de baixa abstração (ex.: bordas, variações de contraste, transições de fase em frequências locais);
2. **Camadas Intermédias:** Agrupam as primitivas em padrões estruturais mais complexos (ex.: texturas, cantos, motivos geométricos, componentes parciais);
3. **Camadas Profundas:** Codificam conceitos semânticos globais e altamente abstratos (ex.: geometrias de órgãos e lesões em imagiologia médica, identidades de objetos, assinaturas de degradação em sinais industriais).

```
 +---------------------------------------------------------------------------------------+
 |                     FLUXO COMPUTACIONAL HIERÁRQUICO EM DEEP LEARNING                  |
 +---------------------------------------------------------------------------------------+
                                                                                          
     Dados Brutos             Camadas Iniciais           Camadas Intermédias          Camadas Profundas              Saída / Decisão
      (Ex.: Píxeis,              (Primitivas                (Partes, Texturas,           (Conceitos Semânticos         (Classificação,
      Sinais, Séries)             Locais, Bordas)            Motivos Espaciais)           de Alto Nível)                 Regressão, RUL)
                                                                                          
   +-----------------+      +--------------------+      +--------------------+      +-----------------------+      +-----------------+
   |                 |      |                    |      |                    |      |                       |      |  y_chapeu =     |
   |   x in R^{D}    | ---> |    f^{(1)}(x)      | ---> |    f^{(l)}(...)    | ---> |      f^{(L)}(...)     | ---> |  softmax(z) ou  |
   |                 |      |                    |      |                    |      |                       |      |  linear(z)      |
   +-----------------+      +--------------------+      +--------------------+      +-----------------------+      +-----------------+
                                                                                                                            |
                                                                                                                            v
                                                                                                                   +-----------------+
                                                                                                                   | Perda L(y, y^)  |
                                                                                                                   +-----------------+
                                                                                                                            |
   <----------------------------- Retropropagação do Erro (Backpropagation) ------------------------------------------------+
                                       (Ajuste de Pesos via SGD / AdamW)
```

O ajuste dos parâmetros $\theta$ é efetuado através da minimização de uma função de perda empírica $\mathcal{L}(\theta) = \frac{1}{N} \sum_{i=1}^N \ell(f(x_i; \theta), y_i) + \Omega(\theta)$ mediante otimização baseada em gradientes, utilizando o algoritmo da **retropropagação do erro (*backpropagation*)**, que aplica a regra da cadeia do cálculo diferencial para propagar os sinais de erro da saída até à primeira camada.

---

### 1.2 Problema Fundamental que Aborda

O *Deep Learning* surgiu para resolver limitações estruturais intransponíveis dos modelos de aprendizagem estatística tradicionais (*shallow learning*) e das redes neuronais artificiais da primeira e segunda gerações:

1. **A Maldição da Dimensionalidade e o Teorema da Aproximação Universal:**
   - O Teorema da Aproximação Universal (Hornik et al., 1989; Cybenko, 1989) comprovou que uma rede neuronal com apenas uma camada oculta e neurónios suficientes pode aproximar qualquer função contínua arbitrária.
   - Contudo, este resultado clássico não impõe limites sobre o número de unidades necessárias: em problemas de alta dimensionalidade (como visão computacional ou processamento de séries temporais com milhares de variáveis), o número de neurónios numa arquitetura rasa cresce exponencialmente com a dimensão do espaço de entrada ($O(2^D)$), tornando o modelo computacionalmente intratável e estatisticamente propenso a sobreajustamento (*overfitting*).
   - O paradigma profundo explora a **reutilização composicional de representações**: ao empilhar camadas hierárquicas, uma rede profunda pode expressar funções de complexidade equivalente utilizando um número exponencialmente menor de parâmetros ($O(\text{poly}(D))$).

2. **O Colapso do Desaparecimento e Explosão do Gradiente (*Vanishing and Exploding Gradients*):**
   - Durante décadas, o treino de redes com mais de 3 ou 4 camadas estagnava devido ao comportamento do gradiente retropropagado através de funções de ativação saturantes (como a sigmoide logística $\sigma(z) = \frac{1}{1 + e^{-z}}$ e a tangente hiperbólica $\tanh(z)$).
   - Pela regra da cadeia, o gradiente em relação aos pesos da primeira camada envolve produtos cumulativos de derivadas locais:
     $$\frac{\partial \mathcal{L}}{\partial W^{(1)}} = \frac{\partial \mathcal{L}}{\partial a^{(L)}} \left[ \prod_{k=2}^L W^{(k)\top} \text{diag}(\sigma'(z^{(k)})) \right] \frac{\partial a^{(1)}}{\partial W^{(1)}}$$
   - Como a derivada máxima da sigmoide é $\sigma'(z) \le 0.25$, multiplicar sucessivamente matrizes com valores inferiores a 1 fazia com que o gradiente decaísse exponencialmente para zero em camadas anteriores, paralisando a aprendizagem.
   - O *Deep Learning* moderno superou este bloqueio através de duas inovações determinantes:
     - **Funções de Ativação Não-Saturantes:** Notavelmente a Unidade Linear Retificada (**ReLU**, $\text{ReLU}(z) = \max(0, z)$), cuja derivada no semi-eixo positivo é constante e unitária ($\sigma'(z) = 1$ para $z > 0$), eliminando a atenuação do gradiente;
     - **Inicialização Paramétrica Adequada e Normalização:** Esquemas de inicialização adaptados à variância das ativações (He, Glorot/Xavier) e camadas de normalização (*Batch Normalization*, *Layer Normalization*) que controlam a covariância interna dos dados.

3. **O Fenómeno da Degradação e a Revolução das Conexões Residuais (*Skip Connections*):**
   - À medida que os investigadores empilhavam dezenas de camadas lineares e ReLUs, observou-se que modelos com 50 camadas exibiam maior erro de treino do que modelos com 20 camadas, um problema não atribuível a *overfitting*, mas sim à dificuldade de otimização em paisagens de perda não-convexas (*degradation problem*).
   - A introdução da arquitetura **ResNet** (He et al., 2016) resolveu esta barreira ao reformular os blocos funcionais através de conexões de atalho: em vez de forçar as camadas a mapear a função latente subjacente $H(x)$, força-se a aproximação do mapeamento residual $F(x) = H(x) - x$, resultando na formulação:
     $$y = F(x, \{W_i\}) + x$$
   - Isto garante caminhos desimpedidos de propagação de sinal (*gradient highways*), viabilizando o treino estável de redes com centenas ou milhares de camadas.

4. **Invariância Espacial e Conectividade Local:**
   - Redes totalmente conectadas (*Multi-Layer Perceptrons* - MLPs) tratam cada píxel como uma variável independente, descartando a topologia espacial e exigindo uma quantidade impraticável de pesos.
   - O *Deep Learning* para visão estruturou-se em torno de **campos recetivos locais** e **partilha de pesos** (*weight sharing*), formalizados nas Redes Neuronais Convolucionais (CNNs), proporcionando equivariância e invariância à translação.

5. **Desafios Contemporâneos de Generalização sob Dinâmica Não-Estacionária:**
   - A investigação atual foca-se na vulnerabilidade dos modelos profundos perante **deslocamento de domínio (*domain shift*)** (diferença entre o ambiente de treino e o ambiente de teste no mundo real) e **deriva de conceito (*concept drift*)** em fluxos de dados industriais (*data stream learning*), exigindo modelos adaptativos, mecanismos de atenção foveal e arquiteturas robustas baseadas em representações multimodais.

---

### 1.3 Taxonomia e Principais Famílias de Abordagens no Estado da Arte

O espetro do *Deep Learning* contemporâneo compreende cinco grandes famílias de abordagens metodológicas e arquiteturais:

```
                                  Taxonomia de Deep Learning (DL)
                                                 |
         +--------------------+------------------+------------------+--------------------+
         |                    |                  |                  |                    |
   1. Família Visual    2. Família         3. Família         4. Família           5. Família
      Convolucional &      Sequencial &       Transformers &     Híbrida &            Generativa &
      Leve (CNNs/Edge)     Recorrente         Atenção Visual     Visão-Linguagem      Bio-Inspirada
         |                    |                  |                  |                    |
    - Clássicas:         - RNNs padrão      - ViT (Patch-based)- CNN-Transformer:   - Autoencoders &
      VGG, ResNet,       - LSTM (Gates)     - Swin Transformer   CoAtNet, MaxViT      CVAE (Saúde Ind.)
      DenseNet           - GRU                (Janelas Desloc.)- VLMs Contrastivos: - Hard Attention:
    - Leves: MobileNet,  - SERMON (Stream)  - Mecanismos de      CLIP, SigLIP         RAM, MRAM
      ShuffleNet                              Autoatenção                             (Fóvea/Sacadas)
```

#### Família 1: Redes Convolucionais Espaciais e Otimizações Leves (CNNs & Edge DL)
- **Princípio Operacional:** Utilização de operadores de convolução discreta bidimensional com filtros partilhados que deslizam pela imagem, capturando correlações locais entre píxeis vizinhos.
- **Modelos Canónicos Clássicos:**
  - **VGG:** Estabeleceu a regra empírica do empilhamento de pequenos filtros convolucionais $3 \times 3$, demonstrando que múltiplos estágios estreitos superam filtros largos ($7 \times 7$ ou $11 \times 11$) com menor número de parâmetros e maior não-linearidade.
  - **ResNet:** Introdução de conexões residuais (*skip connections*) somadas elemento a elemento ($F(x) + x$), permitindo ultrapassar a barreira das centenas de camadas sem degradação do gradiente.
  - **DenseNet:** Interconexão direta entre todas as camadas de um bloco (*dense connections* com concatenação de mapas de características), maximizando o fluxo de gradiente e a reutilização de atributos.
- **Modelos Otimizados para Hardware com Recursos Limitados (Edge AI):**
  - **MobileNet:** Substituição da convolução padrão por **convoluções separáveis em profundidade (*depthwise separable convolutions*)**, decompondo a operação num filtro espacial por canal (*depthwise* $K \times K \times 1$) seguido de um filtro pontual linear de projeção entre canais (*pointwise* $1 \times 1 \times C$). Reduz o custo computacional e o número de parâmetros por um fator de aproximadamente:
    $$\frac{1}{N} + \frac{1}{K^2} \approx \frac{1}{9} \text{ (para filtros } 3 \times 3 \text{)}$$
  - **ShuffleNet:** Combina convoluções agrupadas pontuais (*group convolutions*) com uma operação de baralhamento de canais (*channel shuffle*), permitindo que a informação flua livremente entre diferentes grupos de canais sem incorrer na sobrecarga computacional de convoluções densas $1 \times 1$.

#### Família 2: Redes Recorrentes e Processamento Sequencial (RNNs / LSTMs)
- **Princípio Operacional:** Processamento de dados estruturados em sequências temporais através da manutenção de um estado oculto interno $h_t$ atualizado a cada instante temporal: $h_t = \tanh(W x_t + U h_{t-1} + b)$.
- **Modelos Canónicos:**
  - **Long Short-Term Memory (LSTM):** Desenvolvida para mitigar o desaparecimento do gradiente temporal em horizontes de longo alcance, introduzindo uma célula de estado $C_t$ governada por três portas diferenciáveis:
    - *Porta de Esquecimento ($f_t$):* Decide que informação descartar do estado passado: $f_t = \sigma(W_f [h_{t-1}, x_t] + b_f)$;
    - *Porta de Entrada ($i_t$):* Decide que novos valores atualizar no estado: $i_t = \sigma(W_i [h_{t-1}, x_t] + b_i)$;
    - *Porta de Saída ($o_t$):* Condiciona a leitura do estado para gerar a ativação oculta externa: $o_t = \sigma(W_o [h_{t-1}, x_t] + b_o)$ com $h_t = o_t \odot \tanh(C_t)$.
  - **Modelos Evolutivos para Fluxos de Dados (ex.: SERMON):** Redes recorrentes capazes de alterar dinamicamente a sua estrutura de neurónios e camadas em tempo de execução para acomodar dados industriais não estacionários e variações operacionais contínuas.

#### Família 3: Vision Transformers (ViTs) e Mecanismos de Autoatenção
- **Princípio Operacional:** Eliminação completa dos operadores de convolução local em favor de mecanismos de **autoatenção escalada multi-cabeça (*Multi-Head Self-Attention - MHSA*)**, calculando pesos de afinidade entre todas as regiões da imagem simultaneamente.
- **Modelos Canónicos:**
  - **Vision Transformer (ViT):** Divide uma imagem $2D$ em grelhas de blocos independentes (*patches* de $16 \times 16$), projeta-os linearmente em vetores de características $1D$, adiciona codificações posicionais (*position embeddings*) e processa-os como uma sequência de tokens. A autoatenção é regida por:
    $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V$$
    Possui menor viés indutivo de localidade face às CNNs, exigindo pré-treino massivo, mas alcançando capacidades superiores de abstração global.
  - **Swin Transformer (Shifted Windows):** Introduz uma estrutura hierárquica multiescala com computação de autoatenção confinada a janelas locais que se deslocam (*shifted windows*) entre camadas sucessivas. Reduz a complexidade computacional da atenção de quadrática $O(N^2)$ para linear $O(N)$ face à resolução da imagem, restaurando a aplicabilidade em deteção e segmentação densa.

#### Família 4: Arquiteturas Híbridas e Modelos Visão-Linguagem (VLMs)
- **Princípio Operacional:** Integração sinérgica das vantagens das convoluções com a expressividade da atenção global, ou alinhamento intermodal entre dados visuais e descrições semânticas em texto.
- **Modelos Canónicos:**
  - **Arquiteturas Híbridas (ex.: CoAtNet, MaxViT):** Posicionam estágios convolucionais nos níveis iniciais da rede (onde o viés indutivo de localidade e invariância à translação é computacionalmente mais eficiente para extrair bordas e texturas) e blocos de autoatenção nos estágios profundos (para modelar interações semânticas globais de longo alcance).
  - **Vision-Language Models (ex.: CLIP, SigLIP):** Modelos pré-treinados contrastivamente em centenas de milhões de pares imagem-texto. O modelo aprende a projetar imagens e frases num espaço latente partilhado, maximizando o produto interno entre pares corretos e minimizando-o para pares incorretos. Destacam-se pela **robustez excepcional sob mudança de domínio (*domain shift*)** e capacidade de classificação sem treino específico prévio (*zero-shot*).

#### Família 5: Modelos Generativos, Autoencoders e Atenção Bio-Inspirada
- **Princípio Operacional:** Modelação não supervisionada de distribuições de dados para compressão e reconstrução, ou reprodução ativa dos padrões de exploração foveal e sacádica do sistema visual humano.
- **Modelos Canónicos:**
  - **Autoencoders e Autoencoders Variacionais Condicionais (CVAE):** Codificam sinais industriais de sensores para variedades latentes de baixa dimensionalidade, permitindo reconstruir sinais nominais e inferir desvios para estimar o estado de degradação estrutural e a Vida Útil Restante (*Remaining Useful Life* - RUL).
  - **Modelos de Atenção Rígida e Visão Foveal (RAM, DRAM, MRAM):** Modelos recorrentes que não processam a imagem inteira de uma só vez. Em vez disso, extraem sequencialmente pequenas janelas de alta resolução (*glimpses*) em coordenadas espaciais selecionadas por uma política estocástica treinada via Aprendizagem por Reforço (algoritmo REINFORCE). Redes multi-nível (ex.: MRAM) desacoplam o controlo motor ocular (movimentos sacádicos de exploração e fixações de detalhe) da execução da tarefa de classificação, resultando em menor custo computacional e maior interpretabilidade biológica.

---

### 1.4 Dicionário de Conceitos-Chave (Pedagógico e Rigoroso)

*   **Hierarquia de Características (*Feature Hierarchy*):**  
    Propriedade fundamental das redes profundas pela qual camadas sucessivas sintetizam representações abstratas e invariantes a partir de representações simples e locais das camadas precedentes.
*   **Conexões de Atalho / Residuais (*Skip/Shortcut Connections*):**  
    Caminhos alternativos no grafo computacional que transportam o sinal de ativação de uma camada diretamente para camadas posteriores ($F(x) + x$), permitindo que os gradientes fluam sem atenuação durante a retropropagação e estabilizando o treino de redes ultra-profundas.
*   **Convolução Separável em Profundidade (*Depthwise Separable Convolution*):**  
    Técnica de fatorização matemática que divide a convolução padrão numa fase espacial independente por canal (*depthwise*) e numa fase pontual de mistura linear de canais (*pointwise* $1 \times 1$), reduzindo drasticamente as operações de vírgula flutuante (FLOPs).
*   **Baralhamento de Canais (*Channel Shuffle*):**  
    Operação de reestruturação de tensores que intercala canais resultantes de convoluções agrupadas distintas, assegurando a comunicação de informação entre grupos sem a necessidade de convoluções densas dispendiosas.
*   **Autoatenção Multi-Cabeça (*Multi-Head Self-Attention - MHSA*):**  
    Mecanismo que projeta os dados de entrada em múltiplos subespaços de *Queries*, *Keys* e *Values*, calculando matrizes de afinidade que ponderam dinamicamente a relevância de cada região ou token em relação a todos os restantes elementos da sequência.
*   **Viés Indutivo (*Inductive Bias*):**  
    Conjunto de pressupostos prévios que uma arquitetura incorpora sobre os dados. As CNNs possuem forte viés indutivo de localidade e equivariância à translação; os Vision Transformers possuem fraco viés indutivo inicial, requerendo maiores volumes de dados para aprender essas regularidades autonomamente.
*   **Atenção Rígida (*Hard Attention*) vs. Atenção Suave (*Soft Attention*):**  
    Atenção suave gera uma máscara ponderada diferenciável contínua sobre toda a imagem (calculada via *softmax*). A atenção rígida seleciona deterministicamente ou estocasticamente uma sub-região discreta (*glimpse* foveal), sendo não-diferenciável e exigindo otimização por gradientes de política (*Policy Gradients*).
*   **Movimentos Sacádicos e Fixações (*Saccades and Fixational Movements*):**  
    Dinâmica inspirada no olho humano: *fixações* concentram a análise foveal de alta resolução em detalhes finos de uma região confinada; *sacadas* representam saltos oculares balísticos rápidos que deslocam o campo de visão para novas regiões da cena.
*   **Deriva de Conceito (*Concept Drift* / *Data Drift*):**  
    Alteração temporal e não-estacionária na distribuição de probabilidade conjunta dos dados $P(X, Y)$, frequente em ambientes industriais devido a variações sazonais, desgaste mecânico de sensores e mudanças de regime operacional de máquinas.
*   **Vida Útil Restante (*Remaining Useful Life - RUL*) & Índice de Saúde (*Health Index*):**  
    Métricas centrais de prognóstico na manutenção preditiva: o Índice de Saúde quantifica o grau relativo de degradação de um equipamento mecânico em relação ao seu estado nominal, e a RUL estima o tempo operacional restante até à ocorrência de uma falha funcional catastrófica.
*   **Mudança de Domínio (*Domain Shift*):**  
    Divergência entre a distribuição estatística dos dados utilizados no treino e a dos dados encontrados no ambiente operacional real ($P_{\text{treino}}(X) \neq P_{\text{teste}}(X)$), testando a capacidade de generalização e invariância dos modelos.
*   **Classificação Multi-Rótulo (*Multi-Label Classification*):**  
    Cenário em que cada exemplo de entrada pode pertencer simultaneamente a múltiplas classes binárias independentes (ex.: uma imagem de fundo de olho apresentando concomitantemente retinopatia diabética e glaucoma), exigindo perdas como *Binary Cross-Entropy* por classe em vez de *Categorical Cross-Entropy*.

---

## 2. Protocolo de Pesquisa Sistemática (Search Strategy)

Para assegurar uma cobertura bibliográfica completa, atualizada e metodologicamente sólida no vasto domínio do *Deep Learning*, agentes autónomos e investigadores devem executar o protocolo estruturado delineado em seguida.

### 2.1 Venues Científicos Prioritários

A recolha bibliográfica deve concentrar-se estritamente em publicações submetidas a arbitragem científica rigorosa (*peer-review*) nas seguintes conferências e revistas internacionais de referência:

| Categoria | Sigla / Nome do Venue | Qualificação / Foco Científico |
|:---|:---|:---|
| **Conferências Principais de IA e ML (Core A\*)** | **NeurIPS** (Neural Information Processing Systems) | Fundamentos teóricos, autoatenção, otimização e modelos generativos |
| | **ICML** (International Conference on Machine Learning) | Algoritmos de treino, teoria de gradientes, generalização e invariâncias |
| | **ICLR** (International Conference on Learning Representations) | Representações profundas, Transformers, modelos auto-supervisionados |
| | **AAAI** (Association for the Advancement of Artificial Intelligence) | Aplicações abrangentes de IA, métodos híbridos e raciocínio neural |
| **Conferências de Visão Computacional (Core A\*)** | **CVPR** (Computer Vision and Pattern Recognition) | Arquiteturas visuais, CNNs, ViTs, modelos multimodais e benchmarking |
| | **ICCV / ECCV** (Int. / European Conf. on Computer Vision) | Modelos fundamentais visuais, segmentação e atenção hierárquica |
| **Revistas Científicas de Alto Fator de Impacto** | **IEEE T-PAMI** (Trans. on Pattern Analysis and Machine Intelligence) | Artigos de referência metodológica profunda em visão e representação |
| | **IEEE T-NNLS** (Trans. on Neural Networks and Learning Systems) | Otimização de redes neuronais, estabilidade de gradientes e convergência |
| | **JMLR** (Journal of Machine Learning Research) | Formulações matemáticas rigorosas e provas teóricas em aprendizagem profunda |
| | **Applied Intelligence / Applied Soft Computing** | Aplicações industriais reais, diagnóstico de falhas mecânicas e prognóstico |
| **Conferências Especializadas e de Domínio** | **ICONIP** (International Conference on Neural Information Processing) | Modelos bio-inspirados, atenção recorrente foveal e neurociência computacional |
| | **MICCAI** (Medical Image Computing and Computer Assisted Intervention) | Redes profundas aplicadas a rastreio e imagiologia biomédica |
| **Repositórios de Preprints Verificados** | **arXiv** (`cs.CV`, `cs.LG`, `cs.AI`, `stat.ML`) | Descobertas recentes e pré-publicações emergentes (últimos 12 a 24 meses) |

---

### 2.2 Equações de Pesquisa Booleanas Recomendadas

As equações infra encontram-se estruturadas e calibradas para execução nos principais motores de busca académicos (Google Scholar, Semantic Scholar, IEEE Xplore, ScienceDirect, ACM Digital Library e Scopus):

#### Bloco A: Benchmarking Comparativo e Arquiteturas Visuais (CNNs vs. ViTs vs. Híbridos vs. VLMs)
```text
("deep learning" OR "neural network") AND ("benchmark" OR "comparative study") AND ("convolutional neural network" OR "CNN") AND ("vision transformer" OR "ViT") AND ("hybrid" OR "vision-language model" OR "CLIP") AND ("image classification" OR "multi-label")
```

#### Bloco B: Otimização Arquitetural e Modelos Leves para Hardware Edge
```text
("deep learning" OR "convolutional neural network") AND ("lightweight" OR "edge AI" OR "real-time") AND ("depthwise separable" OR "MobileNet" OR "ShuffleNet" OR "parameter efficiency" OR "embedded devices")
```

#### Bloco C: Mecanismos de Atenção Rígida, Dinâmica Bio-Inspirada e Visão Foveal
```text
("recurrent attention model" OR "hard attention" OR "visual attention") AND ("fixation" OR "saccadic" OR "saccade") AND ("reinforcement learning" OR "glimpse" OR "foveal vision" OR "MRAM")
```

#### Bloco D: Aplicações Industriais de Deep Learning, Manutenção Preditiva e Concept Drift
```text
("deep learning" OR "machine learning") AND ("fault diagnosis" OR "fault prognosis" OR "predictive maintenance") AND ("remaining useful life" OR "RUL" OR "health index") AND ("concept drift" OR "data stream learning" OR "non-stationary")
```

#### Bloco E: Controlo Robótico Autónomo Baseado em Redes Neuronais vs. Métodos Clássicos
```text
("neural network" OR "deep learning") AND ("mobile robot" OR "AGV" OR "line following") AND ("control" OR "autonomous navigation") AND ("PID" OR "LSTM" OR "CNN")
```

---

### 2.3 Janela Temporal de Análise

A prospeção da literatura deve ser dividida em duas perspetivas temporais complementares:

1. **Janela Seminal e Fundacional (1997 – 2017):**
   - **Objetivo:** Dominar os alicerces teóricos e as arquiteturas canónicas que moldaram a disciplina.
   - **Marcos Históricos Relevantes:**
     - Hochreiter & Schmidhuber (1997 - Formulação da LSTM);
     - LeCun et al. (1998 - LeNet-5 e consolidação do treino de CNNs com gradiente);
     - Krizhevsky et al. (2012 - AlexNet e demonstração prática do poder computacional de GPUs e ReLUs no ImageNet);
     - Simonyan & Zisserman (2014 - VGG e padronização de filtros $3 \times 3$);
     - He et al. (2016 - ResNet e arquitetura de conexões residuais curtas);
     - Huang et al. (2017 - DenseNet e reaproveitamento denso de características);
     - Howard et al. (2017 - MobileNet e convoluções separáveis em profundidade);
     - Vaswani et al. (2017 - *Attention Is All You Need* e introdução do Transformer).

2. **Janela de Avanços Recentes e Estado da Arte (Últimos 3 a 5 anos):**
   - **Objetivo:** Mapear a transição das CNNs clássicas para modelos baseados em autoatenção, arquiteturas híbridas de alta eficiência, redes bio-inspiradas e soluções para ambientes dinâmicos do mundo real.
   - **Marcos Recentes Relevantes:**
     - Dosovitskiy et al. (2020 - Vision Transformer / ViT);
     - Liu et al. (2021 - Swin Transformer e autoatenção em janelas locais deslocadas);
     - Dai et al. (2021 - CoAtNet) e Tu et al. (2022 - MaxViT) (arquiteturas híbridas convolução-atenção);
     - Radford et al. (2021 - CLIP) e Zhai et al. (2023 - SigLIP) (modelos multimodais visão-linguagem robustos a *domain shift*);
     - Pan et al. (ICONIP 2025 - MRAM e modelos recorrentes de atenção hierárquica foveal com sacadas emergentes);
     - Dey et al. (2026 - Benchmarking unificado de CNNs, ViTs, Híbridos e VLMs em tarefas multi-doença).

---

## 3. Critérios de Elegibilidade (Inclusão e Exclusão)

Para preservar o mais elevado rigor epistemológico na compilação do estado da arte em *Deep Learning*, cada artigo identificado na pesquisa sistemática deve ser submetido a uma triagem baseada nos seguintes critérios formais:

```
                       Artigo Identificado na Pesquisa
                                      |
                +---------------------+---------------------+
                |                                           |
                v                                           v
       Critérios de Inclusão (+)                  Critérios de Exclusão (-)
    - Baselines SOTA comparativos              - Sem baselines competitivos
    - Transparência arquitetural/hiperparâm.   - Opacidade de parâmetros e código
    - Métricas de eficiência (FLOPs/Parâm.)    - Validação circular / Data Leakage
    - Protocolo rigoroso de dados              - Estudos conceituais sem teste empírico
                |                                           |
                v                                           v
        ACEITE NO REPOSITÓRIO                       REJEITADO
```

### 3.1 Critérios de Inclusão (+)
O documento deve preencher cumulativamente pelo menos **três** dos seguintes requisitos para ser admitido na tabela de revisão:
1. **Validação Empírica Contra Baselines Reconhecidos:** Comparação explícita da arquitetura proposta contra referências consolidadas da literatura (ex.: ResNet-50, VGG-16, ViT-Base, Swin, MobileNet, ou controladores clássicos como PID em tarefas robóticas) sob condições idênticas de teste.
2. **Transparência Arquitetural e Metodológica:** Descrição precisa das hiperparametrizações utilizadas: dimensões das camadas, funções de ativação, taxas de aprendizagem (*learning rate*), otimizadores (SGD, AdamW), esquemas de redução de taxa (*schedulers*), funções de perda e termos de regularização.
3. **Avaliação Abrangente de Eficiência Computacional e Trade-offs:** Relato explícito de métricas de custo e eficiência além da precisão pura, tais como: contagem de parâmetros treináveis, operações de vírgula flutuante (FLOPs/MACs), latência de inferência (milissegundos por amostra), débito de processamento (FPS) e consumo de memória GPU/RAM.
4. **Rigor e Integridade nos Conjuntos de Dados:** Utilização de bases de dados abertas e reconhecidas pela comunidade científica (ex.: ImageNet, CIFAR-10/100, RFMiD, Messidor-2, C-MAPSS, MNIST/Fashion-MNIST), com divisão clara e isolada entre partições de treino, validação e teste.
5. **Avaliação de Robustez e Generalização Fora de Domínio:** Demonstração empírica de como o modelo reage a perturbações, ruído estocástico, mudança de domínio (*cross-dataset validation*) ou condições de desvio estatístico (*concept drift*).

### 3.2 Critérios de Exclusão (-)
Devem ser sumariamente desqualificados os estudos que manifestem qualquer uma das seguintes insuficiências metodológicas:
1. **Comparações Artificiais ou Ausência de Baselines Relevantes:** Ensaios que comparam o modelo proposto apenas contra palpites aleatórios ou variantes triviais intencionalmente sub-otimizadas, ignorando os modelos de referência dominantes no estado da arte.
2. **Opacidade Arquitetural e Falta de Reprodutibilidade:** Artigos que ocultam detalhes essenciais sobre a configuração das redes, esquemas de aumento de dados (*data augmentation*) ou que utilizam conjuntos de dados privados impossíveis de verificar ou auditar.
3. **Contaminação de Dados (*Data Leakage*) e Validação Falaciosa:** Casos em que amostras do conjunto de teste foram utilizadas no treino ou no ajuste de limiares, ou em que o aumento de dados (*data augmentation*) foi executado antes da partição dos dados, enviesando artificialmente os resultados obtidos.
4. **Artigos Exclusivamente Conceituais ou Opinativos:** Textos de divulgação técnica, ensaios teóricos desprovidos de implementação empírica reproduzível, resumos de conferências sem revisão técnica completa por pares e publicações em periódicos predatórios desprovidos de arbitragem científica fiável.

---

## 4. Roteiro de Extração e Síntese Analítica

Para cada artigo aprovado na triagem de elegibilidade, o curador ou agente deve aplicar obrigatoriamente a seguinte **Checklist de Análise Crítica em 5 Pontos**:

```
+---------------------------------------------------------------------------------+
|               CHECKLIST ANALÍTICA DE EXTRAÇÃO DE ARTIGOS                        |
+---------------------------------------------------------------------------------+
| 1. Problema e Motivação                                                         |
|    - Que limitação específica de representação ou otimização o trabalho aborda? |
|    - Porque falham as arquiteturas convencionais ou métodos prévios no cenário? |
+---------------------------------------------------------------------------------+
| 2. Inovação Metodológica / Arquitetural                                         |
|    - Qual o novo bloco matemático, camada ou mecanismo de atenção introduzido? |
|    - Como difere a propagação de sinal face aos modelos de referência?          |
+---------------------------------------------------------------------------------+
| 3. Datasets e Protocolo de Avaliação                                            |
|    - Quais os conjuntos de dados utilizados (resolução, número de amostras/classes)?|
|    - Como foram configuradas as partições de treino, validação e teste externo? |
+---------------------------------------------------------------------------------+
| 4. Resultados Empíricos e Trade-offs (Precisão vs. Eficiência)                  |
|    - Quais os ganhos absolutos/relativos em métricas-chave (AUC, F1, Accuracy, RUL)?|
|    - Qual o custo em parâmetros, FLOPs e tempo de inferência por amostra?       |
+---------------------------------------------------------------------------------+
| 5. Limitações e Desafios em Aberto (Open Challenges)                           |
|    - Sob que condições de ruído, iluminação ou escala o modelo degrada?        |
|    - Que lacunas de interpretabilidade, robustez ou escalabilidade persistem?   |
+---------------------------------------------------------------------------------+
```

### Detalhe Operacional de Cada Ponto da Checklist:

1. **Problema e Motivação:**
   - Registar com precisão o estrangulamento teórico ou computacional contestado (ex.: complexidade quadrática dos Transformers que impede o processamento de imagens de alta resolução; incapacidade de CNNs capturarem contexto global; perda de sensibilidade em diagnóstico médico com classes altamente desbalanceadas; falência de controladores clássicos em trajetórias robóticas não-lineares).
2. **Inovação Metodológica/Arquitetural:**
   - Isolar a formulação matemática central: descrever as equações da função de perda (ex.: *focal loss*, perdas contrastivas, perdas compostas de atenção), os blocos residuais modificados, os mecanismos de cálculo de atenção ou a divisão de tarefas entre camadas recorrentes.
3. **Datasets e Protocolo de Avaliação:**
   - Catalogar rigorosamente as bases experimentais: número de classes, número de amostras, dimensionalidade das entradas, métodos de balanceamento de classes e métricas de desempenho selecionadas (ex.: AUC-ROC, Macro/Micro F1, Acurácia Top-1/Top-5, RMSE, MAE).
4. **Resultados Empíricos e Trade-offs:**
   - Extrair comparações numéricas objetivas contra o segundo melhor modelo. Avaliar a relação custo-benefício: se um Vision Transformer atinge mais 1,5% de F1-Score do que uma CNN otimizada mas quadruplica os FLOPs e o tempo de inferência, esse compromisso deve ser explicitamente registado.
5. **Limitações e Desafios em Aberto:**
   - Identificar com transparência as fragilidades assumidas pelos autores ou inferidas da metodologia: dependência de pré-treino maciço em hardware topo de gama, sensibilidade a variações de iluminação e ruído de sensores em robótica, ou dificuldade de adaptação a *concept drift* severo em ambientes industriais não-estacionários.

---

## 5. Padrão de Formatação para a Tabela de Literatura (`README.md`)

O ficheiro `README.md` localizado neste mesmo diretório funciona como o repositório consolidado do conhecimento curado. Todas as novas entradas devem seguir escrupulosamente a estrutura de 6 colunas detalhada infra.

### 5.1 Especificação Rigorosa das 6 Colunas Obrigatórias

| Coluna | Título | Tipo de Conteúdo e Regras de Formatação |
|:---:|:---|:---|
| **1** | **Nome** | Título oficial e integral do artigo em língua inglesa, exatamente como publicado pelo editor. |
| **2** | **Detalhes** | Metadados catalográficos estruturados verticalmente utilizando quebras de linha duplas em HTML (`<br><br>`), contendo obrigatoriamente: Autores, Data de publicação, Publisher, Livro/Journal/Conferência, Volume, Número (se aplicável), Páginas e link oficial do DOI. |
| **3** | **Abstract** | Excertos textuais literais e fiéis do resumo original do artigo, delimitados obrigatoriamente entre aspas duplas (`"..."`). Cada citação deve capturar os pontos axiais: problema, arquitetura e conclusões principais. |
| **4** | **Conclusion** | Excertos textuais literais retirados da secção final de conclusões do artigo original, delimitados obrigatoriamente entre aspas duplas (`"..."`). |
| **5** | **Resumo (NotebookLM)** | Síntese analítica aprofundada em língua portuguesa, estruturada de forma estrita em **quatro parágrafos encadeados** separados por `<br><br>` (ver secção 5.2). |
| **6** | **Citação** | Referência bibliográfica completa em conformidade com a **Norma Vancouver (NLM)**, finalizada com a hiperligação DOI ativa. |

---

### 5.2 Estrutura Padronizada do "Resumo (NotebookLM)" (4 Parágrafos)

A coluna 5 deve obedecer impreterivelmente à progressão temática dos quatro parágrafos infra, assegurando clareza analítica e consistência em todo o repositório:

*   **Parágrafo 1 — Contexto e Problema de Investigação:**  
    Apresentação do artigo, introdução da categoria de problema em *Deep Learning* e identificação explícita do desafio técnico (ex.: custo computacional elevado, limitação de baselines clássicos, ausência de estudos unificados comparativos, ou dificuldades de generalização em cenários complexos).
*   **Parágrafo 2 — Mecanismo Proposto e Inovação Arquitetural:**  
    Explicação aprofundada do método, modelo ou protocolo concebido pelos autores (ex.: família de modelos testada, introdução de mecanismos de autoatenção hierárquica, estruturas de convolução separável, ou desacoplamento funcional entre exploração espacial e classificação).
*   **Parágrafo 3 — Validação Experimental e Métricas Numéricas:**  
    Descrição dos conjuntos de dados utilizados (abertos, clínicos ou industriais), baselines de comparação direta e quantificação dos resultados de desempenho alcançados (valores exatos de AUC, F1-Score, precisão, ganhos de eficiência computacional ou comparação com controladores clássicos).
*   **Parágrafo 4 — Significado Teórico, Limitações e Desafios:**  
    Discussão do impacto no estado da arte, aplicabilidade prática em condições reais (ex.: viabilidade em dispositivos com recursos computacionais reduzidos ou sob deslocamento de domínio) e limitações ou direções de trabalho futuro apontadas pelos autores.

---

### 5.3 Exemplo Canónico de Registo na Tabela

Para servir de modelo de referência absoluta, apresenta-se de seguida o exemplo de preenchimento integral de uma entrada na tabela, extraído a partir dos dados do artigo de Dey et al. (2026):

```markdown
| Benchmarking Convolutional, Transformer, Hybrid, and Vision Language Models for Multi Disease Retinal Screening | **Autores:** Durjoy Dey, Aymane Ajba, Yuhong Yan<br><br>**Data de publicação:** 2026<br><br>**Publisher:** 10th International Conference on Medical and Health Informatics, Kyoto, Japan<br><br>**Livro/Journal:** preprint arXiv:2605.26283<br><br>**Volume:** 1<br><br>**Páginas:** 12<br><br>**DOI:** https://doi.org/10.48550/arXiv.2605.26283 | "In this work, we benchmark twelve architectures across four model families: convolutional neural networks (CNNs), vision transformers (ViTs), hybrid CNN–transformer backbones, and vision–language models (VLMs), using the Retinal Fundus Multi-disease Image Dataset (RFMiD), a multi-label fundus dataset with 28 disease classes."<br><br>"On RFMiD, all architectures perform well on the binary screening task (AUC above 84%), but attention-based models dominate: SwinTiny and the hybrid CoAtNet0 and MaxViTTiny achieve the highest AUC and F1, and significantly improve macro and micro F1 in the more challenging multi-label setting."<br><br>"To our knowledge, this is the first study to benchmark convolutional, transformer, hybrid, and vision–language models on RFMiD using a unified evaluation protocol with external validation on Messidor-2." | "Across both settings, attention-based architectures, particularly SwinTiny and the hybrid CoAtNet0 andMaxViTTinymodels, consistently outperform classical convolutional networks, while vision–language models deliver competitive but not superior performance on RFMiD."<br><br>"When evaluated on the external Messidor-2 dataset with a referable diabetic retinopathy endpoint, hybrid and transformermodels again achieve the strongest overall results, and the SigLIP-Base384 vision–language model demonstrates relatively improved robustness under domain and label set shift."<br><br>"More broadly, our results suggest that attention-based vision architectures, particularly Swin-based and hybrid CNN–transformer designs, are strong candidates for real-world retinal screening systems when computational resources permit." | Este artigo avalia e compara doze arquiteturas de *deep learning* inseridas em quatro categorias (Redes Neuronais Convolucionais - CNNs, Vision Transformers - ViTs, modelos híbridos e Vision-Language Models - VLMs) para o rastreio automatizado de múltiplas doenças da retina.<br><br>Utilizando a base de dados RFMiD, o estudo foca-se em duas tarefas: uma de rastreio binário (presença ou ausência de doença) e outra de classificação multirrótulo (detetar o espetro completo de patologias).<br><br>Os resultados demonstram que os modelos baseados em atenção (especialmente o SwinTiny e os modelos híbridos CoAtNet0 e MaxViTTiny) superam consistentemente as CNNs clássicas em ambas as tarefas.<br><br>A validação externa na base de dados Messidor-2 confirmou a superioridade global dos modelos híbridos e transformadores, destacando também a forte robustez do modelo VLM SigLIP-Base384 perante a mudança de domínio e de rótulos. | Dey D, Ajba A, Yan Y. Benchmarking Convolutional, Transformer, Hybrid, and Vision Language Models for Multi Disease Retinal Screening. In: 10th International Conference on Medical and Health Informatics, Kyoto, Japan. arXiv preprint arXiv:2605.26283; 2026. 12 p. https://doi.org/10.48550/arXiv.2605.26283 |
```

---

## 6. Guardrails de Qualidade & Anti-Alucinação

Para garantir a absoluta idoneidade e fiabilidade científica do repositório, qualquer agente de inteligência artificial ou investigador humano que realize a curadoria bibliográfica deve cumprir estritamente os seguintes pontos de controlo e salvaguarda:

### 6.1 Validação Rigorosa de DOIs e Metadados
- **Verificação Ativa da Resolução do DOI:** Cada hiperligação de DOI inserida deve ser inspecionada e validada através de resolução HTTPS no formato canónico `https://doi.org/10.xxxx/...`. É expressamente proibido conjeturar ou inventar prefixos ou sufixos de DOI.
- **Auditoria de Metadados Editoriais:** Os metadados catalográficos (nomes de autores com ortografia correta, data de publicação, nome exato do periódico/conferência, volume, número e paginação) devem ser confrontados diretamente com fontes de indexação internacional consolidadas, como CrossRef, DBLP, PubMed ou a página oficial do editor (Springer, IEEE, ACM, Elsevier, MDPI).

### 6.2 Proibição Estrita de Interpolação ou Arredondamento Fraudulento de Métricas
- **Fidelidade Numérica Absoluta:** As métricas de avaliação reportadas no artigo original devem ser transcritas com exatidão matemática. Se o artigo indica um AUC de $84.2\%$ ou uma amostra de $44$ estudos a partir de $4549$ registos, é estritamente vedado arredondar para "cerca de 85%" ou "aproximadamente 4500 artigos".
- **Identificação Transparente de Métricas:** Devem ser mantidas as distinções estritas entre métricas semelhantes (ex.: discriminar claramente entre *Macro F1* e *Micro F1*, precisão *Top-1* vs. *Top-5*, AUC-ROC vs. AUC-PR).

### 6.3 Fidelidade Literal das Citações Diretas
- As colunas **Abstract** e **Conclusion** constituem registos de prova documental primária. Devem conter exclusivamente passagens textuais literais entre aspas (`"..."`), extraídas do corpo do artigo original. É expressamente interdito efetuar paráfrases, fundir frases distantes sem indicação formal ou introduzir juízos de valor dentro dos campos de citação.

### 6.4 Sobriedade Terminológica e Neutralidade Epistemológica
- As sínteses e resumos analíticos devem manter uma linguagem científica sóbria, objetiva e desprovida de termos promocionais ou hiperbólicos (ex.: banir expressões como "algoritmo revolucionário", "precisão quase mágica", "solução perfeita e definitiva").
- O foco analítico deve recair sempre sobre as propriedades funcionais das redes, os compromissos entre complexidade e desempenho computacional (*trade-offs* de precisão vs. latência/memória) e as limitações de generalização e aplicabilidade no mundo real.
