# Protocolo de Curadoria e Navegação na Literatura: Out-of-Distribution (OOD) em Redes Neuronais

> **Ficheiro Central Associado:** `README.md` (no mesmo diretório)  
> **Natureza do Documento:** Guia metodológico de recolha, análise crítica e referenciação científica para agentes autónomos e investigadores.  
> **Domínio Científico:** Inteligência Artificial / Aprendizagem Automática / Deteção de Out-of-Distribution (OOD), Quantificação de Incerteza e Fiabilidade de Redes Neuronais Profundas (*Trustworthy Deep Learning & Uncertainty Estimation*).

---

## 1. Enquadramento Teórico & Âmbito do Tópico

### 1.1 Definição Canónica na Literatura Internacional de IA
A deteção de amostras fora de distribuição (**Out-of-Distribution - OOD Detection**) é a área fundamental da fiabilidade e segurança em Inteligência Artificial que visa dotar modelos preditivos da capacidade de determinar se uma dada entrada em tempo de inferência provém da mesma distribuição estatística subjacente aos dados utilizados no seu treino, designada por distribuição em-domínio (**In-Distribution - ID**), ou se, pelo contrário, constitui uma instância anómala, desconhecida ou semanticamente disjunta (**Out-of-Distribution - OOD**).

Formalmente, considere-se um espaço de entrada $\mathcal{X} \subset \mathbb{R}^d$ e um conjunto finito de classes fechadas $\mathcal{Y}_{\text{in}} = \{1, 2, \dots, C\}$. Durante a fase de treino supervisionado, a rede neuronal profunda é otimizada sobre amostras independentes e identicamente distribuídas (i.i.d.) extraídas de uma distribuição conjunta em-domínio:
$$(x, y) \sim P_{\text{in}}(X, Y)$$

Em tempo de teste e operação no mundo real, o modelo é confrontado com uma distribuição marginal mista $P_{\text{test}}(X)$, a qual pode ser decomposta numa mistura estatística:
$$P_{\text{test}}(X) = (1 - \pi) P_{\text{in}}(X) + \pi P_{\text{out}}(X)$$
onde $\pi \in [0, 1]$ denota a probabilidade a priori de ocorrência de uma anomalia e $P_{\text{out}}(X)$ representa a distribuição OOD. O suporte semântico dos dados gerados por $P_{\text{out}}$ está tipicamente associado a classes ausentes durante o treino, tal que:
$$\mathcal{Y}_{\text{out}} \cap \mathcal{Y}_{\text{in}} = \emptyset$$

A tarefa de deteção de OOD consiste num problema de teste de hipóteses binário formulado sobre uma função de pontuação real (*scoring function*) $S: \mathcal{X} \to \mathbb{R}$ e um limiar de decisão pré-estabelecido $\gamma \in \mathbb{R}$:
$$G_\gamma(x) = \begin{cases} 
\text{ID (Aceitar)}, & \text{se } S(x) \ge \gamma \\ 
\text{OOD (Rejeitar)}, & \text{se } S(x) < \gamma 
\end{cases}$$

```
=========================================================================================
          ESPAÇO LATENTE DE CARACTERÍSTICAS (FEATURE SPACE) E FRONTEIRAS DE REJEIÇÃO
=========================================================================================

                      ^ Característica z2
                      |
                      |            +-------------------+
                      |            | Classe ID 1 (C_1) |
                      |            |    (mu_1, Sigma)  |
                      |            +-------------------+
                      |                  * * * * *
                      |                *  x_ID1   *
                      |                 * * * * *
                      |
                      |                                       [ Amostra Near-OOD ]
                      |                                              x_near
                      |                                         (Semelhante a C2,
                      |                                        mas classe invisível)
                      |
                      |     +-------------------+
                      |     | Classe ID 2 (C_2) |
                      |     |    (mu_2, Sigma)  |
                      |     +-------------------+
                      |           * * * * *
                      |         *  x_ID2    *
                      |          * * * * *
                      |
                      |                                          [ Amostra Far-OOD ]
                      |                                                x_far
                      |                                         (Domínio completamente
                      |                                               alienígena)
                      +------------------------------------------------------------> 
                     0                                              Característica z1

  Legenda:
  ( * * * ) : Regiões de Alta Densidade ID delimitadas pela Distância de Mahalanobis / Mahalanobis++
  x_near    : Near-OOD (Alta similaridade contextual, baixa probabilidade condicional)
  x_far     : Far-OOD (Espaço latente residual, norma reduzida ou estatísticas espúrias)
```

A literatura divide os desvios de distribuição em duas categorias principais de complexidade:
1. **Far-OOD:** Amostras com discrepâncias estatísticas, semânticas e contextuais extremas em relação aos dados de treino (exemplo canónico: classificador treinado em dígitos manuscritos MNIST testado sobre fotografias naturais CIFAR-10, ou classificador de mamografia confrontado com fotografias de animais).
2. **Near-OOD:** Amostras que partilham o mesmo tipo de sensor, estilo visual ou semântica global com os dados ID, mas pertencem a subclasses ou patologias nunca observadas durante a fase de treino (exemplo: modelo treinado para diagnosticar Degeneração Macular Relacionada com a Idade em Tomografia de Coerência Ótica [OCT] confrontado com Oclusão da Veia Retiniana ou Doença de Stargardt). Este cenário representa a maior causa de falhas catastróficas em aplicações reais.

---

### 1.2 Problema Fundamental que Aborda
A investigação em deteção de Out-of-Distribution responde a quatro limitações estruturais severas inerentes às redes neuronais profundas contemporâneas:

1. **Hiper-Confiança Patológica do Classificador Softmax (*The Overconfidence Trap*):**
   - Classificadores neurais padrão utilizam a função *softmax* aplicada a vetores de ativação linear (*logits*) $z(x)$:
     $$p(y = c \mid x) = \frac{\exp(z_c(x))}{\sum_{j=1}^C \exp(z_j(x))}$$
   - A função *softmax* divide o espaço de entrada em regiões poliédricas infinitas delimitadas por hiperplanos de decisão. Pontos que se encontram a distâncias arbitrárias de todos os dados de treino (mesmo em regiões de densidade nula) continuam a cair no lado positivo de algum hiperplano, gerando distribuições de pseudo-probabilidade normalizadas com confiança próxima de 100% ($p \approx 1.0$) em entradas completamente absurdas ou inexistentes.

2. **Distorção Geométrica induzida pelo Colapso Neural (*Neural Collapse*):**
   - No regime terminal de treino (*terminal phase of training*), com otimizadores estocásticos e funções de perda de entropia cruzada, verifica-se o fenómeno de *Neural Collapse*: as representações latentes das classes ID na penúltima camada colapsam geometricamente para os vértices de uma estrutura rígida tipo Simplex Equiangular (*Equiangular Tight Frame* - ETF).
   - Este colapso suprime drasticamente a variância intra-classe nas direções discriminativas, o que é benéfico para a taxa de erro ID, mas projeta amostras anómalas em direções residuais não mapeadas, levando a estimativas de covariância subótimas e permitindo que anomalias colidam com os centróides aprendidos.

3. **Viés da Norma e Violação das Premissas Gaussianas (*Feature Norm Bias*):**
   - Métodos baseados em distâncias estatísticas (como a clássica distância de Mahalanobis) assumem que as características $z(x)$ seguem uma distribuição Gaussiana multivariada $\mathcal{N}(\mu_c, \Sigma)$. No entanto, ativações após funções ReLU ou normalizações em redes modernas apresentam caudas pesadas (*heavy tails*) e fortes flutuações na norma euclidiana $\|z(x)\|_2$.
   - Amostras OOD desprovidas de textura (ruído gaussiano, imagens uniformes) produzem representações com norma muito pequena, aproximando-se artificialmente da média global e enganando detetores clássicos que não normalizam a escala do espaço vetorial.

4. **Separação entre Incerteza Epistémica e Incerteza Aleatória:**
   - As redes neuronais tradicionais não discriminam a fonte da sua hesitação:
     * **Incerteza Aleatória (AU):** Ruído intrínseco aos dados, ambiguidade na fronteira de decisão ou oclusão parcial;
     * **Incerteza Epistémica (EU):** Falta de conhecimento e ausência de suporte no conjunto de dados de treino (*lack of evidence*).
   - Apenas a incerteza epistémica é indicativa genuína de desvio OOD, exigindo que os modelos extraiam métricas de representação e densidade geométrica em vez de puras dispersões entrópicas de saída.

---

### 1.3 Taxonomia e Principais Famílias de Abordagens no Estado da Arte

O ecossistema contemporâneo de deteção de OOD organiza-se em cinco grandes famílias teóricas e metodológicas:

```
                                  Taxonomia de Deteção de OOD em IA
                                                 |
         +-------------------+-------------------+-------------------+-------------------+
         |                   |                   |                   |                   |
  1. Baseados no      2. Baseados no      3. Exposição a      4. Métodos Híbridos 5. Extensões Cross-
     Espaço de           Espaço Latente      Outliers (OE)       e Teoria de         Architecture
     Saída / Logits      (Feature Space)     e Regularização     Covariância         e Fundacionais
         |                   |                   |                   |                   |
   - MSP Baseline      - Mahalanobis       - Outlier Exposure  - HUE-OOD           - CNNs (ResNet,
   - ODIN (T + pert)     (Lee et al.)        (OE Clássico)       (Fusão Rank-based)    EfficientNet)
   - Energy Score      - Mahalanobis++     - Few-Shot OE       - Ensemble EU/AU    - Vision Transformers
   - ReAct               (L2 Norm Sphere)    (Reject Bucket)   - Estimadores MMSE    (ViT, Swin, ConvNeXt)
   - Evidência         - MM++ Multilayer   - Regularização por   (Ledoit-Wolf,     - VLM / Auto-sup.
     Dirichlet           Entropy Drops       Label Embeddings    RBLW, OAS)          (CLIP, DINO)
```

#### Família 1: Métodos Baseados no Espaço de Saída e Logits (*Post-Hoc Output Space*)
Atuam diretamente sobre as probabilidades ou *logits* gerados pela rede pré-treinada:
- **Maximum Softmax Probability (MSP):** O limiar de corte $S(x) = \max_c p(y=c \mid x)$ constitui o baseline canónico de Hendrycks & Gimpel. É rápido, mas altamente vulnerável à sobreconfiança.
- **ODIN (Out-of-DIstribution detector for Neural networks):** Aplica escalonamento de temperatura aos logits $z(x)/T$ com $T > 1$ e adiciona pequenas perturbações adversariais guiadas pelo gradiente na entrada de teste $\tilde{x} = x - \epsilon \operatorname{sign}(-\nabla_x \log S_{\hat{y}}(x; T))$, aumentando a discrepância entre amostras ID e OOD.
- **Energy-based OOD Score:** Substitui a probabilidade softmax pela função de energia Helmholtz $E(x; f) = -T \cdot \log \sum_{c=1}^C \exp(z_c(x)/T)$. Ao contrário do softmax, a energia não força uma soma unitária, correlacionando-se monotonicamente com a densidade marginal de probabilidade $\log p(x)$.
- **ReAct (Rectified Activation):** Observa que amostras OOD ativam de forma desproporcional nós específicos nas camadas terminais; ao truncar as ativações com um teto percentilar $h_l = \min(h_l, \tau)$, reequilibra as pontuações energéticas.
- **Formalismos Evidenciais e Dirichlet:** Parametrizam a saída através de uma distribuição de Dirichlet sobre o simplex categórico via ativações não-negativas (ex.: *softplus*), modelando explicitamente a falta de evidência como incerteza de segundo nível.

#### Família 2: Métodos Baseados no Espaço de Representação e Distâncias (*Feature Space Geometry*)
Inspecionam a distribuição das características internas extraídas pela penúltima camada $z = \phi(x) \in \mathbb{R}^d$:
- **Distância de Mahalanobis Clássica (Lee et al.):** Modela as classes ID como distribuições Gaussianas com matriz de covariância partilhada $\Sigma$:
  $$M(x) = \min_{c \in \{1,\dots,C\}} (z - \mu_c)^T \Sigma^{-1} (z - \mu_c)$$
- **Análise Marginal e Componentes de Menor Variância (PCA / NPCA):** Demonstra que as direções principais com menor variância explicada no treino (que contêm ruído irrelevante para classificar classes ID) são as que detêm maior sensibilidade discriminatória perante anomalias.
- **Mahalanobis++ (Mueller & Hein):** Projeta as características na hiperesfera unitária através de normalização $\ell_2$: $\tilde{z} = z / \|z\|_2$. Esta transformação simples atenua o viés da magnitude, restabelece a simetria esférica e supera métodos complexos em dezenas de arquiteturas modernas.
- **MM++ (Multilayer Mahalanobis++):** Supera a limitação da penúltima camada (suscetível a colapso neural) monitorizando as quedas de densidade de entropia ao longo das camadas intermédias, unificando representações hierárquicas através de uma matriz de covariância global regularizada por Ledoit-Wolf.
- **Calibração Dinâmica de Covariância em Tempo de Teste:** Atualiza a matriz de covariância a priori em tempo real utilizando os vetores da entrada em análise, confinando a adaptação ao espaço residual para proteger a integridade semântica ID.

#### Família 3: Exposição a Outliers e Regularização Durante o Treino (*Outlier Exposure & Supervised Priors*)
Modelos que intervêm na função de perda durante o treino ou fine-tuning:
- **Outlier Exposure (OE):** Incorpora um conjunto auxiliar massivo de dados não anotados $D_{\text{out}}^{\text{aux}}$, penalizando o modelo através de uma perda de entropia cruzada uniforme:
  $$\mathcal{L}(x, y) = \mathcal{L}_{\text{CE}}(f(x), y) + \lambda \, \mathbb{E}_{\tilde{x} \sim D_{\text{out}}^{\text{aux}}} \left[ \mathcal{H}\left(\mathcal{U}, f(\tilde{x})\right) \right]$$
- **Few-Shot OE com Classe de Rejeição (*Reject Bucket*):** Abordagem viável para contextos com restrição extrema de dados (ex.: imagiologia médica), expondo a rede a uma quantidade diminuta de outliers (ex.: 8 amostras por patologia não vista) mapeados para uma classe de rejeição dedicada, combinando-a com distâncias de cosseno no espaço latente.
- **Otimização de Espaço de Rótulos (*Label Embeddings*):** Substitui as codificações rígidas *one-hot* por representações contínuas estimadas por inferência Bayesiana empírica (via divergência KL ou distância de Mahalanobis), permitindo calibrar incertezas em tarefas geoespaciais e de sensoriamento remoto.

#### Família 4: Métodos Híbridos, Fusão de Incertezas e Teoria de Covariância
- **HUE-OOD (Hybrid Uncertainty-Evidential Dynamics):** Agregação ortogonal de três vertentes: incerteza preditiva estocástica (via Monte Carlo Dropout), incerteza de representação latente (distância de Mahalanobis) e evidência Dirichlet (via *softplus* nos logits), fundidas sem hiperparâmetros por via de ordenação em percentis (*rank-based fusion*).
- **Ensembles de Incerteza (EU + AU):** Combinação de métricas de densidade (incerteza epistémica) e métricas de dispersão probabilística (incerteza aleatória) por regressão logística.
- **Estimadores MMSE de Covariância para Alta Dimensionalidade ($p \gg n$):**
  * *Ledoit-Wolf (LW):* Combinação convexa ótima entre a covariância amostral e a matriz identidade;
  * *Rao-Blackwell Ledoit-Wolf (RBLW):* Condicionamento estatístico suficiente que melhora analiticamente o erro quadrático médio face a LW para dados Gaussianos;
  * *Oracle Approximating Shrinkage (OAS):* Limite analítico iterativo do estimador ótimo não-observável (*clairvoyant oracle*), fundamental para estabilizar matrizes de precisão em espaços latentes de alta dimensão.

#### Família 5: Extensões Cross-Architecture e Domínios Complexos
- **De CNNs a Vision Transformers e Híbridos:** Transição dos métodos concebidos para ResNet/DenseNet para arquiteturas baseadas em auto-atenção (ViT, Swin, DeiT) e convoluções modernas (ConvNeXt, EVA02).
- **Modelos Visuo-Linguísticos e Auto-Supervisionados:** Aplicação de distâncias de representação zero-shot em embeddings congelados de CLIP e DINO, prescindindo de cabeças de classificação supervisionadas.
- **Processamento de Linguagem Natural (PLN):** Deteção de intenções fora do domínio (*out-of-domain intent detection*) em sistemas de diálogo conversacional via Transformers (BERT/RoBERTa) acoplados à distância de Mahalanobis, demonstrando resiliência à compressão por destilação de modelos.

---

### 1.4 Dicionário de Conceitos-Chave (Pedagógico e Rigoroso)

*   **In-Distribution (ID):**  
    Conjunto de dados e distribuição de probabilidade condicional e marginal sobre a qual o modelo foi originalmente otimizado durante o treino supervisionado.
*   **Out-of-Distribution (OOD):**  
    Amostras de teste geradas a partir de um suporte estatístico ou semântico que diverge da distribuição de treino, representando classes ou domínios não observados.
*   **Near-OOD vs. Far-OOD:**  
    Distinção qualitativa de proximidade no espaço de dados. *Near-OOD* caracteriza-se por alta similaridade morfológica e contextual com as classes conhecidas (desafio crítico de deteção); *Far-OOD* reside em domínios completamente alheios ao contexto do modelo.
*   **Incerteza Epistémica (Epistemic Uncertainty - EU):**  
    Incerteza do modelo resultante da escassez ou total ausência de dados de treino na região em análise. É redutível mediante a recolha de novas amostras e constitui o motor central da deteção de OOD.
*   **Incerteza Aleatória (Aleatoric Uncertainty - AU):**  
    Incerteza estocástica irredutível provocada por ruído inerente nos sensores, ambiguidade intrínseca de rótulos ou sobreposição nas fronteiras de separação de classes.
*   **Distância de Mahalanobis (Mahalanobis Distance):**  
    Métrica métrica semi-definida positiva que calcula a distância entre um ponto vetorial e um centróide de classe ponderada pela matriz de covariância:
    $$d_M(z, \mu_c) = \sqrt{(z - \mu_c)^T \Sigma^{-1} (z - \mu_c)}$$
    Invariante a transformações lineares e sensível à dispersão direcional dos dados.
*   **Covariância Amarrada (*Tied Covariance*):**  
    Premissa em Análise Discriminante Linear (LDA) onde todas as classes partilham uma única matriz de covariância comum calculada pela média ponderada das covariâncias intra-classe:
    $$\Sigma_{\text{tied}} = \frac{1}{N} \sum_{c=1}^C \sum_{i \in I_c} (z_i - \mu_c)(z_i - \mu_c)^T$$
*   **Colapso Neural (*Neural Collapse*):**  
    Fenómeno observado no treino até convergência estrita onde os mapas de características da penúltima camada convergem para uma configuração de variância intra-classe nula, alinhando-se com matrizes de pesos ortonormais simplificadas.
*   **Mahalanobis++:**  
    Refinamento metodológico da distância de Mahalanobis que impõe a normalização $\ell_2$ aos vetores de características antes de estimar as médias e a covariância, eliminando distorções de magnitude e aproximando os dados de distribuições esféricas ideais.
*   **MM++ (Multilayer Mahalanobis++):**  
    Estrutura que monitoriza a evolução hierárquica das características ao longo de múltiplas camadas da rede, selecionando nós intermediários através de quedas discretas na densidade de entropia espetral e agregando-os num espaço vetorial unificado regularizado.
*   **Encolhimento de Covariância (*Covariance Shrinkage* - Ledoit-Wolf / OAS):**  
    Formulação analítica de regularização que atenua o mau condicionamento numérico da matriz de covariância amostral em espaços de alta dimensão através de uma média ponderada com um alvo estruturado (matriz identidade):
    $$\hat{\Sigma}_{\text{shrink}} = (1 - \rho) S + \rho \frac{\operatorname{tr}(S)}{d} I$$
*   **Espaço Residual (*Residual Space*):**  
    O subespaço vetorial ortogonal ao subespaço principal gerado pelas direções dominantes das classes ID. As anomalias manifestam-se predominantemente no espaço residual, onde a variância explicada de treino é diminuta.
*   **Outlier Exposure (OE):**  
    Técnica de treino que expõe deliberadamente o classificador a grandes volumes de dados anómalos não rotulados, impondo uma distribuição a posteriori plana ou alocando-os a uma classe de rejeição (*reject bucket*).
*   **FPR95 (False Positive Rate at 95% True Positive Rate):**  
    Métrica padrão da indústria e academia que mede a percentagem de amostras OOD incorretamente classificadas como ID quando o limiar do detetor está calibrado para aceitar com precisão 95% de todas as amostras ID genuínas. Quanto mais baixo for o valor, superior é o sistema.
*   **AUROC (Area Under the Receiver Operating Characteristic Curve):**  
    Área sob a curva entre a taxa de verdadeiros positivos e a taxa de falsos positivos ao longo de todos os limiares de corte possíveis, operando como uma métrica de separabilidade independente de limiar.
*   **AUPR (Area Under the Precision-Recall Curve):**  
    Área sob a curva de Precisão vs. Cobertura (*Recall*), calculada separadamente assumindo amostras ID ou amostras OOD como classe positiva (*AUPR-In* e *AUPR-Out*), crítica para cenários com desequilíbrio acentuado de classes.

---

## 2. Protocolo de Pesquisa Sistemática (Search Strategy)

Para assegurar uma cobertura exaustiva, de elevada qualidade epistemológica e isenta de enviesamento na literatura de deteção de OOD, investigadores e agentes autónomos devem seguir o protocolo sistemático estipulado infra.

### 2.1 Venues Científicos Prioritários
As pesquisas bibliográficas devem priorizar estritamente publicações que tenham passado por rigoroso processo de revisão por pares (*peer-review*) nas seguintes conferências e revistas de topo internacional:

| Categoria | Sigla / Nome do Venue | Qualificação / Foco Científico |
|:---|:---|:---|
| **Conferências de IA & Aprendizagem Automática (Core A\*)** | **NeurIPS** (Neural Information Processing Systems) | Fundamentos de densidade, estimativa de incerteza e distância de Mahalanobis seminal |
| | **ICML** (International Conference on Machine Learning) | Calibração de covariância, dinâmicas de espaço latente e garantias teóricas |
| | **ICLR** (International Conference on Learning Representations) | Representação profunda em OOD, Energy scores e normalização de características |
| | **AAAI** (Association for the Advancement of Artificial Intelligence) | Deteção de anomalias em Transformers, NLP e sistemas de tomada de decisão |
| **Conferências de Visão Computacional & Padrões** | **CVPR** (Computer Vision and Pattern Recognition) | Deteção de OOD em imagens complexas, ReAct, ViM e métodos geométricos |
| | **ICCV** / **ECCV** (Int. Conf. on Computer Vision) | Near-OOD em visão computacional, robustez a corrupções e desvios de textura |
| | **ICPR** (Int. Conf. on Pattern Recognition) | Modelação de normalidade com Gaussianas multivariadas e PCA |
| **Revistas Científicas de Referência** | **IEEE T-PAMI** (Trans. on Pattern Analysis and Machine Intelligence) | Fundamentos teóricos e metodologias unificadas em visão e representação |
| | **JMLR** (Journal of Machine Learning Research) | Fundamentação estatística, limites de erro e propriedades de convergência |
| | **IEEE TSP** (Trans. on Signal Processing) | Estimação de matrizes de covariância, encolhimento de Ledoit-Wolf e OAS |
| | **IEEE T-GRS** (Trans. on Geoscience and Remote Sensing) | Quantificação de incerteza em observação da Terra e Label Embeddings |
| | **Nature / Scientific Reports** | Aplicações médicas rigorosas de OOD (ex.: tomografia de coerência ótica) |
| **Conferências de Processamento de Linguagem Natural** | **ACL** / **EMNLP** / **COLING** | Deteção de intenções fora do domínio (*out-of-domain intents*) e diálogo |
| **Repositórios de Preprints Verificados** | **arXiv** (`cs.LG`, `cs.CV`, `cs.AI`, `stat.ML`) | Descobertas recentes nos últimos 12 a 24 meses (MM++, Mahalanobis++, HUE-OOD) |

---

### 2.2 Equações de Pesquisa Booleanas Recomendadas

As seguintes cadeias de pesquisa encontram-se estruturadas e calibradas para motores de indexação científica (IEEE Xplore, Google Scholar, Semantic Scholar, Scopus, Web of Science, ACM Digital Library e arXiv):

#### Bloco A: Métodos Baseados em Distância, Espaço Latente e Geometria de Características
```text
("out-of-distribution detection" OR "OOD detection" OR "out-of-domain detection") AND ("Mahalanobis distance" OR "feature space" OR "Gaussian discriminant analysis" OR "Mahalanobis++" OR "penultimate layer") AND ("neural collapse" OR "feature normalization" OR "residual space")
```

#### Bloco B: Métodos Post-Hoc Baseados em Logits, Energia e Calibração de Saída
```text
("out-of-distribution" OR "OOD") AND ("maximum softmax probability" OR "temperature scaling" OR "ODIN" OR "energy-based" OR "activation rectification" OR "ReAct") AND ("false positive rate" OR "AUROC" OR "trustworthy AI")
```

#### Bloco C: Estimação de Incerteza Híbrida, Outlier Exposure e Teoria de Covariância
```text
("out-of-distribution" OR "anomaly detection") AND ("epistemic uncertainty" OR "aleatoric uncertainty" OR "Dirichlet" OR "evidential deep learning" OR "outlier exposure" OR "reject bucket") AND ("Ledoit-Wolf" OR "RBLW" OR "Oracle Approximating Shrinkage" OR "covariance estimation")
```

#### Bloco D: OOD em Modelos Modernos (Transformers, Visuo-Linguísticos e Aplicações Críticas)
```text
("out-of-distribution detection" OR "near-OOD") AND ("Vision Transformer" OR "ViT" OR "Swin" OR "ConvNeXt" OR "CLIP" OR "foundation models") AND ("medical imaging" OR "retinal OCT" OR "remote sensing" OR "autonomous systems")
```

---

### 2.3 Janela Temporal de Análise

A recolha documental deve cobrir sistematicamente dois períodos complementares:

1. **Janela Seminal e Fundacional (2010 – 2020):**
   - **Objetivo:** Compreender a formulação matemática inicial e as técnicas clássicas de deteção de anomalias.
   - **Marcos Fundacionais:**
     - Chen et al. (IEEE TSP 2010 - *Shrinkage for MMSE Covariance Estimation*): Formalização dos estimadores RBLW e OAS, pilares para matrizes de covariância bem condicionadas em alta dimensão.
     - Hendrycks & Gimpel (ICLR 2017 - *A Baseline for Detecting Out-of-Distribution Inputs*): Introdução do baseline canónico de probabilidade máxima softmax (MSP).
     - Liang, Li & Srikant (ICLR 2018 - *ODIN*): Proposta de escalonamento de temperatura e perturbações guiadas por gradiente.
     - Lee et al. (NeurIPS 2018 - *A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks*): Obra seminal que estabelece a Análise Discriminante Gaussiana e a distância de Mahalanobis em redes neurais.
     - Hendrycks, Mazeika & Dietterich (ICLR 2019 - *Deep Anomaly Detection with Outlier Exposure*): Formalização do paradigma de treino com dados OOD auxiliares.
     - Kamoi & Kobayashi (arXiv 2020 - *Why is the Mahalanobis Distance Effective for Anomaly Detection?*): Descoberta teórica de que a distância de Mahalanobis opera primariamente através das componentes principais de menor variância (PCA).
     - Xu et al. (COLING 2020 - *A deep generative distance-based classifier for out-of-domain detection*): Aplicação de GDA e Mahalanobis em sistemas de diálogo.

2. **Janela de Avanços Recentes e Estado da Arte (Últimos 3 a 5 anos / 2021 – 2026):**
   - **Objetivo:** Mapear a transição para métodos sem parâmetros, normalização esférica, análise multi-camada e validação em modelos baseados em atenção (Transformers) e dados complexos.
   - **Marcos Recentes Relevantes:**
     - Liu et al. (NeurIPS 2020) & Sun et al. (NeurIPS 2021 - *ReAct*): Métodos baseados em energia e truncamento de ativações.
     - Podolskiy et al. (AAAI 2021 - *Revisiting Mahalanobis Distance for Transformer-based Out-of-Domain Detection*): Generalização do método de Mahalanobis para embeddings contextuais de Transformers com robustez à destilação.
     - Rippel et al. (ICPR 2021 - *Modeling the Distribution of Normal Data in Pre-Trained Deep Features*): Modelação de distribuições Gaussianas multivariadas e validação empírica das componentes residuais em MVTec AD.
     - Kaur et al. (ICML Workshop 2021 - *Detecting OODs as Datapoints with High Uncertainty*): Desacoplamento e fusão de incerteza epistémica e aleatória.
     - Araújo et al. (Scientific Reports 2023 - *Few-shot out-of-distribution detection for automated screening in retinal OCT images*): Validação clínica de Few-Shot Outlier Exposure e distância de Cosseno em patologias oculares complexas.
     - Schweden et al. (IEEE TGRS 2025 - *Can Uncertainty Quantification Benefit From Label Embeddings?*): Uso de espaços de Dirichlet e formulações empíricas de Bayes para OOD em sensoriamento remoto.
     - Mueller & Hein (arXiv 2025 - *Mahalanobis++: Improving OOD Detection via Feature Normalization*): Correção teórica e empírica do viés de norma através de normalização $\ell_2$, estabelecendo novo estado da arte em 44 arquiteturas.
     - Guo et al. (ICML 2025 - *Improving out-of-distribution detection via dynamic covariance calibration*): Calibração de covariância no espaço residual durante a inferência.
     - Hossain et al. (arXiv 2026 - *MM++: Unsupervised Scale-Invariant Multilayer OOD Detection*): Seleção adaptativa de camadas intermediárias por decaimento de entropia e encolhimento de Ledoit-Wolf.
     - Nguyen (Research Square 2026 - *HUE-OOD: Hybrid Uncertainty--Evidential Dynamics*): Fusão pós-processamento de incerteza preditiva estocástica, representacional e evidencial de Dirichlet.

---

## 3. Critérios de Elegibilidade (Inclusão e Exclusão)

Para blindar o repositório contra alucinações metodológicas e garantir relevância académica consistente, todo e qualquer artigo candidato deve ser avaliado contra o seguinte protocolo de triagem:

```
                            Artigo Identificado na Pesquisa
                                           |
                    +----------------------+----------------------+
                    |                                             |
                    v                                             v
            Critérios de Inclusão (+)                     Critérios de Exclusão (-)
        - Métricas Canónicas (FPR95, AUROC)           - Apenas acurácia de classificação ID
        - Baselines Consolidados (MSP, Mahalanobis)   - Sem separação estrita entre ID e OOD
        - Transparência Matemática da Pontuação       - Truques de sintonia com dados de teste
        - Benchmarks Rigorosos (ImageNet/Near-OOD)    - Métodos dependentes de rotulagem OOD
        - Reprodutibilidade e Detalhes de Hardware    - Artigos opinativos ou predatórios
                    |                                             |
                    v                                             v
             ACEITE NA TABELA                             REJEITADO
```

### 3.1 Critérios de Inclusão (+)
O artigo deve satisfazer cumulativamente pelo menos **três** dos seguintes requisitos fundamentais:
1. **Reporte Obrigatório de Métricas Canónicas de Deteção:** Documentação explícita de métricas padronizadas: Taxa de Falsos Positivos a 95% de Sensibilidade (**FPR95**), Área sob a Curva ROC (**AUROC**) e Área sob a Curva Precision-Recall (**AUPR-In / AUPR-Out**). Artigos que reportam unicamente acuidade de classificação clássica não são elegíveis.
2. **Comparação com *Baselines* Canónicos:** Avaliação comparativa obrigatória contra métodos estabelecidos de referência, incluindo obrigatoriamente o baseline de máxima probabilidade (MSP), ODIN, Energy Score e/ou a formulação clássica de Mahalanobis.
3. **Formalização Matemática Clara da Função de Pontuação $S(x)$:** Apresentação inequívoca das equações analíticas que determinam o cálculo do *score* (ex.: projeção esférica, encolhimento de matrizes de covariância, formulação de perturbação de entrada ou integrais evidenciais).
4. **Isolamento Estrito dos Dados de Validação e Teste:** Garantia metodológica de que nenhum dado pertencente aos conjuntos OOD de teste foi utilizado para ajustar parâmetros livres, limiares de decisão ou hiperparâmetros de calibração.
5. **Avaliação em Benchmarks Padronizados e Cenários Desafiantes:** Utilização de conjuntos de dados reconhecidos pela comunidade científica internacional (ex.: CIFAR-10/100 como ID versus SVHN/Textures/LSUN/Places365 como OOD; ImageNet-1K versus ImageNet-O, OpenImage-O, NINCO; ou conjuntos médicos/industriais canónicos como MVTec AD e bases públicas de OCT retinal).

### 3.2 Critérios de Exclusão (-)
Devem ser sumariamente desconsiderados os trabalhos que incorram em qualquer uma das seguintes falhas:
1. **Omissão de Métricas de Deteção de Anomalias:** Trabalhos que avaliam modelos perante entradas ruidosas mas limitam a avaliação à degradação da acurácia ID, sem calcular curvas ROC ou taxas de rejeição.
2. **Contaminação de Domínio (*Data Leakage*):** Métodos que utilizam as classes OOD de teste durante a fase de treino ou que realizam calibração supervisionada de hiperparâmetros diretamente sobre os alvos a detetar.
3. **Sobrecarga Computacional Intratável sem Ganho Proporcional:** Métodos que exigem amostragem por difusão inversa complexa ou dezenas de milhares de passagens estocásticas para classificar uma única imagem, sem justificação teórica de eficiência.
4. **Artigos Opinativos, Editoriais ou sem Revisão por Pares Idónea:** Relatórios preliminares de fóruns informais, postagens de blogues, sínteses não supervisionadas e artigos oriundos de revistas sem comitê editorial reconhecido.

---

## 4. Roteiro de Extração e Síntese Analítica

Ao analisar cada publicação validada na triagem de elegibilidade, o curador ou agente de IA deve preencher exaustivamente a seguinte **Checklist de 5 Pontos**:

```
+-----------------------------------------------------------------------------------------+
|                    CHECKLIST ANALÍTICA DE EXTRAÇÃO DE ARTIGOS DE OOD                    |
+-----------------------------------------------------------------------------------------+
| 1. Problema e Motivação                                                                 |
|    - Que lacuna metodológica aborda (sobreconfiança, violação gaussiana, colapso)?      |
|    - Em que regime atua (Post-hoc / Zero Retraining, Outlier Exposure, Few-Shot)?        |
+-----------------------------------------------------------------------------------------+
| 2. Inovação Metodológica / Algorítmica                                                  |
|    - Em que espaço atua o score (logits, penúltima camada, multicamada, Dirichlet)?     |
|    - Que formulação matemática rege a função de pontuação S(x) e a covariância?         |
|    - Como mitiga o viés de magnitude da norma ou a maldição da dimensionalidade?        |
+-----------------------------------------------------------------------------------------+
| 3. Datasets e Protocolo de Avaliação                                                    |
|    - Quais os benchmarks In-Distribution e Out-of-Distribution avaliados?               |
|    - Que arquiteturas dorsais foram testadas (ResNet, DenseNet, ViT, Swin, ConvNeXt)?   |
+-----------------------------------------------------------------------------------------+
| 4. Resultados Empíricos e Trade-offs                                                    |
|    - Quais os valores numéricos exatos de FPR95, AUROC e AUPR face aos baselines?       |
|    - Qual o impacto no tempo de inferência, estabilidade de memória e complexidade?     |
+-----------------------------------------------------------------------------------------+
| 5. Limitações e Desafios em Aberto (Open Challenges)                                   |
|    - Sob que cenários o método falha (near-OOD com alta sobreposição, desequilíbrio)?   |
|    - Quais as dependências estruturais não resolvidas (inversão matricial, calibração)? |
+-----------------------------------------------------------------------------------------+
```

### Detalhe Operacional de Cada Ponto da Checklist:

1. **Problema e Motivação:**  
   Identificar a premissa teórica contestada ou a limitação prática atacada (ex.: o fracasso da distância de Mahalanobis devido a normas heterogéneas, o custo computacional de re-treinar redes densas, a dificuldade de detetar patologias invisíveis em imagiologia clínica ou a perda de informação discriminativa na penúltima camada devido a colapso neural).
2. **Inovação Metodológica/Algorítmica:**  
   Registar as expressões matemáticas centrais: a definição da função de distância $d(z, \mu)$, a regularização da matriz de covariância ($\Sigma_{\text{shrink}}$ via Ledoit-Wolf ou OAS), mecanismos de projeção na hiperesfera ($\ell_2$-normalization), fusão de *ranks* de incerteza ou estratégias de perturbação de entrada.
3. **Datasets e Protocolo de Avaliação:**  
   Catalogar com precisão os pares de conjuntos ID $\to$ OOD utilizados (ex.: CIFAR-10 $\to$ SVHN/LSUN, ImageNet-1K $\to$ ImageNet-O/NINCO/Textures, MVTec AD, So2Sat LCZ42, bases de OCT macular) e os modelos avaliados (DenseNet-121, ResNet-50, EfficientNet, ViT-B/16, CLIP ViT-L/14).
4. **Resultados Empíricos e Trade-offs:**  
   Extrair dados numéricos rigorosos: redução de pontos percentuais no FPR95 (ex.: "redução de 7,6% no FPR95 em ImageNet com 44 arquiteturas"), ganhos de AUROC, tempo computacional de inferência em GPU e consumo de memória associado à inversão ou armazenamento de matrizes de precisão.
5. **Limitações e Desafios em Aberto:**  
   Anotar com transparência as fragilidades apontadas pelos autores: sensibilidade a amostras ID com ruído de anotação, instabilidade numérica da inversão matricial em espaços de características de altíssima dimensão ($d > 2048$), dificuldade em separar classes Near-OOD com semântica quase indistinguível e comportamento perante modelos sobredotados (*Vision-Language Models*).

---

## 5. Padrão de Formatação para a Tabela de Literatura (`README.md`)

O ficheiro central `README.md` localizado no mesmo diretório organiza o estado da arte do domínio numa tabela estruturada de 6 colunas. Qualquer atualização subsequente deve cumprir escrupulosamente a formatação padronizada descrita abaixo.

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
    Apresentação do artigo, contextualização do desafio de deteção de Out-of-Distribution, anomalias ou estimativa de incerteza em redes neuronais profundas, explicitando com clareza o problema estrutural, teórico ou prático que os autores se propõem resolver.
*   **Parágrafo 2 — Mecanismo Proposto e Inovação Metodológica:**  
    Descrição minuciosa do algoritmo ou metodologia proposta (natureza do cálculo da pontuação de confiança, espaço em que atua — representação, logits ou camadas intermediárias —, formulações de regularização de covariância, projeção em hiperesfera ou modelos de quantificação de incerteza).
*   **Parágrafo 3 — Validação Experimental e Métricas Numéricas:**  
    Exposição sintética do enquadramento experimental (conjuntos de dados ID e OOD, arquiteturas de redes testadas) e quantificação rigorosa dos resultados alcançados (valores exatos de AUROC, FPR95, AUPR, redução de variância e comparações com baselines SOTA).
*   **Parágrafo 4 — Significado Teórico, Limitações e Desafios:**  
    Avaliação do impacto científico da contribuição para a fiabilidade e segurança de sistemas de Inteligência Artificial, acompanhada pela identificação franca das limitações do método, premissas teóricas subjacentes e linhas de investigação futura sugeridas.

---

### 5.3 Exemplo Canónico de Registo na Tabela

Apresenta-se de seguida o modelo de preenchimento integral em Markdown baseado num dos trabalhos de maior impacto do repositório:

```markdown
| Mahalanobis++: Improving OOD Detection via Feature Normalization | **Autores:** Mueller & Hein<br><br>**Data de publicação:** 2025<br><br>**Livro/Journal:** arXiv preprint<br><br>**Volume:** arXiv:2505.18032<br><br>**Páginas:** 1–18<br><br>**DOI:** https://doi.org/10.48550/arXiv.2505.18032 | "While post-hoc methods based on the Mahalanobis distance applied to pre-logit features are among the most effective for ImageNet-scale OOD detection, their performance varies significantly across models. We connect this inconsistency to strong variations in feature norms, indicating severe violations of the Gaussian assumption underlying the Mahalanobis distance estimation."<br><br>"We show that simple $\ell_2$-normalization of the features mitigates this problem effectively, aligning better with the premise of normally distributed data with shared covariance matrix."<br><br>"Extensive experiments on 44 models across diverse architectures and pretraining schemes show that $\ell_2$-normalization improves the conventional Mahalanobis distance-based approaches significantly and consistently, and outperforms other recently proposed OOD detection methods." | "We showed that the frequently occurring failure cases of the Mahalanobis distance as an OOD detection method are related to violations of the method’s basic assumptions. We showed that the feature norms vary much stronger than expected under a Gaussian model, that the feature distributions are strongly heavy-tailed and that feature norms correlate with the Mahalanobis score - irrespective of whether a sample is ID or OOD."<br><br>"We introduced Mahalanobis++, a simple remedy consisting of $\ell_2$ normalization that effectively mitigates those problems. In particular, the resulting feature distributions are more aligned with a normal distribution, less heavy-tailed, and the class variances are more similar, leading to improved OOD detection results across a wide range of models."<br><br>"Mahalanobis++ outperforms the conventional Mahalanobis distance in 41/44 cases, rendering it clearly the most effective method across models. It outperforms the previously best baseline ViM by 7 FPR points on average on the OpenOOD datasets, and is the best method for 4 of the 5 top models." | Este artigo analisa as causas fundamentais do comportamento inconsistente dos métodos de deteção de amostras fora de distribuição (*Out-of-Distribution* - OOD) baseados na distância de Mahalanobis calculada sobre os vetores de características da penúltima camada (*pre-logit features*). Apesar da sua eficácia consolidada em problemas à escala do ImageNet, o desempenho desta família de detetores oscila consideravelmente consoante a arquitetura da rede neuronal e o esquema de pré-treino utilizado.<br><br>Os autores demonstram que esta volatilidade se deve a severas violações das premissas estatísticas de base: as normas euclidianas das características apresentam flutuações desproporcionais entre e dentro das classes, contradizendo a assunção de uma distribuição Gaussiana multivariada com matriz de covariância partilhada (*tied covariance*). Como resultado, amostras OOD com magnitude de ativação reduzida são incorretamente classificadas com baixo risco de anomalia.<br><br>Para mitigar esta vulnerabilidade, a publicação introduz o Mahalanobis++, uma modificação analítica que impõe a normalização $\ell_2$ aos vetores de características antes da estimação dos centróides de classe, da matriz de covariância e do cálculo das distâncias. Ao restringir o espaço vetorial à superfície da hiperesfera unitária e reter estritamente a orientação direcional, o método harmoniza a variância das classes e aproxima as características de uma distribuição esférica controlada.<br><br>A validação experimental em 44 arquiteturas neuronais distintas (englobando ConvNeXt, Swin Transformers, Vision Transformers clássicos e ResNets) nos conjuntos de dados ImageNet e CIFAR-100 comprovou a superioridade do Mahalanobis++, superando a formulação tradicional em 41 dos 44 modelos (com redução média de 7,6% no FPR95 no ImageNet) e ultrapassando baselines modernos como ViM, KNN, ReAct e Energy Score, afirmando-se como um novo padrão para detetores *post-hoc*. | Mueller M, Hein M. Mahalanobis++: Improving OOD Detection via Feature Normalization. arXiv preprint arXiv:2505.18032; 2025. https://doi.org/10.48550/arXiv.2505.18032. |
```

---

## 6. Guardrails de Qualidade & Anti-Alucinação

Para assegurar integridade científica, rigor metodológico e repetibilidade inatacável nas pesquisas de deteção de Out-of-Distribution, agentes autónomos e investigadores humanos devem obedecer estritamente aos seguintes guardrails operacionais:

### 6.1 Validação Funcional de DOIs e Fontes Indexadas
- Todo o identificador de objeto digital (**DOI**) deve ser verificado em bases ativas com o prefixo oficial `https://doi.org/...`. Não é admitida a inclusão de ligações mortas, domínios fictícios ou aproximações inventadas.
- Metadados como volume, número de fascículo, ano e páginas devem ser conferidos por verificação direta em repositórios fidedignos (IEEE Xplore, DBLP, ACM DL, SpringerLink, PubMed ou Google Scholar).

### 6.2 Proibição Estrita de Fabricação e Arredondamento de Métricas
- Valores de **FPR95**, **AUROC**, **AUPR** e reduções percentuais de erro reportados devem corresponder com exatidão matemática aos quadros e tabelas experimentais da publicação fonte.
- É categoricamente proibido extrapolar resultados (exemplo: se um algoritmo atinge 95.8% de AUROC no MVTec AD em EfficientNet, é vedado afirmar que atinge "cerca de 96% em qualquer rede ou dataset de anomalias").

### 6.3 Preservação Literal das Citações Textuais
- Os blocos correspondentes ao **Abstract** e à **Conclusion** na tabela de literatura devem constituir **transcrições literais** e integrais de passagens da publicação em língua inglesa, mantidas rigorosamente entre aspas duplas.
- É interdita qualquer reformulação, resumo automático ou tradução no interior desses campos delimitados.

### 6.4 Neutralidade Epistemológica e Sobriedade Científica
- As sínteses e revisões analíticas devem pautar-se por uma linguagem académica impessoal, objetiva e sóbria.
- Estão terminantemente banidos adjetivos propagandísticos e vocabulário sensacionalista (como "método revolucionário", "algoritmo infalível", "precisão mágica"). As contribuições devem ser discutidas à luz das suas premissas teóricas, custo computacional e limitações práticas.
