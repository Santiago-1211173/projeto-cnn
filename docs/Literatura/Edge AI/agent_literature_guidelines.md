# Protocolo de Curadoria e Navegação na Literatura: Edge AI, Sistemas Distribuídos & Alocação de Recursos

> **Ficheiro Central Associado:** `README.md` (no mesmo diretório)  
> **Natureza do Documento:** Guia metodológico de recolha, análise crítica e referenciação científica para agentes autónomos e investigadores.  
> **Domínio Científico:** Inteligência Artificial / Aprendizagem Automática / *Edge Computing*, Redes Neuronais Distribuídas e Alocação Inteligente de Recursos.

---

## 1. Enquadramento Teórico & Âmbito do Tópico

### 1.1 Definição Canónica na Literatura Internacional de IA
A área de **Edge AI (Inteligência Artificial na Periferia)**, operando em convergência com **Sistemas Distribuídos** e a **Alocação Inteligente de Recursos**, estuda a transferência e a execução eficiente de algoritmos de Aprendizagem Profunda (*Deep Learning* — DL) e tomada de decisão para a periferia da rede (*network edge*), nas imediações dos sensores e dispositivos finais onde os dados são gerados.

Na literatura internacional de referência (e.g., Wang et al., IEEE Communications Surveys & Tutorials, 2020), este domínio estrutura-se em torno de dois pilares complementares e mutuamente benéficos:

1. **Inteligência na Borda (*Edge Intelligence* ou *AI for Edge*):**  
   Focaliza o desenvolvimento, compressão, partição e aceleração de modelos de IA para que possam ser executados com elevada precisão e fidelidade diretamente em plataformas de computação de borda (*edge nodes*, dispositivos móveis, sensores IoT, estações base e sistemas embebidos), respeitando severas restrições operacionais de tempo de resposta, pegada de memória, dissipação térmica e autonomia de bateria.

2. **Borda Inteligente (*Intelligent Edge* ou *Edge for AI*):**  
   Focaliza a aplicação de algoritmos avançados de Inteligência Artificial — designadamente Aprendizagem por Reforço (*Reinforcement Learning* — RL), *Deep Reinforcement Learning* (DRL) e Bandidos Contextuais (*Contextual Bandits*) — para a orquestração autónoma, manutenção e auto-otimização dinâmica da própria infraestrutura distribuída (gestão preditiva de *caching*, atribuição de tarefas a nós heterogéneos, compressão de tráfego de dados e balanceamento de carga).

No centro desta disciplina situa-se o **Continuum Fim-Borda-Nuvem (*End-Edge-Cloud Continuum*)**, um ecossistema cooperativo e hierarquizado em três níveis principais:

```
+-----------------------------------------------------------------------------+
|                          NUVEM CENTRAL (*CLOUD*)                             |
|  - Elevada capacidade computacional (Clusters de GPUs de alto débito)        |
|  - Treino de modelos fundacionais, arquivo global e inferência pesada       |
+-----------------------------------------------------------------------------+
                                      ^
                                      | Rede WAN / Internet (Alta latência)
                                      v
+-----------------------------------------------------------------------------+
|                         NÓS DE BORDA (*EDGE NODES*)                          |
|  - Gateways locais, estações base (5G/6G), servidores de proximidade (MEC)   |
|  - Co-inferência, filtragem de dados, DRL para caching e partição de redes  |
+-----------------------------------------------------------------------------+
                                      ^
                                      | Rede Local / RAN / BLE (Baixa latência)
                                      v
+-----------------------------------------------------------------------------+
|                      DISPOSITIVOS FINAIS (*END DEVICES*)                     |
|  - Sensores IoT, smartphones, aceleradores neuromórficos (SNNs), robôs       |
|  - Inferência em tempo real, saídas antecipadas (EEoI) e pré-processamento  |
+-----------------------------------------------------------------------------+
```

A alocação de recursos traduz-se formalmente num problema de **otimização multiobjetivo** sujeito a restrições rígidas de tempo de execução, consumo energético e capacidade de memória, resolvido tanto por métodos analíticos de Pareto (e.g., $\epsilon$-constraint) como por formulações sequenciais modeladas como Processos de Decisão de Markov (MDP).

---

### 1.2 Problema Fundamental que Aborda

O paradigma clássico de computação em nuvem centralizada (*Cloud-Centric AI*), no qual todos os dados brutos recolhidos são transmitidos através da Internet para serem processados em centros de dados remotos, tornou-se impraticável para a maioria das aplicações contemporâneas devido a quatro limitações estruturais:

1. **Latência Inadmissível e Jitter de Comunicação:**  
   Em sistemas críticos de tempo real (veículos autónomos, robótica cooperativa, cirurgia tele-assistida, monitorização industrial), o tempo de ida e volta (*Round-Trip Time* — RTT) introduzido pela transmissão na WAN é incompatível com orçamentos de latência estritos (inferiores a dezenas de milissegundos).

2. **Estrangulamento da Largura de Banda de Rede:**  
   A proliferação de milhares de milhões de sensores e câmaras de vídeo de alta definição gera um débito de dados massivo que satura a largura de banda das redes de telecomunicações se todos os fluxos forem enviados continuamente em bruto.

3. **Privacidade, Confidencialidade e Soberania dos Dados:**  
   Dados biométricos, telemetria médica e fluxos de áudio e vídeo pessoal acarretam sérias implicações legais e éticas (e.g., RGPD/GDPR, HIPAA). O processamento local na borda assegura que informações sensíveis nunca abandonam a esfera de controlo físico do utilizador.

4. **Resiliência e Continuidade Operacional em Ambientes Desconectados:**  
   Dispositivos que operam em ambientes remotos ou hostis (agricultura de precisão, veículos em túneis, missões espaciais ou marítimas) necessitam de autonomia decisória total perante falhas temporárias ou ausência permanente de conetividade à rede.

#### A Falência do Paradigma Estático "Otimizar-e-Congelar" (*Optimize-then-Freeze*)
Historicamente, as tentativas de contornar estas limitações basearam-se na compressão e quantização *offline* de redes neuronais, seguida do seu congelamento estático no dispositivo (*deploy-and-freeze*). A literatura científica recente (e.g., Pittorino & Roveri, 2026) demonstra categoricamente que esta abordagem estática falha inevitavelmente no mundo real:

- **Não-Estacionariedade e Deriva de Distribuição (*Distribution Drift*):** As distribuições estatísticas dos dados de entrada mudam continuamente ao longo do tempo (*covariate shift* e *concept drift*), conduzindo modelos congelados à perda progressiva de calibração estatística e colapso de acurácia preditiva.
- **Flutuação Temporal de Orçamentos Físicos (*Time-Varying Hardware Budgets*):** No hardware embebido, os recursos operacionais não são constantes. A tensão da bateria diminui, a temperatura do processador sobe (provocando *thermal throttling* ou estrangulamento térmico), e a largura de banda sem fios oscila dinamicamente. Um sistema fixo de inferência resulta inevitavelmente na violação de orçamentos de energia/latência ou no bloqueio do sistema.
- **Necessidade Incontornável de Adaptatividade:** Conclui-se que o verdadeiro *Edge AI* é forçosamente **adaptativo** (*Adaptive Edge AI*), exigindo mecanismos dinâmicos de reconfiguração de computação em tempo de execução, co-inferência hierárquica e aprendizagem contínua no próprio dispositivo.

---

### 1.3 Taxonomia e Principais Famílias de Abordagens no Estado da Arte

A literatura científica contemporânea de Edge AI e Alocação de Recursos organiza-se em cinco grandes famílias arquiteturais e metodológicas:

```
                            Taxonomia de Edge AI & Alocação de Recursos
                                                 |
         +--------------------+------------------+------------------+--------------------+
         |                    |                  |                  |                    |
   1. Compressão        2. Co-Inferência    3. Computação      4. Alocação        5. Edge AI
      Extrema e            e Particionamento   Neuromórfica       Autónoma via       Adaptativo
      Modelos Baixo Bit    End-Edge-Cloud      Bio-Inspirada      RL e DRL           (Lente ASE)
         |                    |                  |                  |                    |
   - PTQ / QAT          - Split Computing   - Spiking Neural   - DDPG / Wolper-   - ASE Lens
   - Ternário 1.58-bit  - LMOS (Pareto)       Networks (SNNs)    tinger Caching   - Auto-destilação
   - Attention Sinks    - Early Exit (EEoI) - Leaky Integrate- - Catcher+ (CBS)     On-Device
   - Memória Episódica  - MEANet              and-Fire (LIF)   - RLQ (Celery /    - Co-adaptação
     Latente (BitMar)     (Fácil/Difícil)   - ECC-SNN Híbrido    Distributed Q)     System-1/2
```

#### Família 1: Compressão Extrema, Quantização de Baixo Bit e Mecanismos de Memória
- **Princípio:** Reduzir a representação numérica dos pesos e ativações de redes neuronais (de representações de virgula flutuante FP32/FP16 para inteiros INT8, INT4 e até redes ternárias extremas a 1.58-bit com valores $\{-1, 0, 1\}$ no estilo BitNet).
- **Mecanismos Complementares:**
  - *Attention Sinks & Sliding Windows:* Para modelos baseados em Transformers de atenção na borda, a retenção de tokens iniciais fixos (*sinks*) combinada com janelas deslizantes limita o crescimento quadrático do consumo de memória VRAM em fluxos sequenciais contínuos.
  - *Memória Episódica Latente Externa (ex.: BitMar, 2025):* Incorporação de memórias baseadas em *slots* chave-valor de capacidade fixa inspiradas na cognição humana. Permitem que modelos ultra-comprimidos (e.g., 14M parâmetros) preservem o contexto histórico de fluxos multimodais longos com velocidade até 7,5 vezes superior e redução de ~79% no consumo de energia.

#### Família 2: Co-Inferência Colaborativa e Particionamento de Modelos (*Split Computing*)
- **Princípio:** Em vez de executar o modelo integralmente no nó periférico ou na nuvem, divide-se a arquitetura neuronal ao longo de pontos de corte estratégicos (*split points*).
- **Abordagens Canónicas:**
  - *Otimização Multiobjetivo de Pareto (ex.: LMOS, 2022):* Formula a seleção do ponto de corte como um problema de otimização $\epsilon$-constraint, minimizando a latência ponta-a-ponta (cálculo na borda + tempo de transmissão de dados intermediários comprimidos + cálculo na nuvem) enquanto maximiza o aproveitamento da memória local do dispositivo (e.g., Raspberry Pi).
  - *Saídas Antecipadas Baseadas em Incerteza (ex.: MEANet, 2021):* Implementa a Saída Antecipada da Inferência (*Early Exit of Inference* — EEoI). Instâncias fáceis terminam o processamento no bloco principal do dispositivo periférico; casos ambíguos ativam blocos adaptativos locais; e amostras de complexidade extrema são descarregadas para a nuvem através de um canal condicional orientado por métricas de entropia.

#### Família 3: Computação Neuromórfica e Redes Bio-Inspiradas na Borda (SNNs)
- **Princípio:** Utilização de Redes Neuronais de Impulsos (*Spiking Neural Networks* — SNNs), a terceira geração de modelos neuronais bio-inspirados baseados em neurónios *Leaky Integrate-and-Fire* (LIF). Os neurónios processam e transmitem informação exclusivamente através de impulsos discretos assíncronos (*spikes*) no domínio temporal.
- **Sistemas Colaborativos Híbridos (ex.: ECC-SNN, 2025):**
  - Associação de uma SNN ultraleve na borda (executada em hardware neuromórfico ou processador de baixo consumo) com uma Rede Neuronal Artificial profunda (ANN clássica) na nuvem.
  - Alinhamento de representações de vírgula flutuante e trens de impulsos (*spike trains*) via destilação de conhecimento, assegurando simultaneamente redução de latência (~39%) e poupanças massivas de energia (~79%).

#### Família 4: Alocação Autónoma de Recursos e Gestão de Infraestrutura via DRL
- **Princípio:** Eliminação de heurísticas estáticas e rígidas de sistemas distribuídos (e.g., LRU, LFU, FIFO, Round-Robin) em favor de agentes autónomos que aprendem políticas ótimas de escalonamento e armazenamento através de interação contínua com o ambiente.
- **Aplicações Fundamentais:**
  - *Content Caching na Borda (ex.: Zhong et al., 2018; Catcher+, 2024; RLCache, 2019):* Agentes baseados em arquiteturas Wolpertinger/DDPG ou meta-aprendizagem de seleção de políticas que aprendem a prever o valor de retenção de conteúdos e tempos de expiração (TTL), maximizando a taxa de acerto (*cache hit rate*) e reduzindo até ~42% do tráfego de dados para o armazenamento secundário.
  - *Escalonamento em Filas Distribuídas (ex.: RLQ, 2023):* Modelação de filas assíncronas heterogéneas (como Celery) como MDPs resolvidos por LinUCB e DoubleDQN, gerindo eficientemente o problema de recompensas diferidas (*delayed feedback*) e reduzindo tempos de espera até 20 vezes sem conhecimento prévio dos requisitos das tarefas.

#### Família 5: Edge AI Adaptativo e Operação em Longo Curso (Enquadramento ASE)
- **Princípio:** Formalização do ciclo de vida de modelos periféricos sob uma ótica dinâmica em que o sistema reconfigura a sua arquitetura, estado interno e regime de trabalho face a desvios no ambiente.
- **A Lente Agent-System-Environment (ASE, 2026):**
  - Mapeia o fluxo observável $(o_t, c_t)$, o estado interno em evolução $s_t$, e as intervenções admissíveis $a_t$ sob restrições orçamentais mutáveis.
  - Propõe mecanismos de auto-destilação local (*on-device self-distillation*) sem armazenamento de exemplares brutos para mitigar o esquecimento catastrófico (*catastrophic forgetting*) durante atualizações incrementais locais.

---

### 1.4 Dicionário de Conceitos-Chave (Pedagógico e Rigoroso)

*   **Edge AI (Inteligência Artificial na Periferia):**  
    Execução de algoritmos de aprendizagem profunda e extração de padrões em dispositivos de hardware locais próximos da origem física dos dados, mitigando dependências críticas da nuvem.
*   **End-Edge-Cloud Continuum (Continuum Fim-Borda-Nuvem):**  
    Arquitetura de computação distribuída integrada e sem descontinuidades, que harmoniza o processamento entre dispositivos finais embebidos, servidores de proximidade na borda e centros de dados centralizados na nuvem.
*   **Edge Intelligence vs. Intelligent Edge:**  
    Distinção canónica: *Edge Intelligence* refere-se à habilitação de modelos de IA na infraestrutura periférica; *Intelligent Edge* refere-se ao uso de agentes inteligentes para orquestrar e otimizar os próprios recursos da borda.
*   **Particionamento de Modelos (*Model Splitting / Partitioning*):**  
    Divisão de uma arquitetura neuronal em dois ou mais blocos sequenciais através de um ponto de corte (*split point*), distribuindo o processamento entre hardware local e remoto.
*   **Saída Antecipada da Inferência (*Early Exit of Inference* — EEoI):**  
    Mecanismo arquitetural que acopla ramificações classificadoras laterais em camadas intermédias de uma rede profunda, permitindo que instâncias com elevada certeza preditiva terminem a inferência precocemente, poupando energia e tempo de ciclo.
*   **Quantização de Baixo Bit & Redes Ternárias a 1.58-bit:**  
    Mapeamento de pesos contínuos em níveis discretos reduzidos. No regime de 1.58-bit (ternário), os pesos assumem apenas valores no conjunto $\{-1, 0, 1\}$, substituindo multiplicações dispendiosas de vírgula flutuante por adições e subtrações inteiras.
*   **Sinks de Atenção (*Attention Sinks*):**  
    Propriedade empírica e técnica de projeto em Transformers onde um número pequeno de tokens iniciais retém frações massivas da atenção latente, estabilizando o mecanismo de atenção quando associado a janelas deslizantes de memória finita.
*   **Memória Episódica Latente Externa (*External Episodic Latent Memory*):**  
    Módulo de retenção associativa de vetores latentes que atua como memória contextual persistente de curto/médio prazo, desonerando o núcleo preditivo de reprocessar sequências temporais extensas.
*   **Redes Neuronais de Impulsos (*Spiking Neural Networks* — SNNs) & Modelo LIF:**  
    Modelos neuronais bio-inspirados em que a comunicação ocorre por disparos binários instantâneos (*spikes*). O neurónio *Leaky Integrate-and-Fire* (LIF) acumula potencial de membrana através de uma fuga contínua até atingir um limiar de disparo, reiniciando o seu estado após emitir o pulso.
*   **Co-Inferência Guiada por Incerteza / Entropia Normalizada:**  
    Estratégia de despacho colaborativo onde a borda calcula o grau de incerteza da sua predição através da entropia de Shannon normalizada. Predições com entropia inferior a um limiar fixo são aceites localmente; amostras com alta entropia são encaminhadas para a nuvem.
*   **Aprendizagem Incremental no Dispositivo (*On-Device Incremental Learning*):**  
    Capacidade de um modelo periférico atualizar os seus parâmetros localmente face a novas amostras ou classes sem reinicializar o treino e sem requerer o reenvio de dados para servidores centrais.
*   **Esquecimento Catastrófico (*Catastrophic Forgetting*):**  
    Tendência abrupta e patológica de redes neuronais artificiais para perderem completamente o conhecimento adquirido em tarefas anteriores ao serem ajustadas exclusivamente com novos dados.
*   **Auto-Destilação Local (*On-Device Self-Distillation*):**  
    Técnica de regularização onde o modelo atual utiliza as predições de uma versão histórica de si mesmo (ou de uma ramificação mais confiável) como alvos suaves (*soft targets*), preservando a estabilidade funcional sem necessitar de armazenar amostras passadas.
*   **Gestão de Cache Orientada por Reforço (*RL-Driven Content Caching*):**  
    Aplicação de agentes de aprendizagem por reforço para inferir dinamicamente a utilidade de objetos em memória rápida, superando regras clássicas baseadas em recência (LRU) ou frequência (LFU).
*   **Arquitetura Wolpertinger:**  
    Enquadramento de DRL concebido para gerir espaços de ações discretas massivos, combinando um ator contínuo (DDPG), restrição via $k$-vizinhos mais próximos ($k$-NN) e um crítico para selecionar a melhor ação discreta com baixo custo computacional.
*   **Recompensas Diferidas em Filas Distribuídas (*Delayed Feedback in Distributed Queues*):**  
    Fenómeno intrínseco aos sistemas de computação distribuída assíncrona, onde a confirmação e a métrica de desempenho de uma tarefa alocada ocorrem com atraso temporal imprevisível face ao momento em que a decisão de escalonamento foi executada.
*   **Fronteira de Pareto & Método $\epsilon$-Constraint:**  
    Conceito e técnica de otimização em que se procura um conjunto de soluções onde nenhum critério (ex.: latência) pode ser melhorado sem degradar outro (ex.: consumo de memória), convertendo objetivos secundários em restrições parametrizadas por $\epsilon$.
*   **Enquadramento Agent-System-Environment (ASE):**  
    Formalismo operacional recente que conceptualiza o nó de borda como um agente interactivo situado que monitoriza dinamicamente o fluxo externo de dados e o estado físico interno do sistema, executando reconfigurações arquiteturais sob restrições orçamentais mutáveis.

---

## 2. Protocolo de Pesquisa Sistemática (Search Strategy)

Para garantir uma curadoria reprodutível, neutra e cientificamente profunda, os agentes e investigadores devem aplicar o protocolo estruturado delineado de seguida.

### 2.1 Venues Científicos Prioritários

A literatura deve ser extraída prioritariamente de veículos de topo com revisão por pares rigorosa, abrangendo as áreas de Inteligência Artificial, Sistemas Distribuídos, Redes de Comunicação e Engenharia de Computadores:

| Categoria | Sigla / Nome do Periódico ou Conferência | Foco Editorial Relevante |
|:---|:---|:---|
| **Sistemas Distribuídos & Computadores** | **IEEE TPDS** (Trans. on Parallel and Distributed Systems) | Filas distribuídas, escalonamento, alocação de tarefas |
| | **IEEE TC** (Transactions on Computers) | Armazenamento, hierarquias de cache, arquitetura de hardware |
| | **IEEE ICDCS** (Intl. Conf. on Distributed Computing Systems) | Sistemas colaborativos ponta-a-ponta, edge-cloud systems |
| | **ACM EuroSys / USENIX ATC / NSDI** | Sistemas operacionais distribuídos, infraestruturas de produção |
| **Comunicações & Edge Computing** | **IEEE CST** (Communications Surveys & Tutorials) | Revisões taxonómicas completas de Edge Intelligence |
| | **IEEE T-MC** (Transactions on Mobile Computing) | Modelos de mobilidade, latência em redes móveis e MEC |
| | **IEEE INFOCOM / ACM MobiCom** | Redes de comunicação periférica, compressão de tráfego |
| | **IEEE COMSNETS** | Redes de computadores, particionamento de redes neuronais |
| **Inteligência Artificial & Aprendizagem** | **NeurIPS / ICML / ICLR** | Teoria de quantização, DRL, eficiência e representações |
| | **IJCAI / AAAI** | Algoritmos de colaboração edge-cloud, SNNs, otimização |
| | **IEEE T-PAMI** | Teoria e arquiteturas de redes neuronais eficientes |
| **Workshops Especializados & Preprints** | **BabyLM / MLSys Workshops** | Modelos ultra-leves, transformers a baixo bit, TinyML |
| | **arXiv** (`cs.DC`, `cs.LG`, `cs.AI`, `cs.AR`, `cs.NE`) | Pré-publicações recentes sobre Edge AI adaptativo |

---

### 2.2 Equações de Pesquisa Booleanas Recomendadas

As equações infra encontram-se estruturadas e calibradas para introdução direta em motores de busca bibliográfica académica (IEEE Xplore, ACM Digital Library, Scopus, Google Scholar, Web of Science):

#### Bloco A: Compressão Extrema, Quantização e Redes Ultraleves
```text
("edge AI" OR "edge intelligence" OR "resource-constrained edge") AND ("low-bit" OR "1.58-bit" OR "ternary quantization" OR "BitNet") AND ("episodic memory" OR "attention sinks" OR "multimodal fusion")
```

#### Bloco B: Particionamento Colaborativo e Co-Inferência Edge-Cloud
```text
("edge-cloud collaboration" OR "split computing" OR "model partitioning") AND ("CNN" OR "deep neural network") AND ("early exit" OR "latency-memory optimization" OR "Pareto" OR "entropy filtering")
```

#### Bloco C: Computação Neuromórfica e Redes de Impulsos (SNNs)
```text
("spiking neural networks" OR "SNN") AND ("edge computing" OR "edge-cloud") AND ("energy-efficient" OR "neuromorphic") AND ("on-device incremental learning" OR "knowledge distillation")
```

#### Bloco D: Alocação Autónoma de Recursos e Gestão de Cache via DRL
```text
("reinforcement learning" OR "deep reinforcement learning" OR "contextual bandit") AND ("content caching" OR "cache replacement" OR "workload allocation" OR "distributed queues") AND ("edge" OR "cloud storage")
```

#### Bloco E: Edge AI Adaptativo e Operação em Longo Curso
```text
("edge AI" OR "edge intelligence") AND ("adaptive" OR "continual learning" OR "distribution drift") AND ("Agent-System-Environment" OR "time-varying budgets" OR "on-device self-distillation")
```

---

### 2.3 Janela Temporal de Análise

A estratégia de referenciação organiza-se em dois horizontes temporais analíticos:

1. **Horizonte Seminal e Fundacional (2014 – 2019):**
   - **Propósito:** Mapear a transição da computação em nuvem pura para a borda; origens da quantização estática (FP16/INT8); primeiras propostas de *Deep Reinforcement Learning* aplicado ao armazenamento (*caching*) em estações base móveis (e.g., Zhong et al., 2018; RLCache, 2019) e primeiros enquadramentos de *Early Exit* (BranchyNet).
2. **Horizonte do Estado da Arte Recente (Últimos 3 a 5 anos / 2020 – 2026):**
   - **Propósito:** Capturar a emergência de arquiteturas colaborativas multi-camadas (MEANet, LMOS); sinergias neuromórficas híbridas ANN-SNN (ECC-SNN); quantização extrema com retenção episódica (BitMar); integração de DRL e Bandidos em sistemas de produção distribuídos (RLQ, Catcher+); e a formalização teórica do *Adaptive Edge AI* (Pittorino & Roveri, 2026).

---

## 3. Critérios de Elegibilidade (Inclusão e Exclusão)

O processo de triagem bibliográfica de publicações candidatas obedece a um crivo analítico sistemático:

```
                      Artigo Identificado na Pesquisa
                                     |
               +---------------------+---------------------+
               |                                           |
               v                                           v
      Critérios de Inclusão (+)                  Critérios de Exclusão (-)
   - Validação empírica rigorosa              - Apenas simulação abstrata
   - Métricas de hardware reais               - Omissão de latência de rede
   - Baselines competitivos SOTA              - Heurísticas ingénuas
   - Reprodutibilidade metodológica           - Ensaios puramente conceptuais
               |                                           |
               v                                           v
        INTEGRADO NA TABELA                        REJEITADO
```

### 3.1 Critérios de Inclusão (+)
O documento deve satisfazer cumulativamente pelo menos **três** dos seguintes requisitos:
1. **Validação Empírica em Hardware Real ou Traces Industriais:** Ensaios executados em dispositivos embebidos reais (e.g., Raspberry Pi, NVIDIA Jetson, módulos neuromórficos) ou através de *traces* de produção reconhecidos (e.g., Google Borg, Alibaba Cloud, Tencent traces, Yahoo! Cloud Serving Benchmark — YCSB).
2. **Quantificação Explícita de Trade-offs Multidimensionais:** Medição rigorosa de compromissos entre métricas concorrentes: Acurácia vs. Latência ponta-a-ponta, Consumo Energético (Joules/Watts), Pegada de Memória (RAM/VRAM) e Tráfego de Rede transmitido.
3. **Comparação com Baselines Competitivos Modernos:** Comparação direta contra o estado da arte reconhecido (e.g., contra políticas de cache ARC/LIRS/CACHEUS; contra escalonadores industriais E-PVM; ou contra modelos consolidados ResNet/MobileNetV2).
4. **Transparência Algorítmica e Reprodutibilidade:** Detalhe exaustivo das formulações de perda, topologias de rede, parâmetros de DRL (descontos $\gamma$, taxas de exploração, *experience replay*), e disponibilidade ou explicitação de código e configurações.
5. **Enquadramento em Ambientes de Recursos Restritos:** Foco concreto em restrições de computação, comunicação, energia ou adaptação a desvios no fluxo de dados.

### 3.2 Critérios de Exclusão (-)
Devem ser rejeitados artigos que apresentem uma ou mais das seguintes carências:
1. **Modelos Puramente Teóricos Desconectados de Hardware:** Trabalhos que assumem capacidades de processamento ou largura de banda infinitas, ou que ignoram o custo de comunicação entre nós no particionamento de redes.
2. **Avaliações com Baselines Trivialmente Fracos:** Ensaios que comparam novas políticas exclusivamente com abordagens aleatórias ou com esquemas arcaicos sem incluir alternativas competitivas contemporâneas.
3. **Artigos Opinativos e Relatórios Técnicos sem Arbitragem:** Textos de divulgação geral, apresentações sem artigos completos associados, ou pré-publicações com alegações não fundamentadas em evidências empíricas.
4. **Falta de Especificação Arquitetural:** Textos que omitem como a rede foi particionada, quais os limiares de decisão nas saídas antecipadas ou qual a função de recompensa adotada no agente de reforço.

---

## 4. Roteiro de Extração e Síntese Analítica

Para cada artigo selecionado, o agente ou investigador deve preencher obrigatoriamente a seguinte **Checklist de 5 Pontos de Síntese Analítica**:

```
+---------------------------------------------------------------------------------+
|               CHECKLIST ANALÍTICA DE EXTRAÇÃO DE ARTIGOS                        |
+---------------------------------------------------------------------------------+
| 1. Problema e Motivação                                                         |
|    - Que estrangulamento concreto aborda (latência, energia, cache, drift)?     |
|    - Por que falham as soluções estáticas ou heurísticas convencionais?         |
+---------------------------------------------------------------------------------+
| 2. Inovação Metodológica / Arquitetural                                         |
|    - Qual a arquitetura central (ex.: MEANet, ECC-SNN, LMOS, Catcher+, RLQ)?    |
|    - Qual o algoritmo matemático subjacente (DDPG, LinUCB, DoubleDQN, Pareto)?  |
+---------------------------------------------------------------------------------+
| 3. Datasets, Traces e Protocolo de Avaliação                                    |
|    - Que dados foram usados (CIFAR, ImageNet, YCSB, Borg traces, Alibaba)?      |
|    - Qual a plataforma de hardware de ensaio (Raspberry Pi, Jetson, Nuvem)?     |
+---------------------------------------------------------------------------------+
| 4. Resultados Empíricos e Trade-offs Multidimensionais                          |
|    - Quais os ganhos numéricos exatos em latência, energia, acurácia e tráfego? |
|    - Qual a penalidade ou compromisso operacional verificado?                   |
+---------------------------------------------------------------------------------+
| 5. Limitações e Desafios em Aberto (Open Challenges)                           |
|    - Em que condições de rede ou carga o modelo perde eficácia?                 |
|    - Que questões permanecem por resolver para operação contínua no terreno?    |
+---------------------------------------------------------------------------------+
```

### Detalhe Operacional de Cada Ponto da Checklist:
1. **Problema e Motivação:** Registar a lacuna tecnológica precisa — e.g., ineficiência de políticas estáticas de cache LRU/LFU sob padrões dinâmicos, sobrecarga da nuvem com instâncias simples, ou incapacidade de transformers em manter memória em hardware de 14M parâmetros.
2. **Inovação Metodológica/Arquitetural:** Especificar os blocos matemáticos do sistema — definição do espaço de estados, ações e recompensas nos agentes DRL; formulação de funções de perda conjunta em destilação; ou formulação do critério de divisão particionada.
3. **Datasets e Protocolo de Avaliação:** Identificar se os testes foram conduzidos em dados sintéticos (e.g., Zipf), conjuntos de referência de visão computacional, ou *traces* reais de telemetria de centros de dados de grande escala.
4. **Resultados Empíricos e Trade-offs:** Anotar sempre os valores numéricos exatos citados no artigo — percentagem de redução de consumo energético, fator multiplicador de velocidade de inferência (*speedup*), taxa de acerto de cache (*hit rate*) e redução de tráfego de rede.
5. **Limitações e Desafios em Aberto:** Registar as fragilidades apontadas pelos próprios autores — e.g., complexidade de treino conjunto ANN-SNN, sobrecarga de sincronização assíncrona, dependência de canais de comunicação estáveis para saídas complexas.

---

## 5. Padrão de Formatação para a Tabela de Literatura (`README.md`)

O ficheiro central `README.md` localizado na pasta de Edge AI integra uma tabela de revisão do estado da arte de 6 colunas, cujo padrão deve ser escrupulosamente respeitado em qualquer nova inserção.

### 5.1 Especificação Rigorosa das 6 Colunas Obrigatórias

| Coluna | Título | Diretrizes Específicas de Conteúdo e Formatação |
|:---:|:---|:---|
| **1** | **Nome** | Título oficial completo do artigo científico em língua inglesa. |
| **2** | **Detalhes** | Metadados bibliográficos estruturados verticalmente com tags HTML `<br><br>`:<br>• **Autores:** Primeiro autor *et al.* ou lista integral.<br>• **Ano:** Ano de publicação.<br>• **Pub:** Sigla do evento ou revista (e.g., IEEE TPDS, IJCAI, ICDCS, CISS, BabyLM). |
| **3** | **Abstract** | Excertos textuais literais entre aspas (`"..."`) extraídos diretamente do resumo oficial do artigo, focando a motivação, proposta metodológica e declarações centrais de desempenho. |
| **4** | **Conclusion** | Excertos textuais literais entre aspas (`"..."`) retirados das conclusões do artigo original, refletindo as deduções dos autores e os desafios futuros. |
| **5** | **Resumo (NotebookLM)** | Síntese analítica detalhada em língua portuguesa estruturada impreterivelmente em **quatro parágrafos encadeados** com `<br><br>` (ver secção 5.2). |
| **6** | **Citação** | Referência bibliográfica completa na **Norma Vancouver (NLM)**, finalizada com a hiperligação oficial ativa para o respetivo identificador DOI (ou arXiv permalink com prefixo DOI quando aplicável). |

---

### 5.2 Estrutura Padronizada do "Resumo (NotebookLM)" (4 Parágrafos)

A redação da quinta coluna deve cumprir escrupulosamente a seguinte cadência concetual:

*   **Parágrafo 1 — Contextualização do Problema e Motivação Operacional:**  
    Apresentação do artigo, definição da classe de problema dentro do espetro de Edge AI ou alocação distribuída de recursos e explicação do motivo pelo qual abordagens centralizadas ou estáticas anteriores se revelam inadequadas.
*   **Parágrafo 2 — Mecanismo Proposto e Inovação Arquitetural:**  
    Exposição rigorosa da metodologia concebida (partição por blocos, rede de impulso neuromórfica, quantização a 1.58-bit com memória episódica, ou formulação de MDP com algoritmos DRL/Bandits). Explicação da dinâmica entre componentes (borda, nuvem, agente de decisão).
*   **Parágrafo 3 — Protocolo Experimental e Métricas Quantitativas:**  
    Resumo dos cenários de teste, dados e plataformas de teste utilizadas (benchmarks, traces industriais, dispositivos periféricos). Transcrição precisa das métricas alcançadas (percentagens exatas de redução de latência, poupança de energia, taxa de acerto de cache, redução de tráfego de dados).
*   **Parágrafo 4 — Significado Teórico, Limitações e Desdobramentos:**  
    Discussão das implicações para o estado da arte de Edge AI e sistemas distribuídos, indicação clara das limitações operacionais admitidas no trabalho e pistas de investigação futura.

---

### 5.3 Exemplo Canónico de Registo na Tabela

Para aferição e calibração de estilo, apresenta-se o registo oficial do artigo **ECC-SNN** constante no `README.md`:

```markdown
| ECC-SNN: Cost-Effective Edge-Cloud Collaboration for Spiking Neural Networks | **Autores:** Yu et al.<br><br>**Ano:** 2025<br><br>**Pub:** IJCAI | "To address these challenges, we propose ECC-SNN, a novel edge-cloud collaboration framework incorporating energy-efficient spiking neural networks (SNNs) to offload more computational workload from the cloud to the edge, thereby improving costeffectiveness and reducing reliance on the cloud."<br><br>"ECC-SNN employs a joint training approach that integrates ANN and SNN models, enabling edge devices to leverage knowledge from cloud models for enhanced performance while reducing energy consumption and processing latency."<br><br>"Furthermore, ECC-SNN features an on-device incremental learning algorithm that enables edge models to continuously adapt to dynamic environments, reducing the communication overhead and resource consumption associated with frequent cloud update requests."<br><br>"Extensive experimental results on four datasets demonstrate that ECC-SNN improves accuracy by 4.15%, reduces average energy consumption by 79.4%, and lowers average processing latency by 39.1%." | "In this study, we propose ECC-SNN, a cost-effective and efficient edge-cloud collaborative framework designed for SNN-based classifiers."<br><br>"By employing the joint training approach and adaptive on-device incremental learning with the assistance of a powerful ANN model on the cloud server, the SNN model in ECC-SNN gains enhanced predictive capability compared to the standalone edge SNN, significantly reducing both energy costs and inference latency as the system operates in different dynamic IoT scenarios." | Este artigo introduz o ECC-SNN, uma estrutura de colaboração entre a borda e a nuvem (*edge-cloud collaboration*) que combina a eficiência energética e baixa latência das Redes Neuronais de Impulsos (*Spiking Neural Networks* - SNNs) na borda com a elevada precisão de Redes Neuronais Artificiais (*Artificial Neural Networks* - ANNs) na nuvem. O objetivo principal consiste em transferir carga computacional da nuvem para os dispositivos de borda, aumentando a relação custo-eficácia e reduzindo a dependência de servidores remotos.<br><br>O funcionamento do sistema divide-se em três etapas fundamentais:<br><br>1. Fase de Configuração (*Setup Stage*): Utiliza um método de treino conjunto baseado em destilação de conhecimento para transferir informação de uma ANN professora na nuvem para a SNN de borda, realizando o alinhamento de características entre representações numéricas (*floating-point*) e sequências de impulsos (*spike trains*).<br><br>2. Fase de Execução (*Execution Stage*): Aplica uma estratégia de filtragem por entropia normalizada para avaliar a incerteza preditiva da SNN. Amostras com alta confiança são classificadas diretamente na borda, enquanto casos ambíguos ou difíceis são descarregados para a ANN na nuvem.<br><br>3. Fase de Atualização (*Update Stage*): Executa um algoritmo de aprendizagem incremental no próprio dispositivo (*on-device incremental learning*) sem recurso a um *buffer* de exemplares, mitigando o esquecimento catastrófico através de auto-destilação local quando ocorrem desvios de distribuição (*prior probability distribution drift*).<br><br>Avaliado em quatro conjuntos de dados de classificação de imagem (CIFAR-10, CIFAR-100, Caltech e Tiny-ImageNet), o ECC-SNN demonstrou um aumento médio de precisão de 4,15%, uma redução do consumo de energia de 79,4% e uma diminuição da latência de processamento de 39,1% face a abordagens puramente de borda ou de nuvem. | Yu D, Lv C, Du X, Jiang L, Tong W, Liao Z, Zheng X, Deng S. ECC-SNN: Cost-Effective Edge-Cloud Collaboration for Spiking Neural Networks. arXiv preprint arXiv:2505.20835; 2025. https://doi.org/10.48550/arXiv.2505.20835. |
```

---

## 6. Guardrails de Qualidade & Anti-Alucinação

Para assegurar a idoneidade científica e a preservação do repositório como recurso de nível académico internacional, devem ser cumpridas impreterivelmente as seguintes salvaguardas:

### 6.1 Verificação Rigorosa de DOIs e Fontes Primárias
- **Autenticidade das Ligações:** Todo o identificador DOI citado deve ser auditado através do validador oficial `https://doi.org/`. É proibido inserir links truncados ou inventados.
- **Conferência Cruzada de Metadados:** Autores, títulos de periódicos e anos de publicação devem corresponder com exatidão ao registo nos diretórios DBLP, CrossRef, IEEE Xplore ou ACM DL.

### 6.2 Proibição Estrita de Fabrico ou Aproximação de Métricas
- Métricas experimentais (percentagens de poupança energética, tempo de latência em milissegundos, variações de acurácia Top-1/Top-5) devem ser transcritas **exatamente como constam no artigo**. Se um artigo refere "redução média de 79.4%", é estritamente vedado arredondar para "cerca de 80%".
- Quando os resultados empíricos forem dependentes de configurações de hardware específicas (e.g., Raspberry Pi 4 com Broadcom BCM2711), tal condicionamento deve ser explicitamente indicado no texto.

### 6.3 Integridade Textual nas Citações Diretas
- Os campos **Abstract** e **Conclusion** destinam-se exclusivamente a **transcrições literais das palavras dos autores originais**. Não é permitida a reformulação, a supressão descontextualizada ou a inserção de notas pessoais dentro das aspas.

### 6.4 Sobriedade Concetual e Neutralidade Epistemológica
- O tom do documento deve manter sobriedade, rigor analítico e precisão terminológica. Devem evitar-se expressões promocionais ou sensacionalistas (e.g., "arquitetura milagrosa", "avanço incomparável"). As propostas devem ser sempre apreciadas à luz dos seus pressupostos teóricos, custos de computação e limitações operacionais conhecidas.
