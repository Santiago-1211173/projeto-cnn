# Protocolo Mestre de Curadoria e Governação da Literatura Científica (Raiz)

> **Ficheiro Central Associado:** [`README.md`](README.md) (no mesmo diretório)  
> **Compêndio Terminológico Oficial:** [`GLOSSARIO.md`](GLOSSARIO.md) (no mesmo diretório)  
> **Natureza do Documento:** Metaprotocolo de governação científica, taxonomia comparativa, padronização terminológica e regras de roteamento bibliográfico para agentes autónomos e investigadores.  
> **Domínio Científico:** Inteligência Artificial / Aprendizagem Automática / Sistemas Neurais Adaptativos e Tomada de Decisão.

---

## 1. Enquadramento e Ontologia Global do Repositório

O diretório `docs/Literatura` é um centro de conhecimento científico estruturado que cobre o ciclo holístico de sistemas adaptativos inteligentes. Enquanto os ficheiros `agent_literature_guidelines.md` situados em cada subdiretório regulam a profundidade específica de cada matéria, este documento raiz rege a **governação global**, a **ontologia comparativa**, o **alinhamento terminológico com o [`GLOSSARIO.md`](GLOSSARIO.md)** e a **resolução de sobreposição temática entre tópicos**.

### 1.1 As 12 Áreas Científicas e suas Fronteiras Conceptuais

Para evitar dispersão ou catalogação incorreta, qualquer artigo recolhido deve ser atribuído à sua pasta temática primária de acordo com as seguintes fronteiras:

1. **`Contextual Bandit`**: Focado estritamente na tomada de decisão em passo único com feedback parcial (onde o agente recebe uma recompensa imediata sem alterar a transição do estado futuro). *Exclui MDPs de horizonte longo.*
2. **`Continual Learning`**: Focado no problema do esquecimento catastrófico e na aquisição contínua de conhecimento ao longo de sequências de tarefas ou distribuições mutáveis (*lifelong learning*).
3. **`Deep Learning`**: Focado em arquiteturas fundamentais de representação visual e multimodal (CNNs, Vision Transformers, modelos híbridos, funções de ativação, normalização e dinâmicas de gradiente).
4. **`Early-Exit`**: Focado em inferência condicional e adaptativa mediante ramificações intermediárias (*side branches*), balanceamento entre tempo de cálculo e acurácia, e mitigação do *overthinking*.
5. **`Edge AI`**: Focado na translação de modelos para silício e hardware com fortes restrições energéticas e de memória (quantização INT8/FP16, poda estruturada/não estruturada, destilação de conhecimento e latência em tempo real).
6. **`Episodic Memory`**: Focado em estruturas de armazenamento de experiências passadas (buffers de repetição, memórias de trabalho vs. de longo prazo, recuperação métrica por $k$-NN e consolidação não-paramétrica).
7. **`Hopfield Networks`**: Focado em memórias autoassociativas, funções de energia de Lyapunov, Modern Hopfield Networks (Dense Associative Memories) e limites de capacidade de armazenamento exponencial.
8. **`Out-of-Distribution`**: Focado em métodos de quantificação de incerteza (epistémica e aleatória), calibração de probabilidades softmax, deteção de anomalias/novidades e robustez face a desvios de distribuição (*domain shift*).
9. **`Percecao Ativa e Atencao Visual`**: Focado em estratégias bioinspiradas de recolha de informação seletiva (visão foveal, movimentos sacádicos e de fixação, modelos estocásticos de atenção rígida via REINFORCE, como RAM e MRAM).
10. **`Q-Learning`**: Focado na aprendizagem por diferença temporal baseada na função de valor de ação $Q(s,a)$, na aproximação da Equação de Bellman e na estabilidade de redes profundas (DQN, Double DQN, Dueling DQN).
11. **`Reinforcement Learning`**: Focado na teoria geral de MDP/POMDP, otimização direta de políticas (Policy Gradients, Actor-Critic, PPO, SAC), exploração intrínseca, desenho de funções de recompensa e transferência simulação-realidade (*sim-to-real*).
12. **`RL + LLMs`**: Focado na convergência entre modelos de linguagem de grande escala e algoritmos de reforço (RLHF, DPO, raciocínio em múltiplos passos com recompensas verificáveis e LLMs como geradores de recompensas semânticas).

---

## 2. Regras de Resolução de Sobreposição Temática (Cross-Topic Routing)

Na literatura moderna de IA, muitos artigos combinam dois ou mais paradigmas (por exemplo, um artigo que utiliza *Reinforcement Learning* para aprender uma política de *Early-Exit*, ou *Vision Transformers* combinados com *Contextual Bandits*). O agente deve seguir esta árvore de decisão para atribuir a pasta correta:

```
               [Novo Artigo Científico a Registar]
                                |
             O artigo propõe um avanço metodológico
                central num tópico ou usa outro
                    apenas como ferramenta?
                                |
         +----------------------+----------------------+
         |                                             |
[Contribuição Metodológica Central]           [Híbrido de Igual Peso]
         |                                             |
Classificar na pasta da contribuição         Priorizar a pasta onde o
principal. (Ex.: DRL usado para              desafio resolvido é mais
Early-Exit -> Classificar em Early-Exit)     urgente (ou catalogar como
                                             referência cruzada)
```

### Exemplos Práticos de Desempate:
* **Artigo sobre CNNs com saídas intermediárias avaliado em Raspberry Pi:** Se a inovação for a formulação do limiar de saída precoce, vai para `Early-Exit`. Se o foco for o perfil de consumo elétrico, quantização e aceleração de hardware, vai para `Edge AI`.
* **Artigo que usa LLMs para criar funções de recompensa em robótica com PPO:** Vai para `RL + LLMs`, pois a contribuição central está na simbiose entre o modelo de linguagem e o sinal de reforço.
* **Artigo sobre k-NN aplicado a memórias de experiências num agente de decisão:** Se a contribuição for a estabilização do buffer de memória e a recuperação métrica, vai para `Episodic Memory`. Se a contribuição for a prova teórica de arrependimento (*regret*), vai para `Contextual Bandit`.

---

## 3. Governação Terminológica e Manutenção do Glossário Científico (`GLOSSARIO.md`)

O documento [`GLOSSARIO.md`](GLOSSARIO.md) atua como o alicerce concetual unificado de todo o repositório. Para garantir uma linguagem rigorosa, padronizada e pedagógica para orientadores e investigadores de outras áreas científicas, os agentes devem respeitar as seguintes normas:

### 3.1 Dicotomia Didática Obrigatória
Qualquer conceito técnico formalizado no repositório tem de manter equilíbrio absoluto entre dois eixos:
1. **Rigor Matemático e Técnico Formal:** Definição analítica precisa acompanhada de equações formais em LaTeX (formulações de Bellman, funções de energia de Lyapunov, divergências KL, matrizes de covariância, estimadores UCB, etc.).
2. **Intuição Pedagógica do Quotidiano:** Uma analogia do mundo real (engenharia, trânsito, desporto, medicina) que torne o conceito imediatamente compreensível a investigadores que não trabalham diretamente com redes neuronais.

### 3.2 Protocolo de Atualização do Glossário
Sempre que uma nova vaga de literatura introduzir uma arquitetura seminal, métrica inovadora ou formulação algorítmica fundamental ainda não contemplada no compêndio:
* A nova entrada deve ser inserida na ordem alfabética estrita correspondente no [`GLOSSARIO.md`](GLOSSARIO.md).
* A entrada deve conter impreterivelmente as 5 secções padronizadas:
  1. `Área Científica Primária`
  2. `Definição Formal`
  3. `Intuição Pedagógica`
  4. `Papel Prático nas Redes Neuronais`
  5. `Ver Também` (com âncoras internas funcionais).
* As âncoras HTML (`<a id="..."></a>`) devem ser integradas para suportar hiperligações diretas a partir de qualquer documento.

---

## 4. Metodologia Sistemática de Pesquisa (Meta-Search Strategy)

Ao realizar varrimentos bibliográficos globais ou atualizar múltiplos tópicos em simultâneo, o agente deve operar sob um padrão PRISMA simplificado:

1. **Fase de Identificação (Identification):**
   * Consultar bases académicas (Google Scholar, Semantic Scholar, IEEE Xplore, ACM Digital Library, arXiv).
   * Utilizar combinações booleanas que cruzem o tópico pretendido com termos de validação (`"deep learning" OR "neural networks"`) e termos de rigor (`"benchmark" OR "empirical evaluation" OR "state-of-the-art"`).
2. **Fase de Triagem (Screening):**
   * Ler título, abstract e conclusões para descartar trabalhos puramente especulativos, pré-publicações incompletas ou artigos sem métricas quantitativas.
3. **Fase de Elegibilidade (Eligibility):**
   * Verificar se o artigo contém detalhes arquiteturais (função de perda, parâmetros, datasets, baselines de comparação).
4. **Fase de Inclusão (Inclusion):**
   * Extrair os dados estruturados e inseri-los na tabela do `README.md` da subpasta respetiva, atualizando se necessário o resumo de tendências do tópico.

---

## 5. Padrão Global de Formatação da Tabela de Literatura

Todas as tabelas de literatura presentes nas 12 subpastas partilham rigorosamente o formato de 6 colunas, sem exceção:

```markdown
| Nome | Detalhes | Abstract | Conclusion | Resumo (NotebookLM) | Citação |
|:---|:---|:---|:---|:---|:---|
| [Título Oficial em Inglês] | **Autores:** [Nomes]<br><br>**Data de publicação:** [Data]<br><br>**Publisher:** [Editora]<br><br>**Livro/Journal:** [Conferência ou Revista]<br><br>**Volume:** [Vol]<br><br>**Número:** [Nº]<br><br>**Páginas:** [Páginas]<br><br>**DOI:** https://doi.org/[DOI] | "[Excerto textual do abstract]" | "[Excerto textual da conclusão]" | [Resumo pedagógico em Português em 3-4 parágrafos] | [Citação completa em estilo Vancouver com link DOI] |
```

### Diretrizes de Redação para o "Resumo (NotebookLM)":
O resumo é o componente mais valioso para a partilha de conhecimento com investigadores que não trabalham diretamente com redes neuronais. Deve ser redigido em **Português de Portugal** e conter:
* **Parágrafo 1 - Contexto e Problema:** Explicar qual a deficiência das redes clássicas que o artigo pretende corrigir.
* **Parágrafo 2 - Proposta e Mecanismo:** Descrever a solução arquitetural de modo intuitivo e mecanístico (como os dados fluem e como a rede aprende).
* **Parágrafo 3 - Validação Empírica:** Destacar os conjuntos de dados de teste (ex.: MNIST, CIFAR, ImageNet, etc.), as métricas numéricas concretas e os modelos concorrentes superados.
* **Parágrafo 4 - Significado e Impacto:** Sintetizar em 2 frases o que este avanço traz de novo para a área da inteligência artificial.

---

## 6. Salvaguardas Estritas e Protocolo Anti-Alucinação

Para manter a integridade académica do repositório em auditorias externas ou perante orientadores e supervisores:

1. **Validação Ativa de DOIs:** Todo o link de DOI (`https://doi.org/...`) deve corresponder ao identificador digital de objeto real registado na *International DOI Foundation*. Caso um artigo seja um preprint recente do arXiv sem DOI formal emitido, utiliza-se a hiperligação canónica do arXiv (`https://doi.org/10.48550/arXiv.XXXX.XXXXX` ou `https://arxiv.org/abs/XXXX.XXXXX`).
2. **Proibição de Extrapolação de Métricas:** Se um artigo reportar "ganho de 3.2% de precisão e redução de 40% em FLOPs", o resumo deve reproduzir exatamente estes números. É estritamente proibido arredondar ou inventar dados empíricos.
3. **Neutralidade e Rigor Terminológico:** Proibido o uso de chavões comerciais ("solução mágica", "desempenho perfeito", "revolução total"). As limitações do método (ex.: aumento de memória durante o treino, hiperparâmetros sensíveis) devem constar sempre que identificadas pelos autores.
4. **Isolamento de Código Interno:** Nenhuma referência a projetos internos, repositórios locais, nomes de ficheiros de código ou sistemas proprietários deve ser introduzida em qualquer documento deste repositório de literatura.
5. **Conformidade com o Glossário:** As nomenclaturas técnicas e métricas utilizadas na documentação devem coincidir rigorosamente com as definições padronizadas no [`GLOSSARIO.md`](GLOSSARIO.md).
6. **Ausência Absoluta de Emojis:** É estritamente proibida a utilização de qualquer emoji, pictograma ou emoticon em qualquer ficheiro deste repositório (`docs/Literatura/`), garantindo um padrão de apresentação puramente académico, formal e sóbrio.
