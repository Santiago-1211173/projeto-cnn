# Hub Central de Literatura Científica em Inteligência Artificial e Redes Neuronais

> **Natureza do Repositório:** Base de Conhecimento Estruturada, Taxonomia e Protocolo Global de Navegação na Literatura Científica.  
> **Compêndio Terminológico Oficial:** Consulte o [Glossário Científico Unificado de IA e Redes Neuronais](GLOSSARIO.md) para definições formais, intuições pedagógicas e formulações matemáticas completas de A a Z cobrindo as 12 áreas científicas.  
> **Público-Alvo:** Investigadores, orientadores científicos, estudantes de pós-graduação e agentes autónomos de IA.  
> **Âmbito:** Estudo sistemático do ciclo completo de sistemas inteligentes modernos — desde a perceção visual bioinspirada e representações profundas, passando por inferência adaptativa, memória e deteção de incerteza, até à tomada de decisão sequencial por reforço e otimização para hardware embarcado (*Edge*).

---

## 1. Visão Geral e Enquadramento Científico

A investigação contemporânea em Inteligência Artificial (IA) e Aprendizagem Automática (*Machine Learning* - ML) ultrapassou a era dos modelos estáticos e puramente supervisionados. Os desafios da próxima geração de sistemas inteligentes exigem arquiteturas capazes de:
1. **Percecionar ativamente** ambientes complexos e selecionar informação relevante de forma económica (*Perceção Ativa e Atenção Visual*);
2. **Extrair representações latentes robustas** a partir de dados sensoriais brutos (*Deep Learning*);
3. **Decidir com parcimónia computacional**, interrompendo o processamento quando a certeza for suficiente (*Early-Exit*);
4. **Quantificar a sua própria dúvida** e detetar quando os dados divergem da distribuição de treino (*Out-of-Distribution*);
5. **Armazenar e recuperar experiências passadas** de forma não-paramétrica ou associativa (*Episodic Memory* e *Hopfield Networks*);
6. **Agir sequencialmente sob incerteza e feedback parcial**, maximizando recompensas a curto e longo prazo (*Contextual Bandit*, *Q-Learning* e *Reinforcement Learning*);
7. **Aprender continuamente ao longo do tempo** sem apagar conhecimento prévio (*Continual Learning*);
8. **Articular raciocínio semântico e alinhamento** através de modelos de linguagem (*RL + LLMs*);
9. **Operar de forma eficiente em dispositivos com recursos computacionais e energéticos limitados** (*Edge AI*).

Este repositório organiza a literatura internacional revista por pares (*peer-reviewed*) e pré-publicações seminais nestas **12 frentes científicas fundamentais**.

---

## 2. Mapa Conceitual Integrado da Literatura

O diagrama seguinte ilustra como os 12 tópicos se articulam de forma lógica e coerente na conceção de um ecossistema inteligente autónomo e adaptativo:

```
                       +--------------------------------------------------+
                       |             Ambiente / Dados Brutos              |
                       +--------------------------------------------------+
                                                |
                                                v
                       +--------------------------------------------------+
                       | 9. Perceção Ativa e Atenção Visual               |
                       | - Visão foveal e movimentos sacádicos            |
                       | - Seleção de regiões críticas (hard attention)   |
                       +--------------------------------------------------+
                                                |
                                                v
                       +--------------------------------------------------+
                       | 3. Deep Learning                                 |
                       | - Extração hierárquica de características        |
                       | - CNNs, Vision Transformers e Modelos Híbridos   |
                       | - Vetores de representação no espaço latente     |
                       +--------------------------------------------------+
                         /                      |                       \
        [Inferência Rápida]                     |                 [Gestão de Certeza]
                       v                        v                        v
+-------------------------------+ +---------------------------+ +-------------------------------+
| 4. Early-Exit                 | | 8. Out-of-Distribution    | | 6. Episodic Memory            |
| - Ramificações intermediárias | | - Deteção de anomalias    | | - Buffer de experiências k-NN |
| - Prevenção de overthinking   | | - Calibração de incerteza | | 7. Hopfield Networks          |
| - Poupança adaptativa de FLOPs| | - Desvio de domínio       | | - Memória associativa densa   |
+-------------------------------+ +---------------------------+ +-------------------------------+
                                                |
                                                v
                       +--------------------------------------------------+
                       | Tomada de Decisão Sequencial & Controlo:         |
                       | - 1. Contextual Bandit (Passo único / feedback)  |
                       | - 10. Q-Learning (Valores Q e aproximação DQN)   |
                       | - 11. Reinforcement Learning (Políticas & MDP)   |
                       +--------------------------------------------------+
                                                |
                         +----------------------+----------------------+
                         |                                             |
                         v                                             v
       +------------------------------------+        +------------------------------------+
       | 2. Continual Learning              |        | 12. RL + LLMs                      |
       | - Prevenção do esquecimento        |        | - Alinhamento por preferências     |
       | - Adaptação plástica a fluxos      |        | - Raciocínio estruturado           |
       | - Consolidação de parâmetros       |        | - Desenho de funções de recompensa |
       +------------------------------------+        +------------------------------------+
                         \                                             /
                          \                                           /
                           +-----------------------------------------+
                           | 5. Edge AI                              |
                           | - Quantização, poda (pruning)           |
                           | - Eficiência energética em silício      |
                           | - Inferência ultrarrápida em tempo real |
                           +-----------------------------------------+
```

---

## 3. Glossário Científico Unificado de IA e Redes Neuronais

Para nivelar conceitos entre especialistas em redes neuronais e investigadores ou orientadores de outros domínios científicos, este repositório disponibiliza uma obra de referência terminológica e pedagógica autónoma:

* **Ficheiro Central:** [`GLOSSARIO.md`](GLOSSARIO.md)
* **Âmbito:** 111 termos fundamentais que cobrem exaustivamente as 12 frentes científicas da literatura.
* **Organização:** Alfabética estrita (A–Z) com navegação instantânea por âncoras.
* **Arquitetura Didática Padrão de Cada Entrada:**
  1. **Área Científica Primária:** Enquadramento na taxonomia das 12 pastas temáticas.
  2. **Definição Formal:** Precisão matemática com equações completas em LaTeX (formulações de Bellman, divergências KL, matrizes de Fisher, distâncias de Mahalanobis, limites de Hoeffding, etc.).
  3. **Intuição Pedagógica:** Analogias claras e acessíveis do mundo real (engenharia, trânsito, desporto, medicina) concebidas para permitir compreensão imediata por não-especialistas.
  4. **Papel Prático nas Redes Neuronais:** Problema prático de treino ou inferência resolvido pelo conceito (poupança de FLOPs, mitigação de esquecimento catastrófico, atenuação de gradientes, calibração de incerteza).
  5. **Ver Também:** Rede de hiperligações internas bidirecionais entre conceitos correlacionados.

---

## 4. Catálogo das 12 Áreas Científicas

Cada diretório abaixo contém um ficheiro `README.md` (com o resumo temático, dicionário técnico e a tabela de revisão de artigos) e um `agent_literature_guidelines.md` (com o protocolo exato de curadoria e pesquisa):

| # | Tópico Científico | Subárea de Investigação | Questão Científica Central | Atalhos |
| :-: | :--- | :--- | :--- | :---: |
| **1** | **[Contextual Bandit](./Contextual Bandit/)** | Tomada de Decisão com Feedback Parcial | Como selecionar a melhor ação condicionada a um contexto sem conhecer as consequências das ações descartadas? | [README](./Contextual Bandit/README.md) • [Protocolo](./Contextual Bandit/agent_literature_guidelines.md) |
| **2** | **[Continual Learning](./Continual Learning/)** | Aprendizagem Contínua e Vida Útil | Como aprender tarefas novas sequencialmente sem sofrer de esquecimento catastrófico das tarefas antigas? | [README](./Continual Learning/README.md) • [Protocolo](./Continual Learning/agent_literature_guidelines.md) |
| **3** | **[Deep Learning](./Deep Learning/)** | Representações Hierárquicas Profundas | Como estruturar redes profundas (CNNs, ViTs, Híbridos) para extrair invariâncias espaciais e texturas robustas? | [README](./Deep Learning/README.md) • [Protocolo](./Deep Learning/agent_literature_guidelines.md) |
| **4** | **[Early-Exit](./Early-Exit/)** | Inferência Dinâmica e Adaptativa | Como permitir que amostras fáceis saiam em camadas preliminares, poupando energia e evitando o *overthinking*? | [README](./Early-Exit/README.md) • [Protocolo](./Early-Exit/agent_literature_guidelines.md) |
| **5** | **[Edge AI](./Edge AI/)** | Computação Eficiente em Hardware | Como comprimir modelos através de quantização e poda para execução em tempo real em dispositivos com restrições severas? | [README](./Edge AI/README.md) • [Protocolo](./Edge AI/agent_literature_guidelines.md) |
| **6** | **[Episodic Memory](./Episodic Memory/)** | Memória Não-Paramétrica e Replay | Como registar eventos específicos de forma persistente e recuperá-los instantaneamente por semelhança métrica? | [README](./Episodic Memory/README.md) • [Protocolo](./Episodic Memory/agent_literature_guidelines.md) |
| **7** | **[Hopfield Networks](./Hopfield Networks/)** | Memória Autoassociativa e Energia | Como redes associativas modernas e funções de energia conseguem armazenar e recuperar padrões exponencialmente complexos? | [README](./Hopfield Networks/README.md) • [Protocolo](./Hopfield Networks/agent_literature_guidelines.md) |
| **8** | **[Out-of-Distribution](./Out-of-Distribution/)** | Incerteza e Deteção de Anomalias | Como estimar se uma entrada sensorial pertence a uma distribuição desconhecida antes de tomar decisões catastróficas? | [README](./Out-of-Distribution/README.md) • [Protocolo](./Out-of-Distribution/agent_literature_guidelines.md) |
| **9** | **[Percecao Ativa e Atencao Visual](./Percecao Ativa e Atencao Visual/)** | Visão Foveal e Atenção Rígida | Como emular o olho humano processando apenas relances seletivos de alta resolução guiados por políticas estocásticas? | [README](./Percecao Ativa e Atencao Visual/README.md) • [Protocolo](./Percecao Ativa e Atencao Visual/agent_literature_guidelines.md) |
| **10** | **[Q-Learning](./Q-Learning/)** | Métodos Baseados em Valor (*Off-Policy*) | Como aproximar a Equação de Bellman de forma estável quando combinada com representações neurais profundas? | [README](./Q-Learning/README.md) • [Protocolo](./Q-Learning/agent_literature_guidelines.md) |
| **11** | **[Reinforcement Learning](./Reinforcement Learning/)** | Decisões Sequenciais de Longo Prazo | Como otimizar políticas em ambientes dinâmicos e transferir conhecimento simulado para o mundo real (*sim-to-real*)? | [README](./Reinforcement Learning/README.md) • [Protocolo](./Reinforcement Learning/agent_literature_guidelines.md) |
| **12** | **[RL + LLMs](./RL + LLMs/)** | Modelos de Linguagem e Decisão | Como integrar a capacidade de raciocínio de grandes modelos para guiar agentes, formatar recompensas e acelerar a aprendizagem? | [README](./RL + LLMs/README.md) • [Protocolo](./RL + LLMs/agent_literature_guidelines.md) |

---

## 5. Metodologia Sistemática de Curadoria e Inclusão

Toda a literatura registada neste repositório segue um protocolo rigoroso inspirado nas diretrizes **PRISMA** (*Preferred Reporting Items for Systematic Reviews and Meta-Analyses*), garantindo auditabilidade académica:

### 5.1 Fontes Científicas e Critérios de Qualidade
As pesquisas priorizam publicações indexadas nas seguintes instâncias:
- **Conferências A\* / A:** NeurIPS, ICML, ICLR, CVPR, ECCV, ICCV, AAAI, IJCAI, KDD, CoRL.
- **Revistas Q1 / Q2:** IEEE Transactions on Pattern Analysis and Machine Intelligence (T-PAMI), Journal of Machine Learning Research (JMLR), Neural Networks, Applied Intelligence, Artificial Intelligence, Nature Machine Intelligence.
- **Repositórios Abertos de Pré-Publicação:** arXiv (especificamente categorias `cs.LG`, `cs.AI`, `cs.CV`, `stat.ML`), filtrando apenas artigos com demonstrações matemáticas rigorosas ou código aberto associado.

### 5.2 Critérios de Inclusão
1. **Evidência Empírica Verificável:** O estudo deve comparar os seus resultados contra *baselines* reconhecidos em conjuntos de dados públicos de referência (ex.: MNIST, CIFAR-10/100, ImageNet, SVHN, Gym, MuJoCo, etc.).
2. **Transparência Arquitetural:** Existência de formulação matemática clara das perdas, funções de ativação e dimensões tensorais.
3. **Métricas de Eficiência:** Reportar, sempre que disponível, métricas de complexidade (tempo de inferência em milissegundos, número de parâmetros aprendíveis e FLOPs).

### 5.3 Critérios de Exclusão
1. Artigos puramente concetuais ou de opinião sem suporte experimental.
2. Estudos sem grupo de controlo ou sem metodologia de calibração replicável.
3. Publicações de fontes predatórias sem revisão por pares credível.

---

## 6. Estrutura Padrão das Tabelas de Literatura

Para manter a consistência e permitir a leitura comparativa imediata por orientadores e investigadores, todas as tabelas em cada `README.md` contêm exatamente 6 colunas estruturadas:

| Coluna | Descrição e Requisitos de Formatação |
| :--- | :--- |
| **Nome** | Título oficial completo do artigo no idioma original em inglês. |
| **Detalhes** | Metadados catalográficos separados por quebras `<br><br>`:<br>• **Autores:** Lista completa.<br>• **Data:** Dia, mês e ano de publicação.<br>• **Publisher:** Editora científica.<br>• **Livro/Journal:** Nome do periódico ou conferência.<br>• **Volume, Número, Páginas:** Dados do fascículo.<br>• **DOI:** URL resolúvel ativa (ex.: `https://doi.org/...`). |
| **Abstract** | Citações textuais literais entre aspas das passagens mais relevantes do resumo original. |
| **Conclusion** | Citações literais entre aspas das principais deduções e limites reportados nas conclusões dos autores. |
| **Resumo (NotebookLM)** | Síntese analítica em **Português**, redigida em 3 a 4 parágrafos focados em: *Contexto do Problema*, *Mecanismo Arquitetural Proposto*, *Resultados e Ganhos Numéricos*, e *Implicações Teóricas/Práticas*. |
| **Citação** | Citação bibliográfica estrita segundo a norma **Vancouver (NLM)**, com link DOI funcional no final. |

---

## 7. Guia de Leitura Recomendado para Não-Especialistas

Para investigadores ou orientadores cujo domínio principal não seja Redes Neuronais, sugere-se a seguinte sequência pedagógica de leitura:

```
[0. Glossário Científico Unificado]  (Familiarização prévia com os termos e intuições pedagógicas)
         ↓
[1. Deep Learning]                   (Compreender as fundações convolucionais e atencionais)
         ↓
[2. Early-Exit & Edge AI]            (Compreender como os modelos são acelerados para tempo real)
         ↓
[3. Out-of-Distribution]             (Compreender a incerteza e os limites de confiança do modelo)
         ↓
[4. Contextual Bandit & Q-Learning]  (Compreender a tomada de decisão com tentativa e erro sob incerteza)
         ↓
[5. Continual Learning & Memória]    (Compreender como evitar o esquecimento e recuperar dados passados)
         ↓
[6. Perceção Ativa & RL + LLMs]      (Compreender as fronteiras contemporâneas bioinspiradas e generativas)
```

---

## 8. Instruções para Agentes Autónomos de Inteligência Artificial

Qualquer agente de IA (como Antigravity, Claude Code, Cursor, GPT-4, etc.) instruído a atualizar ou interagir com este repositório **DEVE seguir as seguintes diretrizes**:

1. **Agnosticismo de Projeto:** Nunca misture discussões de implementações privadas ou de código interno nos ficheiros deste diretório. A pasta `docs/Literatura` é um módulo científico público e exportável.
2. **Preservação de Integridade:** Antes de inserir uma nova linha numa tabela de literatura, verifique a validade do DOI e assegure que os números citados no resumo constam explicitamente no texto original.
3. **Harmonia Estrutural:** Utilize sempre as diretrizes locais contidas no ficheiro `agent_literature_guidelines.md` presente na pasta de cada tópico antes de recolher novos artigos.
4. **Governação Terminológica:** Ao introduzir novos artigos cujos resumos utilizem conceitos chave ou arquiteturas centrais, verifique a conformidade com as definições do [`GLOSSARIO.md`](GLOSSARIO.md). Se o artigo fundar um conceito seminal novo, proponha a respetiva entrada no glossário respeitando as 5 secções padronizadas.
5. **Ausência Absoluta de Emojis:** É estritamente proibida a inserção de qualquer emoji ou pictograma em qualquer ficheiro deste diretório, mantendo a sobriedade e o rigor gráfico de um compêndio científico formal.
