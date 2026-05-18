# Documentação Técnica: Treino do Agente RL 128D (k-NN Bandit)

Este documento descreve detalhadamente o funcionamento e as decisões de engenharia inerentes ao ficheiro `training/train_rl_128d.py`. 

Este script atua como o motor de "treino" do sistema. No entanto, tratando-se de uma arquitetura de Reinforcement Learning baseada em instâncias (k-NN Bandit) em vez de redes neuronais, o processo não envolve Backpropagation ou ajuste de pesos paramétricos. O objetivo no pipeline do projeto é executar um processo de **Oracle Seeding** (Sementeira Oracular), injetando massivamente conhecimento pré-calculado na Memória Episódica do agente antes de este ser colocado em produção.

---

## 1. O Propósito e a Filosofia de Engenharia

Em sistemas híbridos onde uma CNN atua como extrator de *features* e um agente RL atua como decisor final, treinar os dois componentes em simultâneo desde o zero pode causar instabilidade caótica (*Moving Target Problem*).

Para mitigar isto, o `train_rl_128d.py` assume que a CNN já se encontra treinada (os seus pesos estão congelados). O pipeline de treino restringe-se a expor a CNN a milhares de imagens do *dataset* original e a guardar as suas extrações (os vetores 128D) no "caderno" do agente RL.
Isto ensina o agente a reconhecer, no espaço matemático de 128 dimensões, onde é que as *features* da CNN são fiáveis e onde é que falham miseravelmente sob condições de ruído extremo.

---

## 2. O Pipeline de Treino (Visão Global)

O fluxo de execução do script pode ser visualizado na seguinte tabela, que resume o ciclo de "Sementeira":

| Fase | Ação Principal | O que acontece nos bastidores |
| :--- | :--- | :--- |
| **1. Inicialização** | Carrega a CNN e o `KNNBanditAgent128D`. | A memória gráfica (VRAM) é alocada limitadamente para evitar estrangulamentos. O Agente k-NN é instanciado com a sua memória episódica vazia. |
| **2. Partição Hermética** | Divisão Estratificada 90/10. | Usa `train_test_split` para separar rigorosamente 90% para a memória episódica e 10% intactos para avaliar a precisão real (*Unseen Data*). |
| **3. Oracle Seeding** | Loop de 4 realizações apenas na partição de 90%. | As imagens ruidosas da partição de 90% são injetadas em lotes na CNN, que devolve os vetores latentes 128D e a sua previsão. |
| **4. Função de Recompensa** | Atribuição de Recompensas (+1 / -1). | O agente arquiva a "verdade absoluta" da partição de 90% e penaliza os erros cometidos pela CNN. |
| **5. Compilação Espacial** | Chamada do método `build_index()`. | A memória é compilada numa estrutura algorítmica veloz (*BallTree*/*KDTree*) em C/Numpy antes da avaliação nos 10% *unseen*. |
| **6. Persistência** | Exportação para disco via `save()`. | A memória consolidada é agrupada num único ficheiro `.npz`, preparando o agente para inferência real. |

---

## 3. Análise Detalhada do Código

### 3.1. Ingestão, Partição Hermética e Aumento do Dataset

Antes de extrair as *features* 128D, o sistema precisa de gerir a ingestão dos dados base de treino e assegurar uma avaliação fidedigna isenta de *Data Leakage*. O k-NN Bandit multiplica as instâncias de treino através de injeções sucessivas de ruído artificial, mas **apenas** sobre uma partição de treino isolada.

**1. Carregamento e Partição Estratificada (Train/Test Split):**
O dataset (ex: MNIST `x_train`) é carregado e normalizado. De imediato, o sistema aplica um `train_test_split` estratificado, reservando rigorosamente 90% para a memória do agente e 10% inalterados e ocultos para testar a precisão no final.

```python
logger.info("A criar partição hermética: 90% Sementeira / 10% Avaliação (Unseen Data)...")
x_train, x_test, y_train, y_test = train_test_split(
    x_train_full, y_train_full, 
    test_size=0.10, 
    random_state=42, 
    shuffle=True, 
    stratify=y_train_full
)
```

**2. Aumento de Dados (*Data Augmentation*) por Cenários:**
Para que o agente k-NN se torne robusto face a imagens degradadas que confundam a CNN, o código itera **exclusivamente sobre os 90% de treino** (`x_train`) aplicando manipulações globais. Percorrem-se 4 realizações: o bloco limpo (`0.0`), seguido do bloco com forte ruído Gaussiano (`0.6`) aplicado em três *runs* distintas.

```python
cenarios = [0.0, 0.6, 0.6, 0.6]

for r, intensidade in enumerate(cenarios):
    # A manipulação: Adiciona ruído apenas à partição de Sementeira (90%)
    x_ruido = adicionar_ruido_batch(x_train, intensidade) if intensidade > 0 else x_train
    
    # ... seguindo-se a extração de features e a atribuição de recompensas ...
```

**3. A Mecânica da Manipulação (Injeção de Ruído):**
A função `adicionar_ruido_batch` muta imagens vetorialmente. Adiciona ruído de distribuição normal a toda a matriz de uma assentada, usando `np.clip` para manter os píxeis entre `[0.0, 1.0]`.

```python
def adicionar_ruido_batch(imagens: np.ndarray, intensidade: float = 0.6) -> np.ndarray:
    ruido = np.random.normal(loc=0.0, scale=intensidade, size=imagens.shape)
    return np.clip(imagens + ruido, 0., 1.)
```

> [!NOTE]
> **A Aleatorização Estratificada Mitiga o Enviesamento:** Ao forçar o teste sobre os 10% reservados (`x_test`), garantimos que não existe avaliação baseada em dados já memorizados. O parâmetro `stratify=y_train_full` assegura o balanceamento entre classes e `shuffle=True` elimina tendências sequenciais do *dataset* cru, injetando efetivamente 216.000 experiências diversificadas na árvore (54k base + 162k ruidosas).

### 3.2. Otimização em Lote (Batch Processing) e Estrangulamentos

Uma das práticas mais críticas no processamento de imagens com GPUs é mitigar os gargalos de transferência (I/O Bottlenecks) entre a Memória RAM do sistema e a VRAM da placa gráfica. O script resolve isto através da função `extrair_features_128d_cnn`.

Em vez de enviar e pedir o processamento de uma imagem individual (o que paralisaria o barramento PCIe e desperdiçaria a capacidade massivamente paralela do GPU), a função agrupa os dados de entrada num *Batch* de tamanho 500.

```python
def extrair_features_128d_cnn(imagens: np.ndarray, cnn, batch_size: int = 500):
    todos_estados = []
    todas_preds = []
    for i in range(0, len(imagens), batch_size):
        batch = imagens[i : i + batch_size]
        batch_tensor = tf.convert_to_tensor(batch, dtype=tf.float32)
        outputs = cnn(batch_tensor)
        latent = outputs["latent_features"].numpy()
        probs = outputs["probabilities"].numpy()
        preds = np.argmax(probs, axis=1)
        todos_estados.append(latent)
        todas_preds.append(preds)
    return np.vstack(todos_estados), np.concatenate(todas_preds)
```

> [!TIP]
> **Eficiência Vetorial:** A utilização do `tf.convert_to_tensor` encapsulando uma grelha `[500, 28, 28, 1]`, aliada ao reagrupamento final através das operações ultravelozes em C do Numpy (`np.vstack` e `np.concatenate`), reduz drasticamente os ciclos de CPU. Isto garante que o GPU é alimentado ao máximo da sua capacidade computacional sem estrangular o barramento com pedidos unitários e incessantes de inferência.

### 3.3. A Gestão da Função de Recompensa (Reward Function)

No coração do Contextual Bandit reside o conceito de *Reward* (Recompensa). Ao analisar um estado (vetor 128D), o agente deve saber quão "rentável" foi uma determinada ação no passado. No ciclo `cenarios`, o código molda os valores da função de recompensa diretamente através da observação das falhas e acertos da CNN.

```python
for r, intensidade in enumerate(cenarios):
    # ... (preparação de imagens com ruído e extração de estados) ...
    
    # 1. Recompensas Positivas (Oracle Truth)
    agent.add_experience_batch(estados, y_train, np.ones(len(y_train)))
    
    # 2. Recompensas Negativas (Penalizar os Erros da CNN)
    erros = preds_cnn != y_train
    n_erros = np.sum(erros)
    if n_erros > 0:
        agent.add_experience_batch(
            estados[erros], preds_cnn[erros], np.full(n_erros, -1.0)
        )
```

**Como isto molda a árvore espacial:**
1. A primeira instrução diz ao agente: "A ação ótima para as imagens fornecidas é a classe real (`y_train`), por isso regista isto na tua memória recebendo uma recompensa incondicional de `+1.0`".
2. A segunda instrução, que utiliza operações vetoriais Booleanas (`erros = preds_cnn != y_train`) para uma velocidade extrema, filtra onde é que a CNN tomou decisões erradas devido ao ruído. O código obriga o agente a memorizar essas más decisões, penalizando-as rigorosamente com uma recompensa de `-1.0`.

> [!NOTE]
> **Imunidade Adversarial:** Guardar experiências negativas é um passo brilhante em RL. Se um novo estado (durante a fase de produção) for extremamente ruidoso e muito semelhante às falhas passadas da CNN, os "vizinhos negativos" na árvore espacial suprimirão matematicamente a pontuação da classe incorreta que a CNN proporia.

### 3.4. O Ciclo Simbiótico: `add_experience_batch`, `build_index` e `save`

Uma característica notável da arquitetura (também espelhada no documento `knn_bandit_agent_128d.md`) é que as invocações de `add_experience_batch` **não provocam reconstruções dispendiosas das árvores espaciais**. Estes dados são apenas adicionados (`append`) às listas temporárias Python em RAM.

Para cimentar definitivamente a nova sabedoria absorvida do ciclo em árvore (KDTree/BallTree) pronta a ser alvo de pesquisas ultrarrápidas, é fulcral chamar explicitamente a função `build_index()`. 

```python
    logger.info("A construir o índice k-NN...")
    agent.build_index()
```

> [!IMPORTANT]
> **Sincronização de Estados:** Só depois da invocação do `build_index()` é que as experiências geradas pelo *Oracle Seeding* passam formalmente a existir para as inferências k-NN do agente. Qualquer tentativa de pedir inferências ao agente antes desta chamada resultaria numa previsão instável baseada num índice desatualizado ou inexistente.

Finalmente, a serialização dos dados estabilizados:

```python
    caminho = os.path.join("outputs", "knn_memory_bank_128d.npz")
    agent.save(caminho)
```

O método `save` compacta o índice perfeitamente reconstruído e cristaliza-o no disco rígido na pasta `outputs`. É isto que permite que o sistema híbrido reinicie em produção amanhã ou daqui a 1 ano, recuperando em escassos segundos uma árvore espacial que pode ter levado horas a ser construída.

---

## 4. Conclusão

O script `train_rl_128d.py` destaca-se pela sua eficiência de engenharia. Em vez de se arrastar com simulações morosas de *Deep Reinforcement Learning*, impõe um paradigma determinístico e limpo:

1. **Performance Absoluta**: A extração é operada num formato em lote que suprime os gargalos entre Memória Principal e GPU, gerando rapidamente centenas de milhares de observações 128D num ambiente altamente vectorizado.
2. **Estabilidade de Memória**: A dissociação arquitetural entre as adições de memória (`add_experience_batch`) e a recompilação forçada das partições geométricas do espaço de decisão (`build_index`) permite absorver variações substanciais no *dataset* (Limpo + 3x Ruído) de uma só vez, prevenindo instabilidade sistémica.
3. **Mecanismo de Defesa Robusto**: Ao mapear não apenas as vitórias com lucro positivo, mas deliberadamente castigar (`-1.0`) a zona do espaço latente 128D onde a visão base falha repetidamente, este módulo consagra um sistema que não cometerá o mesmo erro duas vezes. O código transcende o treino e atua efetivamente como a imunização nativa e permanente contra incertezas e falhas catastróficas da rede base CNN.

---

## 5. Exemplo de Execução (Console Log)

Ao executar o script após as alterações de mitigação de *Data Leakage*, o comportamento observado na consola demonstra a eficácia do agente k-NN sobre a avaliação hermética com dados nunca antes vistos (*Unseen Data*).

Observamos que sob condições de forte ruído (0.6), a arquitetura CNN base tem a sua precisão severamente degradada para **32.0%**. Contudo, ao ativarmos a nossa Memória Episódica 128D, o sistema é capaz de intercetar essas falhas catastróficas, recuperando a precisão para impressionantes **88.2%**.

```text
A instanciar o Agente Especialista k-NN Bandit 128D...
A preparar o Ambiente de Treino...
A criar partição hermética: 90% Sementeira / 10% Avaliação (Unseen Data)...
  Partição de Sementeira (Memória): 54000 amostras
  Partição de Avaliação (Teste): 6000 amostras

==================================================
ORACLE SEEDING (Apenas na partição de 90%)
==================================================
  Realização 1/4 (Ruído 0.0)...
    +54,000 positivas, +1,308 negativas | CNN acc: 97.6%

  Realização 2/4 (Ruído 0.6)...
    +54,000 positivas, +36,711 negativas | CNN acc: 32.0%

  Realização 3/4 (Ruído 0.6)...
    +54,000 positivas, +36,742 negativas | CNN acc: 32.0%

  Realização 4/4 (Ruído 0.6)...
    +54,000 positivas, +36,720 negativas | CNN acc: 32.0%

A construir o índice k-NN...
Índice k-NN (128D) construído com 327481 experiências (k=30). PCA=False

==================================================
AVALIAÇÃO REAL (Apenas na partição de 10% Unseen)
==================================================
  [Limpo] CNN: 97.6% | k-NN 128D: 97.4%
  [Ruído 0.3] CNN: 75.6% | k-NN 128D: 86.4%
  [Ruído 0.6] CNN: 32.0% | k-NN 128D: 88.2%

  Memória Total: 327,481
  % Positivas: 66.0%

Concluído! Memória guardada em: outputs\knn_memory_bank_128d.npz
```
