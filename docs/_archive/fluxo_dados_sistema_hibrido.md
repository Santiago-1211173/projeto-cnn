# Fluxo de Dados no Sistema Híbrido: CNN & k-NN Bandit RL

Este documento explica o fluxo de dados do sistema híbrido de Visão Computacional implementado neste repositório. O foco principal é detalhar o ciclo de vida da informação desde a entrada de uma imagem até à decisão final, clarificando exatamente o que acontece quando o sistema transfere o controlo da Rede Neuronal Convolucional (CNN) para o Agente de Reinforcement Learning (k-NN Bandit).

---

## 1. Visão Geral da Arquitetura

O sistema não confia cegamente num único modelo. Ele utiliza uma arquitetura de "defesa em profundidade", combinando dois paradigmas distintos:
1. **Modelo Paramétrico (CNN):** Treinado tradicionalmente para classificar imagens limpas e atuar como um excelente extrator de características (*feature extractor*).
2. **Modelo Baseado em Instâncias (k-NN Bandit RL):** Um agente de Reinforcement Learning dotado de uma Memória Episódica, desenhado para atuar como um "especialista de resgate" quando a CNN se depara com dados altamente ruidosos ou incertos.

---

## 2. Diagrama do Fluxo de Dados (Inscrição / Produção)

O diagrama abaixo ilustra o caminho que os dados percorrem durante a inferência (quando o sistema está em produção a avaliar novas imagens).

```text
[ Imagem de Entrada ]
         |
         v
[ CNN: Extrator de Features ]
         |
         |---> (1. Processamento na CNN)
         |     |--> [ Vetor Latente 128D ]
         |     |--> [ Probabilidades das Classes ]
         |
         v
{ Mecanismo Árbitro } (Avalia a Incerteza)
         |
         |-- (Elevada Confiança)
         |   '--> [ Decisão Final: Classe prevista pela CNN ]
         |
         |-- (Incerteza ou Ruído)
         '--> [ Passagem de Testemunho para o RL ]
                  |
                  |---> (2. Agente k-NN Bandit)
                  |     |--> [ Recebe Estado: Vetor 128D ]
                  |     |--> [ PCA: Redução Opcional para 48D ]
                  |     |--> [ Procura Espacial: BallTree ou KDTree ]
                  |     |--> [ Identificação dos 'k' Vizinhos ]
                  |     |--> [ Ponderação: Inverse Distance Weighting ]
                  |     '--> [ Ação com Maior Recompensa ]
                  |
                  v
         [ Decisão Final: Classe prevista pelo RL ]
```

---

## 3. A Passagem de Testemunho: O que é transmitido?

O momento crítico no fluxo de dados ocorre quando o sistema decide não confiar na classificação direta da CNN (por exemplo, devido a um limiar de distância de Mahalanobis ou deteção de baixa confiança). 

Nesse momento, **a imagem original é descartada das contas do RL**. O que transita para o Agente de Reinforcement Learning é exclusivamente o **Vetor Latente de 128 Dimensões (128D)**.

> [!IMPORTANT]
> **Porquê o Vetor 128D?** 
> A penúltima camada da CNN produz um vetor 128D que encapsula a representação abstrata e espacial da imagem (as *features*). Em vez de tentar processar píxeis em bruto, o agente RL usa este vetor porque ele representa o "estado cognitivo" da CNN. O agente não olha para a imagem; ele olha para *a forma como a CNN interpretou a imagem*.

---

## 4. O Fluxo de Dados Interno no Agente RL

Uma vez que o vetor 128D entra no método `get_action(state_128d)` do `KNNBanditAgent128D`, o fluxo de dados algorítmico segue passos rigorosos de alta performance:

### Passo 1: Modelação do Estado (Compressão PCA)
Se o agente estiver configurado para usar PCA (para mitigar a "Maldição da Dimensionalidade"), o vetor de 128 dimensões é matematicamente projetado num subespaço menor (ex: 48 dimensões). Esta projeção utiliza a mesma matriz de transformação gerada durante a fase de treino.

### Passo 2: Procura Espacial Rápida
O vetor comprimido (query) atua como um "sonar". É injetado no motor de busca espacial (`NearestNeighbors` usando `BallTree` ou `KDTree`). O algoritmo pesquisa na Memória Episódica (que foi pré-carregada para a RAM) e devolve as **distâncias** e os **índices** das `k` experiências passadas mais semelhantes a este estado.

### Passo 3: Ponderação e Votação (Inverse Distance Weighting)
Os vizinhos encontrados trazem consigo as ações (0-9) que tomaram no passado e as recompensas (+1.0 ou -1.0) que receberam. 
O agente não faz uma votação simples. Ele calcula um peso para cada vizinho: `weight = 1.0 / (distancia + 1e-8)`.
- Vizinhos geometricamente muito próximos ao estado atual recebem um peso massivo.
- Vizinhos mais distantes (marginais) têm o seu peso muito reduzido.

### Passo 4: Agregação da Recompensa Esperada
O fluxo agrupa os vizinhos pela ação que tomaram e soma as suas recompensas multiplicadas pelo peso. O resultado é um array com as 10 expectativas de lucro (uma para cada dígito). A ação com o valor matemático mais alto é extraída via `np.argmax()` e devolvida como a decisão final do sistema.

---

## 5. A Relação com a Fase de Treino (Oracle Seeding)

Para que o fluxo de dados acima tenha sucesso na vida real, o Agente k-NN precisava de uma memória rica. O fluxo de dados durante a "Sementeira" (conforme descrito no `train_rl_128d.md`) funciona da seguinte forma para alimentar este pipeline:

1. **Geração de Estados:** O sistema passa centenas de milhares de imagens (limpas e com ruído injetado) pela CNN, apenas para recolher os vetores 128D e as previsões base da CNN.
2. **Injeção da Verdade (+1):** O agente memoriza qual seria a resposta correta para cada vetor 128D, associando-lhe uma recompensa positiva.
3. **Imunização contra Falhas (-1):** Crucialmente, onde a CNN falha (devido ao ruído), o fluxo de treino obriga o agente a memorizar o vetor 128D causador do erro juntamente com a previsão incorreta da CNN, penalizando-o com `-1.0`.

**Conclusão do Fluxo:**
Quando o sistema híbrido passa a ação para o RL, ele está efetivamente a pedir ao agente para verificar, na sua vasta base de dados espacial, se este vetor 128D provém de uma zona "segura" de previsões da CNN ou de uma zona mapeada com erros (recompensas negativas). As recompensas negativas armazenadas "abafam" a resposta incorreta que a CNN ia dar, permitindo ao RL resgatar a previsão e devolver a classe correta.
