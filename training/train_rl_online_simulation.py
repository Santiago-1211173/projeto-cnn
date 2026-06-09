"""
Simulação Online Progressiva do Agente RL (k-NN Bandit 128D).

Conceito:
  - 10% do dataset MNIST é reservado para "sementeira" (Oracle Seeding).
  - 90% é a partição de avaliação (completamente disjunta, zero data leakage).
  - Os 10% são divididos em 10 lotes incrementais.
  - Após ingerir cada lote, o agente é avaliado na partição gigante de 90%.
  - No final, é gerada uma curva de aprendizagem de qualidade de publicação.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from src.models.custom_cnn import RawModel
from src.models.knn_bandit_agent_128d import KNNBanditAgent128D
from src.data.loader import load_mnist_raw

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# ── Hiperparâmetros da Simulação ──────────────────────────────────────────────
NUM_LOTES          = 10        # Número de chunks incrementais
NOISE_EVAL         = 0.6       # Intensidade de ruído na avaliação
CENARIOS_RUIDO     = [0.0, 0.2, 0.4, 0.6, 0.8]  # Noise scenarios for Oracle Seeding
BATCH_SIZE_CNN     = 1024      # Batch size para extração de features (GPU-otimizado)
BATCH_SIZE_RL_EVAL = 2048      # Batch size para avaliação do agente RL
RANDOM_SEED        = 42
# ──────────────────────────────────────────────────────────────────────────────


def adicionar_ruido_batch(imagens: np.ndarray, intensidade: float) -> np.ndarray:
    """Adiciona ruído Gaussiano a um batch de imagens."""
    ruido = np.random.normal(loc=0.0, scale=intensidade, size=imagens.shape)
    return np.clip(imagens + ruido, 0.0, 1.0)


def extrair_features_128d(imagens: np.ndarray, cnn, batch_size: int = BATCH_SIZE_CNN):
    """Extrai features latentes de 128D e previsões da CNN em batches otimizados."""
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


def avaliar_agente_no_eval(agent, features_eval: np.ndarray, labels_eval: np.ndarray,
                           batch_size: int = BATCH_SIZE_RL_EVAL) -> float:
    """Avalia o agente RL na partição de avaliação em batches otimizados."""
    acertos = 0
    total = len(labels_eval)
    for i in range(0, total, batch_size):
        feats_batch = features_eval[i : i + batch_size]
        labels_batch = labels_eval[i : i + batch_size]
        preds_rl = agent.get_action_batch(feats_batch, epsilon=0.0)
        acertos += int(np.sum(preds_rl == labels_batch))
    return acertos / total


def gerar_grafico(historico: list, caminho_saida: str):
    """Gera um gráfico de qualidade de publicação da curva de aprendizagem."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    tamanhos = [h[0] for h in historico]
    acuracias = [h[1] * 100 for h in historico]

    # ── Cores e Estilo ────────────────────────────────────────────────────────
    COR_FUNDO      = '#0D1117'
    COR_AREA       = '#161B22'
    COR_GRADE      = '#21262D'
    COR_TEXTO      = '#C9D1D9'
    COR_TEXTO_DIM  = '#8B949E'
    COR_LINHA      = '#58A6FF'
    COR_GRADIENTE  = '#1F6FEB'
    COR_PONTO      = '#79C0FF'
    COR_BORDA_PT   = '#388BFD'
    COR_ANOTACAO   = '#F0F6FC'

    fig, ax = plt.subplots(figsize=(14, 7), dpi=200)
    fig.patch.set_facecolor(COR_FUNDO)
    ax.set_facecolor(COR_AREA)

    # ── Gradient Fill ─────────────────────────────────────────────────────────
    ax.fill_between(tamanhos, acuracias, alpha=0.15, color=COR_GRADIENTE, zorder=1)

    # ── Linha Principal ───────────────────────────────────────────────────────
    ax.plot(tamanhos, acuracias,
            color=COR_LINHA, linewidth=2.8, zorder=3,
            marker='o', markersize=9,
            markerfacecolor=COR_PONTO, markeredgecolor=COR_BORDA_PT,
            markeredgewidth=1.8)

    # ── Anotações nos pontos ──────────────────────────────────────────────────
    for i, (x, y) in enumerate(zip(tamanhos, acuracias)):
        offset_y = 12 if i % 2 == 0 else -18
        ax.annotate(f'{y:.1f}%',
                    xy=(x, y), xytext=(0, offset_y),
                    textcoords='offset points', ha='center', va='bottom',
                    fontsize=9, fontweight='bold', color=COR_ANOTACAO,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=COR_FUNDO,
                              edgecolor=COR_GRADE, alpha=0.85))

    # ── Eixos e Títulos ───────────────────────────────────────────────────────
    ax.set_xlabel('Episodic Memory Size', fontsize=13, color=COR_TEXTO,
                  fontweight='bold', labelpad=12)
    ax.set_ylabel('RL Accuracy (%)', fontsize=13, color=COR_TEXTO,
                  fontweight='bold', labelpad=12)
    ax.set_title('RL Specialist — Online Learning Curve\n'
                 f'Evaluation on {len(tamanhos)} incremental seeding rounds  |  '
                 f'Test Noise = {NOISE_EVAL}',
                 fontsize=15, color=COR_ANOTACAO, fontweight='bold', pad=20)

    # ── Grid e Moldura ────────────────────────────────────────────────────────
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.4, color=COR_GRADE)
    ax.tick_params(colors=COR_TEXTO_DIM, labelsize=10)
    for spine in ax.spines.values():
        spine.set_color(COR_GRADE)
        spine.set_linewidth(0.8)

    # Formatar eixo X com separador de milhares
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x):,}'))

    # ── Informação de Contexto ────────────────────────────────────────────────
    mem_final = tamanhos[-1]
    acc_final = acuracias[-1]
    acc_inicial = acuracias[0]
    delta = acc_final - acc_inicial

    info_text = (f'Δ Accuracy: +{delta:.1f}pp\n'
                 f'Final Memory: {mem_final:,}\n'
                 f'Peak Accuracy: {max(acuracias):.1f}%')
    ax.text(0.02, 0.97, info_text,
            transform=ax.transAxes, fontsize=10, color=COR_TEXTO_DIM,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor=COR_FUNDO,
                      edgecolor=COR_GRADE, alpha=0.9))

    plt.tight_layout(pad=2.0)

    os.makedirs(os.path.dirname(caminho_saida), exist_ok=True)
    fig.savefig(caminho_saida, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)
    logger.info(f"\n  Gráfico guardado em: {caminho_saida}")


def main():
    # ── 1. Configurar Hardware (GPU) ──────────────────────────────────────────
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logger.info(f"  GPU(s) detectada(s): {len(gpus)}")
    else:
        logger.info("  Sem GPU — a usar CPU.")

    # ── 2. Carregar a CNN (Extrator de Features) ──────────────────────────────
    logger.info("\nA carregar a CNN (Extrator de Features 128D)...")
    cnn = RawModel()
    ckpt = tf.train.Checkpoint(model=cnn)
    latest_ckpt = tf.train.latest_checkpoint(os.path.join("outputs", "checkpoints"))
    if not latest_ckpt:
        logger.error("ERRO: Checkpoint não encontrado em outputs/checkpoints/")
        return
    ckpt.restore(latest_ckpt).expect_partial()
    logger.info("  Pesos restaurados com sucesso.")

    # ── 3. Instanciar o Agente RL ─────────────────────────────────────────────
    agent = KNNBanditAgent128D(k=30, n_actions=10, use_pca=False)

    # ── 4. Carregar e Preparar o Dataset ──────────────────────────────────────
    logger.info("\nA carregar o dataset MNIST...")
    x_full, y_full = load_mnist_raw(os.path.join("data", "MNIST", "raw"), kind='train')
    x_full = x_full.astype(np.float32) / 255.0

    logger.info("A criar partição hermética: 10% Sementeira / 90% Avaliação...")
    x_seed, x_eval, y_seed, y_eval = train_test_split(
        x_full, y_full,
        test_size=0.90,
        random_state=RANDOM_SEED,
        shuffle=True,
        stratify=y_full
    )
    logger.info(f"  Partição de Sementeira (10%): {len(x_seed):,} amostras")
    logger.info(f"  Partição de Avaliação  (90%): {len(x_eval):,} amostras")

    # ── 5. Preparar os Lotes Incrementais ─────────────────────────────────────
    indices_seed = np.arange(len(x_seed))
    np.random.seed(RANDOM_SEED)
    np.random.shuffle(indices_seed)
    lotes = np.array_split(indices_seed, NUM_LOTES)

    logger.info(f"  Divididos em {NUM_LOTES} lotes (~{len(lotes[0])} amostras por lote)")

    # ── 6. Pré-calcular Features de Avaliação (uma única vez) ─────────────────
    logger.info(f"\nA pré-calcular features de avaliação (ruído = {NOISE_EVAL})...")
    x_eval_noisy = adicionar_ruido_batch(x_eval, NOISE_EVAL) if NOISE_EVAL > 0 else x_eval
    features_eval, _ = extrair_features_128d(x_eval_noisy, cnn)
    logger.info(f"  Features de avaliação: {features_eval.shape}")

    # ── 7. Loop de Treino Progressivo ─────────────────────────────────────────
    historico = []  # [(memory_size, rl_accuracy), ...]

    logger.info("\n" + "=" * 60)
    logger.info("  SIMULAÇÃO ONLINE — SEMENTEIRA PROGRESSIVA")
    logger.info("=" * 60)

    for lote_idx, indices_lote in enumerate(lotes):
        lote_num = lote_idx + 1
        x_lote = x_seed[indices_lote]
        y_lote = y_seed[indices_lote]

        logger.info(f"\n── Lote {lote_num}/{NUM_LOTES} "
                    f"({len(indices_lote)} amostras) ──────────────────────")

        # 7a. Oracle Seeding: gerar cenários de ruído para o lote actual
        for r, intensidade in enumerate(CENARIOS_RUIDO):
            x_ruido = adicionar_ruido_batch(x_lote, intensidade) if intensidade > 0 else x_lote

            estados, preds_cnn = extrair_features_128d(x_ruido, cnn)

            # Experiências positivas (label oracle, reward +1.0)
            agent.add_experience_batch(estados, y_lote, np.ones(len(y_lote)))

            # Experiências negativas (CNN errou, reward -1.0)
            erros = preds_cnn != y_lote
            n_erros = int(np.sum(erros))
            if n_erros > 0:
                agent.add_experience_batch(
                    estados[erros], preds_cnn[erros], np.full(n_erros, -1.0)
                )

            acc_cnn = np.mean(preds_cnn == y_lote) * 100
            logger.info(f"    Ruído {intensidade:.1f} → +{len(y_lote):,} pos, "
                        f"+{n_erros:,} neg | CNN acc: {acc_cnn:.1f}%")

        # 7b. Reconstruir o índice k-NN com toda a memória acumulada
        logger.info(f"  A reconstruir índice k-NN (memória: {agent.memory_size:,})...")
        agent.build_index()

        # 7c. Avaliar na partição gigante de 90%
        logger.info("  A avaliar na partição de 90% (unseen)...")
        acc_rl = avaliar_agente_no_eval(agent, features_eval, y_eval)
        historico.append((agent.memory_size, acc_rl))

        logger.info(f"  ✓ Memória: {agent.memory_size:,} | "
                    f"RL Accuracy: {acc_rl * 100:.2f}%")

    # ── 8. Resumo Final ───────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("  RESULTADOS DA SIMULAÇÃO")
    logger.info("=" * 60)
    logger.info(f"  {'Lote':<6} {'Memória':>10} {'RL Acc (%)':>12}")
    logger.info("  " + "-" * 30)
    for i, (mem, acc) in enumerate(historico):
        logger.info(f"  {i+1:<6} {mem:>10,} {acc * 100:>11.2f}%")

    # ── 9. Gerar Gráfico de Publicação ────────────────────────────────────────
    caminho_grafico = os.path.join("visualizations", "rl_online_learning_curve.png")
    gerar_grafico(historico, caminho_grafico)

    logger.info("\nSimulação concluída com sucesso!")


if __name__ == "__main__":
    main()
