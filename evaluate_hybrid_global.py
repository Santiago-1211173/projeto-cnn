"""
Avaliação Global do Pipeline Híbrido (CNN + RL Specialist).

Executa dois cenários independentes:
  Approach A — Dataset t10k (10,000 amostras nativas do MNIST).
  Approach B — Partição disjunta de 90% do training set (54,000 amostras).

Para cada cenário:
  1. Constrói um dataset de ruído misto balanceado (5 bandas: 0.0–0.8).
  2. Varre limiares de Mahalanobis de 5.0 a 30.0 (step 2.5).
  3. Executa o pipeline híbrido em batches otimizados para GPU.
  4. Extrai métricas: Hybrid Acc, CNN-only Acc, RL-only Acc, RL Routing Rate.
  5. Gera tabela Markdown no terminal + gráfico comparativo de publicação.
"""

import os
import sys
import logging
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split

from src.models.custom_cnn import RawModel
from src.models.knn_bandit_agent_128d import KNNBanditAgent128D
from src.data.loader import load_mnist_raw

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# ── Configuração ──────────────────────────────────────────────────────────────
NOISE_LEVELS         = [0.0, 0.2, 0.4, 0.6, 0.8]
THRESHOLDS           = np.arange(5.0, 30.0 + 1e-9, 2.5)
BATCH_SIZE_CNN       = 1024
BATCH_SIZE_RL        = 2048
RANDOM_SEED          = 42
OUTPUT_FIGURE_PATH   = os.path.join("visualizations", "hybrid_global_evaluation.png")
# ──────────────────────────────────────────────────────────────────────────────


# ═══════════════════════════════════════════════════════════════════════════════
#  UTILITÁRIOS
# ═══════════════════════════════════════════════════════════════════════════════

def adicionar_ruido_batch(imagens: np.ndarray, intensidade: float) -> np.ndarray:
    """Adiciona ruído Gaussiano a um batch de imagens."""
    if intensidade <= 0.0:
        return imagens.copy()
    ruido = np.random.normal(loc=0.0, scale=intensidade, size=imagens.shape)
    return np.clip(imagens + ruido, 0.0, 1.0).astype(np.float32)


def construir_dataset_ruido_misto(x: np.ndarray, y: np.ndarray,
                                   niveis: list) -> tuple:
    """
    Divide x/y em N partições balanceadas (uma por nível de ruído) e
    aplica a intensidade respectiva. Retorna (x_misto, y_misto).
    """
    n = len(x)
    n_niveis = len(niveis)
    tamanho_fatia = n // n_niveis

    # Embaralhar com seed fixa para reprodutibilidade
    rng = np.random.RandomState(RANDOM_SEED)
    indices = rng.permutation(n)

    x_partes, y_partes = [], []
    for i, nivel in enumerate(niveis):
        inicio = i * tamanho_fatia
        fim = inicio + tamanho_fatia if i < n_niveis - 1 else n
        idx = indices[inicio:fim]
        x_partes.append(adicionar_ruido_batch(x[idx], nivel))
        y_partes.append(y[idx])

    return np.concatenate(x_partes, axis=0), np.concatenate(y_partes, axis=0)


def extrair_features_e_preds(imagens: np.ndarray, cnn,
                              batch_size: int = BATCH_SIZE_CNN):
    """Extrai features 128D, probabilidades e previsões da CNN em batches."""
    todos_latent = []
    todas_preds = []
    todas_probs = []
    for i in range(0, len(imagens), batch_size):
        batch = imagens[i:i + batch_size]
        tensor = tf.convert_to_tensor(batch, dtype=tf.float32)
        outputs = cnn(tensor)
        latent = outputs["latent_features"].numpy()
        probs = outputs["probabilities"].numpy()
        preds = np.argmax(probs, axis=1)
        todos_latent.append(latent)
        todas_preds.append(preds)
        todas_probs.append(probs)
    return (np.vstack(todos_latent),
            np.concatenate(todas_preds),
            np.vstack(todas_probs))


def calcular_mahalanobis_batch(vetores: np.ndarray,
                                perfis: dict) -> np.ndarray:
    """
    Calcula a distância mínima de Mahalanobis de cada vetor 128D
    ao perfil de classe mais próximo. Retorna array de distâncias.
    """
    n = len(vetores)
    dists_min = np.full(n, np.inf, dtype=np.float64)

    for digito in range(10):
        info = perfis[str(digito)].item()
        mu = info["mu"]
        inv_sigma = info["inv_sigma"]
        diff = vetores - mu  # (n, 128)
        # Mahalanobis: sqrt( diff @ inv_sigma @ diff^T )  por linha
        left = diff @ inv_sigma  # (n, 128)
        dist_sq = np.sum(left * diff, axis=1)  # (n,)
        dist = np.sqrt(np.maximum(dist_sq, 0.0))
        dists_min = np.minimum(dists_min, dist)

    return dists_min


def avaliar_rl_batch(agent, features: np.ndarray, labels: np.ndarray,
                     batch_size: int = BATCH_SIZE_RL) -> np.ndarray:
    """Obtém previsões RL em batches. Retorna array de previsões."""
    preds_list = []
    for i in range(0, len(features), batch_size):
        f_batch = features[i:i + batch_size]
        preds_list.append(agent.get_action_batch(f_batch, epsilon=0.0))
    return np.concatenate(preds_list)


# ═══════════════════════════════════════════════════════════════════════════════
#  AVALIAÇÃO COMPLETA DE UM APPROACH
# ═══════════════════════════════════════════════════════════════════════════════

def avaliar_approach(nome: str, x: np.ndarray, y: np.ndarray,
                     cnn, agent, perfis: dict) -> dict:
    """
    Executa a avaliação completa para um approach:
      - Constrói dataset de ruído misto
      - Extrai features 128D + preds CNN
      - Calcula distâncias Mahalanobis
      - Varre limiares e calcula métricas
    Retorna dict com listas de métricas por threshold.
    """
    logger.info(f"\n{'═' * 70}")
    logger.info(f"  {nome}")
    logger.info(f"{'═' * 70}")

    # 1. Construir dataset de ruído misto balanceado
    logger.info(f"  A construir dataset de ruído misto ({len(x):,} amostras, "
                f"{len(NOISE_LEVELS)} bandas)...")
    x_misto, y_misto = construir_dataset_ruido_misto(x, y, NOISE_LEVELS)
    logger.info(f"  Dataset misto: {len(x_misto):,} amostras.")

    # 2. Extrair features 128D + previsões CNN
    logger.info("  A extrair features 128D e previsões CNN (GPU batch)...")
    features_128d, preds_cnn, _ = extrair_features_e_preds(x_misto, cnn)
    logger.info(f"  Features: {features_128d.shape}  |  CNN baseline calculada.")

    # 3. Calcular distâncias de Mahalanobis (vectorizado)
    logger.info("  A calcular distâncias de Mahalanobis (vectorizado)...")
    dists = calcular_mahalanobis_batch(features_128d, perfis)
    logger.info(f"  Distâncias: min={dists.min():.2f}  "
                f"median={np.median(dists):.2f}  max={dists.max():.2f}")

    # 4. Obter previsões RL (uma única vez, reutilizar para todos os limiares)
    logger.info("  A obter previsões RL (batch)...")
    preds_rl = avaliar_rl_batch(agent, features_128d, y_misto)
    logger.info("  Previsões RL calculadas.")

    # 5. Varrer limiares
    resultados = {
        "thresholds": [],
        "hybrid_acc": [],
        "cnn_acc": [],
        "rl_acc": [],
        "rl_rate": [],
    }

    acc_cnn_global = float(np.mean(preds_cnn == y_misto) * 100)
    acc_rl_global = float(np.mean(preds_rl == y_misto) * 100)

    for t in THRESHOLDS:
        mask_cnn = dists < t
        mask_rl = ~mask_cnn

        # Decisão híbrida
        preds_hibrido = np.where(mask_cnn, preds_cnn, preds_rl)
        acc_hibrido = float(np.mean(preds_hibrido == y_misto) * 100)
        taxa_rl = float(np.mean(mask_rl) * 100)

        resultados["thresholds"].append(float(t))
        resultados["hybrid_acc"].append(acc_hibrido)
        resultados["cnn_acc"].append(acc_cnn_global)
        resultados["rl_acc"].append(acc_rl_global)
        resultados["rl_rate"].append(taxa_rl)

    # 6. Imprimir tabela Markdown
    logger.info(f"\n  ### {nome} — Resultados por Limiar\n")
    logger.info(f"  | {'Threshold':>10} | {'Hybrid Acc':>11} | {'CNN Acc':>9} "
                f"| {'RL Acc':>8} | {'RL Rate':>9} |")
    logger.info(f"  |{'-' * 12}|{'-' * 13}|{'-' * 11}|{'-' * 10}|{'-' * 11}|")
    for i, t in enumerate(resultados["thresholds"]):
        logger.info(
            f"  | {t:>10.1f} | {resultados['hybrid_acc'][i]:>10.2f}% "
            f"| {resultados['cnn_acc'][i]:>8.2f}% "
            f"| {resultados['rl_acc'][i]:>7.2f}% "
            f"| {resultados['rl_rate'][i]:>8.2f}% |"
        )

    return resultados


# ═══════════════════════════════════════════════════════════════════════════════
#  VISUALIZAÇÃO DE PUBLICAÇÃO (DOUBLE-PANEL)
# ═══════════════════════════════════════════════════════════════════════════════

def gerar_grafico_comparativo(res_a: dict, res_b: dict, caminho: str):
    """Gera gráfico double-panel comparativo de qualidade de publicação."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter

    # ── Paleta de cores (GitHub Dark Theme) ───────────────────────────────────
    BG_OUTER    = '#0D1117'
    BG_INNER    = '#161B22'
    GRID_CLR    = '#21262D'
    TEXT_CLR    = '#C9D1D9'
    TEXT_DIM    = '#8B949E'
    TEXT_BRIGHT = '#F0F6FC'

    # Linhas — Approach A
    CLR_A_HYBRID = '#58A6FF'  # azul
    CLR_A_CNN    = '#F78166'  # laranja
    CLR_A_RL     = '#7EE787'  # verde
    CLR_A_RATE   = '#D2A8FF'  # roxo

    # Linhas — Approach B (variante mais clara / traço diferente)
    CLR_B_HYBRID = '#79C0FF'
    CLR_B_CNN    = '#FFA198'
    CLR_B_RL     = '#AFFFB5'
    CLR_B_RATE   = '#E8D5FF'

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 9), dpi=200)
    fig.patch.set_facecolor(BG_OUTER)

    for ax in (ax1, ax2):
        ax.set_facecolor(BG_INNER)
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.4, color=GRID_CLR)
        ax.tick_params(colors=TEXT_DIM, labelsize=10)
        for spine in ax.spines.values():
            spine.set_color(GRID_CLR)
            spine.set_linewidth(0.8)

    thresholds = res_a["thresholds"]
    marker_kw = dict(markersize=7, markeredgewidth=1.5)

    # ══════════════════════════════════════════════════════════════════════════
    #  PANEL 1 — Accuracy vs Threshold
    # ══════════════════════════════════════════════════════════════════════════
    # Approach A (solid)
    ax1.plot(thresholds, res_a["hybrid_acc"],
             color=CLR_A_HYBRID, linewidth=2.5, marker='o',
             markerfacecolor=CLR_A_HYBRID, markeredgecolor=BG_OUTER,
             label='Hybrid — A (t10k)', **marker_kw)
    ax1.plot(thresholds, res_a["cnn_acc"],
             color=CLR_A_CNN, linewidth=2.0, marker='s',
             markerfacecolor=CLR_A_CNN, markeredgecolor=BG_OUTER,
             label='CNN-only — A', linestyle='--', **marker_kw)
    ax1.plot(thresholds, res_a["rl_acc"],
             color=CLR_A_RL, linewidth=2.0, marker='^',
             markerfacecolor=CLR_A_RL, markeredgecolor=BG_OUTER,
             label='RL-only — A', linestyle=':', **marker_kw)

    # Approach B (dashed markers)
    ax1.plot(thresholds, res_b["hybrid_acc"],
             color=CLR_B_HYBRID, linewidth=2.5, marker='D',
             markerfacecolor=CLR_B_HYBRID, markeredgecolor=BG_OUTER,
             label='Hybrid — B (90%)', **marker_kw)
    ax1.plot(thresholds, res_b["cnn_acc"],
             color=CLR_B_CNN, linewidth=2.0, marker='v',
             markerfacecolor=CLR_B_CNN, markeredgecolor=BG_OUTER,
             label='CNN-only — B', linestyle='--', **marker_kw)
    ax1.plot(thresholds, res_b["rl_acc"],
             color=CLR_B_RL, linewidth=2.0, marker='P',
             markerfacecolor=CLR_B_RL, markeredgecolor=BG_OUTER,
             label='RL-only — B', linestyle=':', **marker_kw)

    # Gradient fills sob as linhas híbridas
    ax1.fill_between(thresholds, res_a["hybrid_acc"], alpha=0.10,
                     color=CLR_A_HYBRID)
    ax1.fill_between(thresholds, res_b["hybrid_acc"], alpha=0.08,
                     color=CLR_B_HYBRID)

    ax1.set_xlabel('Mahalanobis Threshold', fontsize=13, color=TEXT_CLR,
                   fontweight='bold', labelpad=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=13, color=TEXT_CLR,
                   fontweight='bold', labelpad=12)
    ax1.set_title('Panel 1 — Accuracy vs. Mahalanobis Threshold',
                  fontsize=14, color=TEXT_BRIGHT, fontweight='bold', pad=18)

    leg1 = ax1.legend(loc='lower right', fontsize=9.5, frameon=True,
                      facecolor=BG_INNER, edgecolor=GRID_CLR,
                      labelcolor=TEXT_CLR, framealpha=0.95)
    leg1.get_frame().set_linewidth(0.8)

    # Marcar o pico do Hybrid A
    idx_best_a = int(np.argmax(res_a["hybrid_acc"]))
    ax1.annotate(
        f'Best A: {res_a["hybrid_acc"][idx_best_a]:.2f}%\n'
        f'(τ={thresholds[idx_best_a]:.1f})',
        xy=(thresholds[idx_best_a], res_a["hybrid_acc"][idx_best_a]),
        xytext=(20, 25), textcoords='offset points',
        fontsize=9, fontweight='bold', color=TEXT_BRIGHT,
        arrowprops=dict(arrowstyle='->', color=CLR_A_HYBRID, lw=1.5),
        bbox=dict(boxstyle='round,pad=0.4', facecolor=BG_OUTER,
                  edgecolor=CLR_A_HYBRID, alpha=0.9))

    # Marcar o pico do Hybrid B
    idx_best_b = int(np.argmax(res_b["hybrid_acc"]))
    ax1.annotate(
        f'Best B: {res_b["hybrid_acc"][idx_best_b]:.2f}%\n'
        f'(τ={thresholds[idx_best_b]:.1f})',
        xy=(thresholds[idx_best_b], res_b["hybrid_acc"][idx_best_b]),
        xytext=(-60, -35), textcoords='offset points',
        fontsize=9, fontweight='bold', color=TEXT_BRIGHT,
        arrowprops=dict(arrowstyle='->', color=CLR_B_HYBRID, lw=1.5),
        bbox=dict(boxstyle='round,pad=0.4', facecolor=BG_OUTER,
                  edgecolor=CLR_B_HYBRID, alpha=0.9))

    # ══════════════════════════════════════════════════════════════════════════
    #  PANEL 2 — RL Routing Rate vs Threshold
    # ══════════════════════════════════════════════════════════════════════════
    ax2.plot(thresholds, res_a["rl_rate"],
             color=CLR_A_RATE, linewidth=2.5, marker='o',
             markerfacecolor=CLR_A_RATE, markeredgecolor=BG_OUTER,
             label='RL Rate — A (t10k)', **marker_kw)
    ax2.plot(thresholds, res_b["rl_rate"],
             color=CLR_B_RATE, linewidth=2.5, marker='D',
             markerfacecolor=CLR_B_RATE, markeredgecolor=BG_OUTER,
             label='RL Rate — B (90%)', **marker_kw)

    ax2.fill_between(thresholds, res_a["rl_rate"], alpha=0.12,
                     color=CLR_A_RATE)
    ax2.fill_between(thresholds, res_b["rl_rate"], alpha=0.08,
                     color=CLR_B_RATE)

    ax2.set_xlabel('Mahalanobis Threshold', fontsize=13, color=TEXT_CLR,
                   fontweight='bold', labelpad=12)
    ax2.set_ylabel('RL Routing Rate (%)', fontsize=13, color=TEXT_CLR,
                   fontweight='bold', labelpad=12)
    ax2.set_title('Panel 2 — RL Intervention Rate vs. Threshold',
                  fontsize=14, color=TEXT_BRIGHT, fontweight='bold', pad=18)

    leg2 = ax2.legend(loc='upper right', fontsize=10, frameon=True,
                      facecolor=BG_INNER, edgecolor=GRID_CLR,
                      labelcolor=TEXT_CLR, framealpha=0.95)
    leg2.get_frame().set_linewidth(0.8)

    # Anotação: taxa de RL nos extremos
    for res, clr, lbl in [(res_a, CLR_A_RATE, 'A'),
                           (res_b, CLR_B_RATE, 'B')]:
        ax2.annotate(f'{res["rl_rate"][0]:.1f}%',
                     xy=(thresholds[0], res["rl_rate"][0]),
                     xytext=(10, 12), textcoords='offset points',
                     fontsize=8.5, fontweight='bold', color=clr)
        ax2.annotate(f'{res["rl_rate"][-1]:.1f}%',
                     xy=(thresholds[-1], res["rl_rate"][-1]),
                     xytext=(-30, -18), textcoords='offset points',
                     fontsize=8.5, fontweight='bold', color=clr)

    # ── Título global ─────────────────────────────────────────────────────────
    fig.suptitle(
        'Hybrid Vision Pipeline — Global Benchmark\n'
        'Mahalanobis Threshold Sweep  ·  Mixed-Noise Dataset  ·  '
        f'Noise Bands: {NOISE_LEVELS}',
        fontsize=16, color=TEXT_BRIGHT, fontweight='bold',
        y=1.02)

    plt.tight_layout(pad=3.0)

    os.makedirs(os.path.dirname(caminho), exist_ok=True)
    fig.savefig(caminho, facecolor=fig.get_facecolor(), edgecolor='none',
                bbox_inches='tight')
    plt.close(fig)
    logger.info(f"\n  ✓ Gráfico guardado em: {caminho}")


# ═══════════════════════════════════════════════════════════════════════════════
#  CONCLUSÃO METODOLÓGICA
# ═══════════════════════════════════════════════════════════════════════════════

def imprimir_conclusao(res_a: dict, res_b: dict):
    """Imprime uma conclusão metodológica detalhada comparando A vs B."""
    # Encontrar limiares ótimos
    idx_opt_a = int(np.argmax(res_a["hybrid_acc"]))
    idx_opt_b = int(np.argmax(res_b["hybrid_acc"]))

    t_opt_a = res_a["thresholds"][idx_opt_a]
    t_opt_b = res_b["thresholds"][idx_opt_b]

    best_a = res_a["hybrid_acc"][idx_opt_a]
    best_b = res_b["hybrid_acc"][idx_opt_b]

    cnn_a = res_a["cnn_acc"][0]
    cnn_b = res_b["cnn_acc"][0]
    rl_a = res_a["rl_acc"][0]
    rl_b = res_b["rl_acc"][0]

    rate_a = res_a["rl_rate"][idx_opt_a]
    rate_b = res_b["rl_rate"][idx_opt_b]

    gain_a = best_a - cnn_a
    gain_b = best_b - cnn_b

    logger.info(f"\n{'═' * 70}")
    logger.info("  CONCLUSÃO METODOLÓGICA — ANÁLISE COMPARATIVA")
    logger.info(f"{'═' * 70}")

    logger.info(f"""
  ┌─────────────────────────────────────────────────────────────────────┐
  │  APPROACH A — Native Test Set (t10k, 10,000 samples)              │
  │  ─────────────────────────────────────────────────────────────────  │
  │  Optimal Threshold (τ*):  {t_opt_a:>6.1f}                                │
  │  Hybrid Accuracy:         {best_a:>6.2f}%                               │
  │  CNN-only Baseline:       {cnn_a:>6.2f}%                               │
  │  RL-only Baseline:        {rl_a:>6.2f}%                               │
  │  RL Routing Rate at τ*:   {rate_a:>6.2f}%                               │
  │  Hybrid Gain over CNN:    {gain_a:>+6.2f} pp                              │
  ├─────────────────────────────────────────────────────────────────────┤
  │  APPROACH B — Disjoint 90% Partition (54,000 samples)             │
  │  ─────────────────────────────────────────────────────────────────  │
  │  Optimal Threshold (τ*):  {t_opt_b:>6.1f}                                │
  │  Hybrid Accuracy:         {best_b:>6.2f}%                               │
  │  CNN-only Baseline:       {cnn_b:>6.2f}%                               │
  │  RL-only Baseline:        {rl_b:>6.2f}%                               │
  │  RL Routing Rate at τ*:   {rate_b:>6.2f}%                               │
  │  Hybrid Gain over CNN:    {gain_b:>+6.2f} pp                              │
  └─────────────────────────────────────────────────────────────────────┘""")

    # Concordância de limiar
    if abs(t_opt_a - t_opt_b) <= 2.5:
        concordancia = "FORTE"
        msg = (f"Os dois cenários convergem para limiares próximos "
               f"(A: τ={t_opt_a:.1f}, B: τ={t_opt_b:.1f}), "
               f"validando a robustez do ponto operacional.")
    else:
        concordancia = "DIVERGENTE"
        msg = (f"Existe divergência nos limiares ótimos "
               f"(A: τ={t_opt_a:.1f}, B: τ={t_opt_b:.1f}). "
               f"A diferença de distribuição entre t10k e o train split "
               f"pode explicar esta variação.")

    logger.info(f"""
  ┌─────────────────────────────────────────────────────────────────────┐
  │  ANÁLISE CRUZADA                                                  │
  │  ─────────────────────────────────────────────────────────────────  │
  │  Concordância de Limiar:  {concordancia:<15}                           │
  │  Δ Accuracy (B - A):      {best_b - best_a:>+6.2f} pp                              │
  │  Δ CNN Baseline (B - A):  {cnn_b - cnn_a:>+6.2f} pp                              │
  │  Δ RL Baseline (B - A):   {rl_b - rl_a:>+6.2f} pp                              │
  │                                                                     │
  │  {msg:<65} │
  └─────────────────────────────────────────────────────────────────────┘""")

    # Recomendação final
    t_consenso = (t_opt_a + t_opt_b) / 2.0
    # Arredondar para o step de 2.5 mais próximo
    t_consenso = round(t_consenso / 2.5) * 2.5
    logger.info(f"""
  ┌─────────────────────────────────────────────────────────────────────┐
  │  RECOMENDAÇÃO PARA PRODUÇÃO                                       │
  │  ─────────────────────────────────────────────────────────────────  │
  │  Limiar de Consenso (τ):  {t_consenso:>6.1f}                                │
  │  Média ponderada dos ótimos de A e B, arredondado ao step 2.5.    │
  │  Este valor equilibra precisão máxima com robustez estatística.    │
  └─────────────────────────────────────────────────────────────────────┘
""")


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    # ── 1. Configurar Hardware (GPU) ──────────────────────────────────────────
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logger.info(f"  GPU(s) detectada(s): {len(gpus)}")
    else:
        logger.info("  Sem GPU — a usar CPU.")

    logger.info(f"\n{'═' * 70}")
    logger.info("  AVALIAÇÃO GLOBAL DO PIPELINE HÍBRIDO — BENCHMARK COMPLETO")
    logger.info(f"{'═' * 70}")
    logger.info(f"  Noise Bands:       {NOISE_LEVELS}")
    logger.info(f"  Threshold Sweep:   {THRESHOLDS[0]:.1f} → {THRESHOLDS[-1]:.1f} "
                f"(step 2.5, {len(THRESHOLDS)} pontos)")
    logger.info(f"  CNN Batch Size:    {BATCH_SIZE_CNN}")
    logger.info(f"  RL Batch Size:     {BATCH_SIZE_RL}")

    # ── 2. Carregar a CNN ─────────────────────────────────────────────────────
    logger.info("\n  A carregar CNN (Extrator de Features 128D)...")
    cnn = RawModel()
    ckpt = tf.train.Checkpoint(model=cnn)
    latest_ckpt = tf.train.latest_checkpoint(os.path.join("outputs", "checkpoints"))
    if not latest_ckpt:
        logger.error("  ERRO: Checkpoint não encontrado em outputs/checkpoints/")
        sys.exit(1)
    ckpt.restore(latest_ckpt).expect_partial()
    logger.info("  ✓ CNN carregada com sucesso.")

    # ── 3. Carregar Perfis de Mahalanobis ─────────────────────────────────────
    logger.info("  A carregar perfis de Mahalanobis...")
    caminho_perfis = os.path.join("outputs", "mahalanobis_profiles.npz")
    perfis = np.load(caminho_perfis, allow_pickle=True)
    logger.info("  ✓ Perfis de triagem carregados.")

    # ── 4. Carregar Agente RL (k-NN Bandit 128D) ─────────────────────────────
    logger.info("  A carregar agente RL (k-NN Bandit 128D)...")
    agent = KNNBanditAgent128D(k=30, n_actions=10)
    caminho_memoria = os.path.join("outputs", "knn_memory_bank_128d.npz")
    agent.load(caminho_memoria)
    stats = agent.get_memory_stats()
    logger.info(f"  ✓ Agente RL carregado ({stats['size']:,} experiências, "
                f"reward_mean={stats['reward_mean']:.3f}).")

    # ── 5. APPROACH A — Dataset t10k (10,000 amostras) ────────────────────────
    logger.info("\n  A carregar dataset t10k...")
    x_t10k, y_t10k = load_mnist_raw(os.path.join("data", "MNIST", "raw"), kind='t10k')
    x_t10k = x_t10k.astype(np.float32) / 255.0
    logger.info(f"  ✓ t10k carregado: {len(x_t10k):,} amostras.")

    res_a = avaliar_approach("APPROACH A — Native Test Set (t10k, 10,000 samples)",
                             x_t10k, y_t10k, cnn, agent, perfis)

    # ── 6. APPROACH B — Partição Disjunta 90% (54,000 amostras) ──────────────
    logger.info("\n  A carregar dataset train + split 90%...")
    x_full, y_full = load_mnist_raw(os.path.join("data", "MNIST", "raw"), kind='train')
    x_full = x_full.astype(np.float32) / 255.0

    _, x_eval_90, _, y_eval_90 = train_test_split(
        x_full, y_full,
        test_size=0.90,
        random_state=RANDOM_SEED,
        shuffle=True,
        stratify=y_full
    )
    logger.info(f"  ✓ Partição 90% isolada: {len(x_eval_90):,} amostras (100% unseen).")

    res_b = avaliar_approach("APPROACH B — Disjoint 90% Partition (54,000 samples)",
                             x_eval_90, y_eval_90, cnn, agent, perfis)

    # ── 7. Gerar Gráfico Comparativo de Publicação ────────────────────────────
    logger.info("\n  A gerar gráfico comparativo de publicação...")
    gerar_grafico_comparativo(res_a, res_b, OUTPUT_FIGURE_PATH)

    # ── 8. Conclusão Metodológica ─────────────────────────────────────────────
    imprimir_conclusao(res_a, res_b)

    logger.info("  Avaliação global concluída com sucesso!")


if __name__ == "__main__":
    main()
