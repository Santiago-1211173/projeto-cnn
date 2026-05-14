"""
Visualização Especializada do Processo de Decisão do Agente k-NN RL.

Gera dois tipos de visualização:
1. Perfil de Confiança Comparativo — Barras lado a lado mostrando CNN vs k-NN
2. Diagrama de Correção (Sankey) — Fluxo de reclassificações do k-NN sobre a CNN
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import os
import logging
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.sankey import Sankey
import matplotlib.colors as mcolors

from src.models.custom_cnn import RawModel
from src.models.knn_bandit_agent import KNNBanditAgent
from src.data.loader import load_mnist_raw

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Cores profissionais para os dígitos 0-9
DIGIT_COLORS = [
    '#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6',
    '#1ABC9C', '#E67E22', '#34495E', '#E91E63', '#00BCD4'
]


def carregar_sistema():
    """Carrega a CNN e o Agente k-NN."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    # CNN
    cnn = RawModel()
    ckpt = tf.train.Checkpoint(model=cnn)
    latest = tf.train.latest_checkpoint(os.path.join("outputs", "checkpoints"))
    ckpt.restore(latest).expect_partial()

    # k-NN Agent
    agent = KNNBanditAgent(k=15, n_actions=10)
    agent.load(os.path.join("outputs", "knn_memory_bank.npz"))

    return cnn, agent


def adicionar_ruido(imagem: np.ndarray, intensidade: float = 0.6) -> np.ndarray:
    ruido = np.random.normal(loc=0.0, scale=intensidade, size=imagem.shape)
    return np.clip(imagem + ruido, 0., 1.)


def encontrar_imagens_ambiguas(x_test, y_test, cnn, n=8):
    """
    Encontra imagens onde a CNN tem baixa confiança ou erra.
    Estas são as mais interessantes para visualizar a intervenção do k-NN.
    """
    candidatos = []

    batch_size = 500
    for i in range(0, len(x_test), batch_size):
        batch = x_test[i : i + batch_size]
        batch_t = tf.convert_to_tensor(batch, dtype=tf.float32)
        outputs = cnn(batch_t)
        probs = outputs["probabilities"].numpy()

        for j in range(len(batch)):
            idx_global = i + j
            top_prob = np.max(probs[j])
            pred = np.argmax(probs[j])
            real = y_test[idx_global]

            # Queremos imagens ambíguas: baixa confiança OU previsão errada
            if top_prob < 0.7 or pred != real:
                candidatos.append({
                    'idx': idx_global,
                    'confianca': top_prob,
                    'pred_cnn': pred,
                    'real': real,
                    'errou': pred != real
                })

    # Dar prioridade às que a CNN errou e têm baixa confiança
    candidatos.sort(key=lambda c: (not c['errou'], c['confianca']))
    return candidatos[:n]


def visualizar_perfil_confianca(x_test, y_test, cnn, agent, output_dir):
    """
    Visualização 1: Perfil de Confiança Comparativo.
    Para cada imagem ambígua, mostra lado a lado:
    - As probabilidades da CNN (10 barras)
    - As recompensas esperadas do k-NN (10 barras)
    """
    logger.info("\n--- Visualização 1: Perfil de Confiança Comparativo ---")

    # Encontrar imagens ambíguas (limpas e com ruído)
    imagens_ruido = np.array([adicionar_ruido(img) for img in x_test[:2000]])
    labels_ruido = y_test[:2000]
    candidatos = encontrar_imagens_ambiguas(imagens_ruido, labels_ruido, cnn, n=8)

    if len(candidatos) == 0:
        logger.warning("Nenhuma imagem ambígua encontrada!")
        return

    n_imgs = min(len(candidatos), 8)
    fig, axes = plt.subplots(n_imgs, 3, figsize=(18, 4 * n_imgs))
    if n_imgs == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle(
        'Perfil de Confiança: CNN vs k-NN Bandit',
        fontsize=18, fontweight='bold', y=0.98
    )

    digits = list(range(10))

    for row, cand in enumerate(candidatos[:n_imgs]):
        idx = cand['idx']
        img = imagens_ruido[idx]
        label_real = labels_ruido[idx]

        # Passar pela CNN
        tensor_img = tf.convert_to_tensor(np.expand_dims(img, 0), dtype=tf.float32)
        outputs = cnn(tensor_img)
        probs_cnn = outputs["probabilities"].numpy()[0]
        pred_cnn = int(np.argmax(probs_cnn))

        # Consultar o k-NN
        expected_rewards = agent.get_expected_rewards(probs_cnn)
        pred_knn = int(np.argmax(expected_rewards))

        # --- Coluna 1: A Imagem ---
        ax_img = axes[row, 0]
        ax_img.imshow(img[:, :, 0], cmap='gray')
        cor_titulo = 'green' if pred_knn == label_real else 'red'
        ax_img.set_title(
            f"Real: {label_real}\nCNN: {pred_cnn} | k-NN: {pred_knn}",
            fontsize=11, fontweight='bold', color=cor_titulo
        )
        ax_img.axis('off')

        # --- Coluna 2: Probabilidades da CNN ---
        ax_cnn = axes[row, 1]
        bar_colors_cnn = ['#3498DB'] * 10
        bar_colors_cnn[pred_cnn] = '#E74C3C'  # Destaque da escolha CNN
        if label_real < 10:
            bar_colors_cnn[label_real] = '#2ECC71'  # Verde para o correto

        bars_cnn = ax_cnn.bar(digits, probs_cnn, color=bar_colors_cnn,
                              edgecolor='white', linewidth=0.5, alpha=0.85)
        ax_cnn.set_title('CNN: Probabilidades (10D)', fontsize=11, fontweight='bold')
        ax_cnn.set_xlabel('Dígito')
        ax_cnn.set_ylabel('Probabilidade')
        ax_cnn.set_xticks(digits)
        ax_cnn.set_ylim(0, max(probs_cnn) * 1.25)
        ax_cnn.axhline(y=0, color='gray', linewidth=0.5)

        # Etiquetas nos topos das barras
        for bar, val in zip(bars_cnn, probs_cnn):
            if val > 0.01:
                ax_cnn.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{val:.2f}', ha='center', va='bottom', fontsize=8)

        # --- Coluna 3: Recompensas Esperadas do k-NN ---
        ax_knn = axes[row, 2]
        bar_colors_knn = ['#F39C12'] * 10
        bar_colors_knn[pred_knn] = '#E74C3C'  # Destaque da escolha k-NN
        if label_real < 10:
            bar_colors_knn[label_real] = '#2ECC71'  # Verde para o correto

        bars_knn = ax_knn.bar(digits, expected_rewards, color=bar_colors_knn,
                              edgecolor='white', linewidth=0.5, alpha=0.85)
        ax_knn.set_title('k-NN: Recompensa Esperada', fontsize=11, fontweight='bold')
        ax_knn.set_xlabel('Dígito (Ação)')
        ax_knn.set_ylabel('Recompensa Esperada')
        ax_knn.set_xticks(digits)
        y_min = min(0, min(expected_rewards) * 1.3)
        y_max = max(expected_rewards) * 1.3 if max(expected_rewards) > 0 else 0.5
        ax_knn.set_ylim(y_min, y_max)
        ax_knn.axhline(y=0, color='gray', linewidth=0.8, linestyle='--')

        for bar, val in zip(bars_knn, expected_rewards):
            if abs(val) > 0.01:
                offset = 0.02 if val >= 0 else -0.06
                ax_knn.text(bar.get_x() + bar.get_width()/2, bar.get_height() + offset,
                           f'{val:+.2f}', ha='center', va='bottom', fontsize=8)

    # Legenda global
    legend_patches = [
        mpatches.Patch(color='#2ECC71', label='Classe Correta'),
        mpatches.Patch(color='#E74C3C', label='Escolha do Modelo'),
        mpatches.Patch(color='#3498DB', label='CNN (Outras)'),
        mpatches.Patch(color='#F39C12', label='k-NN (Outras)'),
    ]
    fig.legend(handles=legend_patches, loc='lower center', ncol=4,
               fontsize=11, frameon=True, bbox_to_anchor=(0.5, 0.0))

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    path = os.path.join(output_dir, "perfil_confianca_cnn_vs_knn.png")
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    logger.info(f"[OK] Perfil de confiança guardado em: {path}")


def visualizar_sankey_correcoes(x_test, y_test, cnn, agent, output_dir):
    """
    Visualização 2: Diagrama de Fluxo de Correções.
    Mostra como as previsões Top-1 da CNN são re-roteadas pelo k-NN.
    """
    logger.info("\n--- Visualização 2: Diagrama de Correções (Sankey) ---")

    # Processar todas as imagens de teste com ruído
    imagens_ruido = np.array([adicionar_ruido(img, 0.6) for img in x_test[:5000]])
    labels = y_test[:5000]

    # Extração em batch
    batch_size = 500
    all_probs = []
    all_preds_cnn = []

    for i in range(0, len(imagens_ruido), batch_size):
        batch = imagens_ruido[i : i + batch_size]
        batch_t = tf.convert_to_tensor(batch, dtype=tf.float32)
        outputs = cnn(batch_t)
        probs = outputs["probabilities"].numpy()
        all_probs.append(probs)
        all_preds_cnn.append(np.argmax(probs, axis=1))

    all_probs = np.vstack(all_probs)
    preds_cnn = np.concatenate(all_preds_cnn)
    preds_knn = agent.get_action_batch(all_probs, epsilon=0.0)

    # Contar as correções (onde o k-NN mudou a decisão da CNN)
    correcoes = preds_cnn != preds_knn
    n_corrigidos = np.sum(correcoes)
    logger.info(f"Total de correções pelo k-NN: {n_corrigidos} / {len(labels)}")

    # Construir a matriz de fluxo CNN -> k-NN (apenas onde houve correção)
    fluxo = np.zeros((10, 10), dtype=int)
    acertos_cnn_para_erro = 0
    erros_cnn_para_acerto = 0

    for i in range(len(labels)):
        if correcoes[i]:
            fluxo[preds_cnn[i], preds_knn[i]] += 1
            if preds_cnn[i] == labels[i] and preds_knn[i] != labels[i]:
                acertos_cnn_para_erro += 1
            elif preds_cnn[i] != labels[i] and preds_knn[i] == labels[i]:
                erros_cnn_para_acerto += 1

    # --- Visualização: Heatmap de Fluxo de Correções ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8),
                                    gridspec_kw={'width_ratios': [1.3, 1]})

    fig.suptitle(
        'Fluxo de Correções: CNN → k-NN Bandit',
        fontsize=16, fontweight='bold', y=1.02
    )

    # --- Painel 1: Heatmap de Reclassificações ---
    # Mascarar a diagonal (não-correções)
    fluxo_display = fluxo.copy().astype(float)
    np.fill_diagonal(fluxo_display, np.nan)

    im = ax1.imshow(fluxo_display, cmap='YlOrRd', interpolation='nearest',
                     aspect='equal')

    ax1.set_xlabel('Decisão Final (k-NN)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Previsão Original (CNN)', fontsize=12, fontweight='bold')
    ax1.set_title('Mapa de Reclassificações', fontsize=13, fontweight='bold')
    ax1.set_xticks(range(10))
    ax1.set_yticks(range(10))
    ax1.set_xticklabels(range(10), fontsize=11)
    ax1.set_yticklabels(range(10), fontsize=11)

    # Anotações nos quadrados
    for i in range(10):
        for j in range(10):
            if i != j and fluxo[i, j] > 0:
                ax1.text(j, i, str(fluxo[i, j]), ha='center', va='center',
                        fontsize=9, fontweight='bold',
                        color='white' if fluxo[i, j] > fluxo.max() * 0.5 else 'black')

    plt.colorbar(im, ax=ax1, label='Nº de Imagens Reclassificadas', shrink=0.8)

    # --- Painel 2: Resumo Estatístico das Correções ---
    acc_cnn = np.mean(preds_cnn == labels) * 100
    acc_knn = np.mean(preds_knn == labels) * 100

    # Decisão híbrida: para imagens onde CNN tem alta confiança, manter CNN
    confianca_cnn = np.max(all_probs, axis=1)
    limiar_confianca = 0.8
    preds_hibrido = np.where(confianca_cnn > limiar_confianca, preds_cnn, preds_knn)
    acc_hibrido = np.mean(preds_hibrido == labels) * 100

    categorias = ['CNN\nSozinha', 'k-NN\nSozinho', 'Sistema\nHíbrido']
    valores = [acc_cnn, acc_knn, acc_hibrido]
    cores = ['#3498DB', '#F39C12', '#2ECC71']

    bars = ax2.bar(categorias, valores, color=cores, edgecolor='white',
                   linewidth=2, width=0.6, alpha=0.9)

    for bar, val in zip(bars, valores):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=14, fontweight='bold')

    ax2.set_ylabel('Precisão (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Comparação de Precisão (Com Ruído)', fontsize=13, fontweight='bold')
    ax2.set_ylim(0, 100)
    ax2.axhline(y=acc_cnn, color='#3498DB', linestyle='--', alpha=0.3)

    # Caixa de estatísticas
    stats_text = (
        f"Correções Totais: {n_corrigidos}\n"
        f"CNN errou → k-NN acertou: {erros_cnn_para_acerto}\n"
        f"CNN acertou → k-NN errou: {acertos_cnn_para_erro}\n"
        f"Ganho líquido: {erros_cnn_para_acerto - acertos_cnn_para_erro}"
    )
    ax2.text(0.5, 0.35, stats_text, transform=ax2.transAxes,
             fontsize=10, verticalalignment='top', horizontalalignment='center',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow',
                      edgecolor='gray', alpha=0.8))

    plt.tight_layout()
    path = os.path.join(output_dir, "fluxo_correcoes_cnn_knn.png")
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    logger.info(f"[OK] Diagrama de correções guardado em: {path}")

    # Estatísticas no terminal
    logger.info(f"\nResumo:")
    logger.info(f"  CNN sozinha (com ruído): {acc_cnn:.1f}%")
    logger.info(f"  k-NN sozinho (com ruído): {acc_knn:.1f}%")
    logger.info(f"  Sistema Híbrido: {acc_hibrido:.1f}%")
    logger.info(f"  CNN→Acerto→k-NN→Erro: {acertos_cnn_para_erro}")
    logger.info(f"  CNN→Erro→k-NN→Acerto: {erros_cnn_para_acerto}")


def main():
    logger.info("==================================================")
    logger.info("VISUALIZAÇÃO DO PROCESSO DE DECISÃO k-NN RL")
    logger.info("==================================================")

    cnn, agent = carregar_sistema()

    # Carregar dados de teste
    x_test, y_test = load_mnist_raw(os.path.join("data", "MNIST", "raw"), kind='t10k')
    x_test = x_test.astype(np.float32) / 255.0

    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    # Gerar as duas visualizações
    visualizar_perfil_confianca(x_test, y_test, cnn, agent, output_dir)
    visualizar_sankey_correcoes(x_test, y_test, cnn, agent, output_dir)

    logger.info("\n==================================================")
    logger.info("TODAS AS VISUALIZAÇÕES GERADAS COM SUCESSO!")
    logger.info("==================================================")


if __name__ == "__main__":
    main()
