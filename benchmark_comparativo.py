"""
Benchmark Comparativo: MLP (Q-Network) vs k-NN Bandit.
Corre AMBOS os agentes RL lado a lado nas mesmas imagens de teste
e produz um relatório comparativo detalhado com métricas hard.

Saída: outputs/comparacao_mlp_vs_knn.png (gráfico) + relatório no terminal.
"""

import os
import time
import logging
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from src.models.custom_cnn import RawModel
from src.models.rl_agent import QNetworkAgent
from src.models.knn_bandit_agent import KNNBanditAgent
from src.models.knn_bandit_agent_128d import KNNBanditAgent128D
from src.data.loader import load_mnist_raw

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

LIMIAR_MAHALANOBIS = 5.0


def calcular_mahalanobis_batch(vetores, mu, inv_sigma):
    """Fórmula vetorizada de Mahalanobis."""
    diff = vetores - mu
    left = np.dot(diff, inv_sigma)
    dist_quadrada = np.sum(left * diff, axis=1)
    return np.sqrt(dist_quadrada)


def extrair_features_cnn(imagens, cnn, batch_size=500):
    """Extrai vetores 128D e probabilidades 10D da CNN em batches."""
    todos_128d, todos_10d, todas_preds = [], [], []
    num_batches = len(imagens) // batch_size
    resto = len(imagens) % batch_size

    for i in range(num_batches + (1 if resto > 0 else 0)):
        start = i * batch_size
        end = min(start + batch_size, len(imagens))
        batch = tf.convert_to_tensor(imagens[start:end], dtype=tf.float32)

        outputs = cnn(batch)
        todos_128d.append(outputs["latent_features"].numpy())
        todos_10d.append(outputs["probabilities"].numpy())
        todas_preds.append(tf.argmax(outputs["probabilities"], axis=1).numpy())

    return np.vstack(todos_128d), np.vstack(todos_10d), np.concatenate(todas_preds)


def avaliar_agente_mlp(agent_mlp, vetores_128d):
    """Gera previsões do agente MLP (Q-Network) usando espaço latente 128D."""
    q_values = agent_mlp(tf.convert_to_tensor(vetores_128d, dtype=tf.float32))
    return tf.argmax(q_values, axis=1).numpy()


def avaliar_agente_knn(agent_knn, probs_10d):
    """Gera previsões do agente k-NN usando probabilidades 10D."""
    return agent_knn.get_action_batch(probs_10d, epsilon=0.0)


def calcular_roteamento(vetores_128d, mapa_ilhas, limiar):
    """Calcula a máscara de roteamento Mahalanobis."""
    todas_distancias = np.zeros((len(vetores_128d), 10))
    for digito in range(10):
        mu = mapa_ilhas[str(digito)].item()["mu"]
        inv_sigma = mapa_ilhas[str(digito)].item()["inv_sigma"]
        todas_distancias[:, digito] = calcular_mahalanobis_batch(vetores_128d, mu, inv_sigma)
    distancias_minimas = np.min(todas_distancias, axis=1)
    mascara_cnn = distancias_minimas < limiar
    return mascara_cnn


def avaliar_cenario(nome, imagens, labels, cnn, agent_mlp, agent_knn, agent_knn_128d, mapa_ilhas):
    """Avalia um cenário completo para ambos os agentes."""

    # 1. Extração de features (partilhada por ambos)
    t0 = time.time()
    vetores_128d, probs_10d, preds_cnn = extrair_features_cnn(imagens, cnn)
    tempo_cnn = time.time() - t0

    # 2. Roteamento Mahalanobis (partilhado)
    mascara_cnn = calcular_roteamento(vetores_128d, mapa_ilhas, LIMIAR_MAHALANOBIS)
    mascara_rl = ~mascara_cnn

    # 3. Previsões dos agentes
    t_mlp = time.time()
    preds_mlp = avaliar_agente_mlp(agent_mlp, vetores_128d)
    tempo_mlp = time.time() - t_mlp

    t_knn = time.time()
    preds_knn = avaliar_agente_knn(agent_knn, probs_10d)
    tempo_knn = time.time() - t_knn

    t_knn_128d = time.time()
    preds_knn_128d = agent_knn_128d.get_action_batch(vetores_128d, epsilon=0.0)
    tempo_knn_128d = time.time() - t_knn_128d

    # 4. Métricas CNN Isolada
    acc_cnn = np.mean(preds_cnn == labels) * 100

    # 5. Métricas por Agente (Isolado — sem roteamento)
    acc_mlp_isolada = np.mean(preds_mlp == labels) * 100
    acc_knn_isolada = np.mean(preds_knn == labels) * 100
    acc_knn_128d_isolada = np.mean(preds_knn_128d == labels) * 100

    # 6. Métricas por Agente (Só no subconjunto roteado pelo Árbitro)
    total_rl = np.sum(mascara_rl)
    total_cnn_route = np.sum(mascara_cnn)

    acertos_cnn_route = np.sum((preds_cnn == labels) & mascara_cnn)

    if total_rl > 0:
        acertos_mlp_rl = np.sum((preds_mlp == labels) & mascara_rl)
        acertos_knn_rl = np.sum((preds_knn == labels) & mascara_rl)
        acertos_knn_128d_rl = np.sum((preds_knn_128d == labels) & mascara_rl)
        acc_mlp_rl = (acertos_mlp_rl / total_rl) * 100
        acc_knn_rl = (acertos_knn_rl / total_rl) * 100
        acc_knn_128d_rl = (acertos_knn_128d_rl / total_rl) * 100
    else:
        acertos_mlp_rl, acertos_knn_rl, acertos_knn_128d_rl = 0, 0, 0
        acc_mlp_rl, acc_knn_rl, acc_knn_128d_rl = 0.0, 0.0, 0.0

    # 7. Accuracy Híbrida Global
    acc_hibrida_mlp = ((acertos_cnn_route + acertos_mlp_rl) / len(labels)) * 100
    acc_hibrida_knn = ((acertos_cnn_route + acertos_knn_rl) / len(labels)) * 100
    acc_hibrida_knn_128d = ((acertos_cnn_route + acertos_knn_128d_rl) / len(labels)) * 100

    # 8. Métricas por classe (para os casos roteados ao RL)
    acc_por_classe_mlp = np.zeros(10)
    acc_por_classe_knn = np.zeros(10)
    acc_por_classe_knn_128d = np.zeros(10)
    count_por_classe = np.zeros(10)
    for d in range(10):
        mask_d = (labels == d) & mascara_rl
        n_d = np.sum(mask_d)
        count_por_classe[d] = n_d
        if n_d > 0:
            acc_por_classe_mlp[d] = np.mean(preds_mlp[mask_d] == labels[mask_d]) * 100
            acc_por_classe_knn[d] = np.mean(preds_knn[mask_d] == labels[mask_d]) * 100
            acc_por_classe_knn_128d[d] = np.mean(preds_knn_128d[mask_d] == labels[mask_d]) * 100

    return {
        "nome": nome,
        "n_total": len(labels),
        "n_cnn": int(total_cnn_route),
        "n_rl": int(total_rl),
        "acc_cnn": acc_cnn,
        "acc_mlp_isolada": acc_mlp_isolada,
        "acc_knn_isolada": acc_knn_isolada,
        "acc_knn_128d_isolada": acc_knn_128d_isolada,
        "acc_mlp_rl": acc_mlp_rl,
        "acc_knn_rl": acc_knn_rl,
        "acc_knn_128d_rl": acc_knn_128d_rl,
        "acc_hibrida_mlp": acc_hibrida_mlp,
        "acc_hibrida_knn": acc_hibrida_knn,
        "acc_hibrida_knn_128d": acc_hibrida_knn_128d,
        "tempo_mlp": tempo_mlp,
        "tempo_knn": tempo_knn,
        "tempo_knn_128d": tempo_knn_128d,
        "tempo_cnn": tempo_cnn,
        "acc_por_classe_mlp": acc_por_classe_mlp,
        "acc_por_classe_knn": acc_por_classe_knn,
        "acc_por_classe_knn_128d": acc_por_classe_knn_128d,
        "count_por_classe": count_por_classe,
    }


def imprimir_relatorio(r):
    """Imprime o relatório formatado para um cenário."""
    logger.info("=" * 85)
    logger.info(f"  {r['nome'].upper()}")
    logger.info("=" * 85)
    logger.info(f"  Imagens: {r['n_total']} | Roteadas CNN: {r['n_cnn']} | Roteadas RL: {r['n_rl']}")
    logger.info("-" * 85)
    logger.info(f"  {'Métrica':<35} {'MLP (Q-Net)':>15} {'k-NN 10D':>15} {'k-NN Improved':>15}")
    logger.info("-" * 85)
    logger.info(f"  {'CNN Sozinha (sem árbitro):':<35} {r['acc_cnn']:>14.1f}% {r['acc_cnn']:>14.1f}% {r['acc_cnn']:>14.1f}%")
    logger.info(f"  {'Agente Isolado:':<35} {r['acc_mlp_isolada']:>14.1f}% {r['acc_knn_isolada']:>14.1f}% {r['acc_knn_128d_isolada']:>14.1f}%")
    logger.info(f"  {'Agente no Subconjunto RL:':<35} {r['acc_mlp_rl']:>14.1f}% {r['acc_knn_rl']:>14.1f}% {r['acc_knn_128d_rl']:>14.1f}%")
    logger.info(f"  {'SISTEMA HÍBRIDO GLOBAL:':<35} {r['acc_hibrida_mlp']:>14.1f}% {r['acc_hibrida_knn']:>14.1f}% {r['acc_hibrida_knn_128d']:>14.1f}%")
    logger.info("-" * 85)
    logger.info(f"  {'Tempo de Inferência:':<35} {r['tempo_mlp']*1000:>12.1f} ms {r['tempo_knn']*1000:>12.1f} ms {r['tempo_knn_128d']*1000:>12.1f} ms")
    logger.info("=" * 85)

    if r['n_rl'] > 0:
        logger.info(f"\n  Acc por Classe (Subconjunto RL, {r['n_rl']} imgs):")
        logger.info(f"  {'Dígito':<8} {'N':>6} {'MLP':>10} {'k-NN 10D':>10} {'k-NN 128D':>10} {'Δ (128D - MLP)':>16}")
        for d in range(10):
            n = int(r['count_por_classe'][d])
            a_mlp = r['acc_por_classe_mlp'][d]
            a_knn = r['acc_por_classe_knn'][d]
            a_knn_128d = r['acc_por_classe_knn_128d'][d]
            delta = a_knn_128d - a_mlp
            sinal = "+" if delta >= 0 else ""
            logger.info(f"  {d:<8} {n:>6} {a_mlp:>9.1f}% {a_knn:>9.1f}% {a_knn_128d:>9.1f}% {sinal}{delta:>15.1f}%")
    logger.info("")


def gerar_grafico(resultados, caminho):
    """Gera gráfico comparativo lado a lado."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Benchmark Comparativo: MLP (Q-Network) vs k-NN Bandit",
                 fontsize=16, fontweight='bold', y=0.98)

    cores_mlp = '#e74c3c'
    cores_knn = '#2ecc71'
    cores_cnn = '#3498db'

    for idx, r in enumerate(resultados):
        ax = axes[idx]

        labels_bar = ['CNN\nSozinha', 'Agente\nIsolado', 'Agente no\nSubconjunto RL', 'Sistema\nHíbrido']
        vals_mlp = [r['acc_cnn'], r['acc_mlp_isolada'], r['acc_mlp_rl'], r['acc_hibrida_mlp']]
        vals_knn = [r['acc_cnn'], r['acc_knn_isolada'], r['acc_knn_rl'], r['acc_hibrida_knn']]
        vals_knn_128d = [r['acc_cnn'], r['acc_knn_128d_isolada'], r['acc_knn_128d_rl'], r['acc_hibrida_knn_128d']]

        x = np.arange(len(labels_bar))
        w = 0.25

        bars_mlp = ax.bar(x - w, vals_mlp, w, label='MLP (Q-Network)', color=cores_mlp, alpha=0.85, edgecolor='white')
        bars_knn = ax.bar(x, vals_knn, w, label='k-NN 10D', color=cores_knn, alpha=0.85, edgecolor='white')
        bars_knn_128d = ax.bar(x + w, vals_knn_128d, w, label='k-NN 128D', color='#9b59b6', alpha=0.85, edgecolor='white')

        # CNN baseline é igual para ambos — pintar de azul
        bars_mlp[0].set_color(cores_cnn)
        bars_knn[0].set_color(cores_cnn)
        bars_knn_128d[0].set_color(cores_cnn)

        # Valores sobre as barras
        for bar in list(bars_mlp) + list(bars_knn) + list(bars_knn_128d):
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width()/2., h + 0.5,
                        f'{h:.1f}%', ha='center', va='bottom', fontsize=7, fontweight='bold')

        ax.set_title(r['nome'], fontsize=13, fontweight='bold', pad=12)
        ax.set_ylabel('Precisão (%)')
        ax.set_xticks(x)
        ax.set_xticklabels(labels_bar, fontsize=9)
        ax.set_ylim(0, 115)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(axis='y', alpha=0.3)

        # Destaque na diferença do sistema híbrido
        delta = r['acc_hibrida_knn_128d'] - r['acc_hibrida_mlp']
        sinal = "+" if delta >= 0 else ""
        cor_delta = '#9b59b6' if delta >= 0 else cores_mlp
        ax.annotate(f'Δ (128D vs MLP) = {sinal}{delta:.1f}%',
                    xy=(3, max(r['acc_hibrida_mlp'], r['acc_hibrida_knn'], r['acc_hibrida_knn_128d']) + 4),
                    fontsize=10, fontweight='bold', color=cor_delta, ha='center')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(caminho, dpi=200, bbox_inches='tight')
    plt.close()
    logger.info(f"[OK] Gráfico guardado em: {caminho}")


def main():
    # 1. Configurar Hardware
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    logger.info("=" * 72)
    logger.info("  BENCHMARK COMPARATIVO: MLP (Q-Network) vs k-NN Bandit")
    logger.info("=" * 72)

    # 2. Carregar CNN
    logger.info("\n[1/4] A carregar a CNN...")
    cnn = RawModel()
    ckpt_cnn = tf.train.Checkpoint(model=cnn)
    ckpt_cnn.restore(tf.train.latest_checkpoint(os.path.join("outputs", "checkpoints"))).expect_partial()

    # 3. Carregar Mahalanobis Profiles
    logger.info("[2/4] A carregar o Árbitro (Mahalanobis)...")
    mapa_ilhas = np.load(os.path.join("outputs", "mahalanobis_profiles.npz"), allow_pickle=True)

    # 4. Carregar Agente MLP (Antigo)
    logger.info("[3/4] A carregar o Agente MLP (Q-Network)...")
    agent_mlp = QNetworkAgent()
    ckpt_agent = tf.train.Checkpoint(model=agent_mlp)
    ckpt_agent.restore(os.path.join("outputs", "rl_agent_weights-1")).expect_partial()

    # 5. Carregar Agente k-NN (Novo)
    logger.info("[4/5] A carregar o Agente k-NN 10D...")
    agent_knn = KNNBanditAgent(k=30, n_actions=10)
    agent_knn.load(os.path.join("outputs", "knn_memory_bank.npz"))

    logger.info("[5/5] A carregar o Agente k-NN 128D...")
    agent_knn_128d = KNNBanditAgent128D(k=30, n_actions=10)
    agent_knn_128d.load(os.path.join("outputs", "knn_memory_bank_128d.npz"))

    # 6. Carregar Dados de Teste
    x_test, y_test = load_mnist_raw(os.path.join("data", "MNIST", "raw"), kind='t10k')
    x_test_limpo = x_test.astype(np.float32) / 255.0

    # Criar cópia corrompida (MESMA seed para fairness)
    np.random.seed(42)
    ruido = np.random.normal(loc=0.0, scale=0.6, size=x_test_limpo.shape)
    x_test_ruido = np.clip(x_test_limpo + ruido, 0., 1.)

    # 7. Avaliar Ambos os Cenários
    logger.info("\n" + "=" * 72)
    r_limpo = avaliar_cenario(
        "Cenário A: Imagens Limpas",
        x_test_limpo, y_test, cnn, agent_mlp, agent_knn, agent_knn_128d, mapa_ilhas
    )
    imprimir_relatorio(r_limpo)

    r_ruido = avaliar_cenario(
        "Cenário B: Imagens com Ruído (σ=0.6)",
        x_test_ruido, y_test, cnn, agent_mlp, agent_knn, agent_knn_128d, mapa_ilhas
    )
    imprimir_relatorio(r_ruido)

    # 8. Gráfico Comparativo
    gerar_grafico([r_limpo, r_ruido], os.path.join("outputs", "comparacao_mlp_vs_knn.png"))

    # 9. Resumo Final
    logger.info("=" * 85)
    logger.info("  RESUMO FINAL")
    logger.info("=" * 85)
    logger.info(f"  {'Cenário':<35} {'Híbrido+MLP':>14} {'Híbrido+k-NN':>14} {'Híbrido+128D':>14} {'Δ (128D-MLP)':>14}")
    logger.info("-" * 85)
    for r in [r_limpo, r_ruido]:
        d = r['acc_hibrida_knn_128d'] - r['acc_hibrida_mlp']
        s = "+" if d >= 0 else ""
        logger.info(f"  {r['nome']:<35} {r['acc_hibrida_mlp']:>13.1f}% {r['acc_hibrida_knn']:>13.1f}% {r['acc_hibrida_knn_128d']:>13.1f}% {s}{d:>11.1f}%")
    logger.info("=" * 85)

    # 10. Tabela de métricas em formato para copiar para o README
    logger.info("\n--- TABELA PARA O README (Markdown) ---")
    logger.info("| Métrica | MLP (Q-Network) | k-NN 10D | k-NN Improved (128D) | Gap (128D vs MLP) |")
    logger.info("|---|---|---|---|---|")
    logger.info(f"| Híbrido (Limpas) | {r_limpo['acc_hibrida_mlp']:.1f}% | {r_limpo['acc_hibrida_knn']:.1f}% | {r_limpo['acc_hibrida_knn_128d']:.1f}% | {r_limpo['acc_hibrida_knn_128d'] - r_limpo['acc_hibrida_mlp']:+.1f}% |")
    logger.info(f"| Híbrido (Ruído σ=0.6) | {r_ruido['acc_hibrida_mlp']:.1f}% | {r_ruido['acc_hibrida_knn']:.1f}% | {r_ruido['acc_hibrida_knn_128d']:.1f}% | {r_ruido['acc_hibrida_knn_128d'] - r_ruido['acc_hibrida_mlp']:+.1f}% |")
    logger.info(f"| Agente Isolado (Limpas) | {r_limpo['acc_mlp_isolada']:.1f}% | {r_limpo['acc_knn_isolada']:.1f}% | {r_limpo['acc_knn_128d_isolada']:.1f}% | {r_limpo['acc_knn_128d_isolada'] - r_limpo['acc_mlp_isolada']:+.1f}% |")
    logger.info(f"| Agente Isolado (Ruído) | {r_ruido['acc_mlp_isolada']:.1f}% | {r_ruido['acc_knn_isolada']:.1f}% | {r_ruido['acc_knn_128d_isolada']:.1f}% | {r_ruido['acc_knn_128d_isolada'] - r_ruido['acc_mlp_isolada']:+.1f}% |")
    logger.info(f"| Inferência (ms) | {r_ruido['tempo_mlp']*1000:.1f} | {r_ruido['tempo_knn']*1000:.1f} | {r_ruido['tempo_knn_128d']*1000:.1f} | - |")
    logger.info(f"| Treino | 10 épocas + backprop | 0 (memória) | 0 (memória) | - |")
    logger.info(f"| Parâmetros | ~8,714 (128→64→10) | 0 (non-parametric) | 0 (non-parametric) | - |")


if __name__ == "__main__":
    main()
