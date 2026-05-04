import re

with open("benchmark_comparativo.py", "r", encoding="utf-8") as f:
    content = f.read()

# Add import
content = content.replace(
    "from src.models.knn_bandit_agent import KNNBanditAgent",
    "from src.models.knn_bandit_agent import KNNBanditAgent\nfrom src.models.knn_bandit_agent_128d import KNNBanditAgent128D"
)

# Update avaliar_cenario signature
content = content.replace(
    "def avaliar_cenario(nome, imagens, labels, cnn, agent_mlp, agent_knn, mapa_ilhas):",
    "def avaliar_cenario(nome, imagens, labels, cnn, agent_mlp, agent_knn, agent_knn_128d, mapa_ilhas):"
)

# Update previsões
content = content.replace(
    "    tempo_knn = time.time() - t_knn",
    "    tempo_knn = time.time() - t_knn\n\n    t_knn_128d = time.time()\n    preds_knn_128d = agent_knn_128d.get_action_batch(vetores_128d, epsilon=0.0)\n    tempo_knn_128d = time.time() - t_knn_128d"
)

# Update metrics
content = content.replace(
    "    acc_knn_isolada = np.mean(preds_knn == labels) * 100",
    "    acc_knn_isolada = np.mean(preds_knn == labels) * 100\n    acc_knn_128d_isolada = np.mean(preds_knn_128d == labels) * 100"
)

content = content.replace(
    "        acertos_knn_rl = np.sum((preds_knn == labels) & mascara_rl)",
    "        acertos_knn_rl = np.sum((preds_knn == labels) & mascara_rl)\n        acertos_knn_128d_rl = np.sum((preds_knn_128d == labels) & mascara_rl)"
)

content = content.replace(
    "        acc_knn_rl = (acertos_knn_rl / total_rl) * 100",
    "        acc_knn_rl = (acertos_knn_rl / total_rl) * 100\n        acc_knn_128d_rl = (acertos_knn_128d_rl / total_rl) * 100"
)

content = content.replace(
    "        acertos_mlp_rl, acertos_knn_rl = 0, 0",
    "        acertos_mlp_rl, acertos_knn_rl, acertos_knn_128d_rl = 0, 0, 0"
)

content = content.replace(
    "        acc_mlp_rl, acc_knn_rl = 0.0, 0.0",
    "        acc_mlp_rl, acc_knn_rl, acc_knn_128d_rl = 0.0, 0.0, 0.0"
)

content = content.replace(
    "    acc_hibrida_knn = ((acertos_cnn_route + acertos_knn_rl) / len(labels)) * 100",
    "    acc_hibrida_knn = ((acertos_cnn_route + acertos_knn_rl) / len(labels)) * 100\n    acc_hibrida_knn_128d = ((acertos_cnn_route + acertos_knn_128d_rl) / len(labels)) * 100"
)

content = content.replace(
    "    acc_por_classe_knn = np.zeros(10)",
    "    acc_por_classe_knn = np.zeros(10)\n    acc_por_classe_knn_128d = np.zeros(10)"
)

content = content.replace(
    "            acc_por_classe_knn[d] = np.mean(preds_knn[mask_d] == labels[mask_d]) * 100",
    "            acc_por_classe_knn[d] = np.mean(preds_knn[mask_d] == labels[mask_d]) * 100\n            acc_por_classe_knn_128d[d] = np.mean(preds_knn_128d[mask_d] == labels[mask_d]) * 100"
)

content = re.sub(
    r"        \"acc_knn_isolada\": acc_knn_isolada,",
    "        \"acc_knn_isolada\": acc_knn_isolada,\n        \"acc_knn_128d_isolada\": acc_knn_128d_isolada,",
    content
)
content = re.sub(
    r"        \"acc_knn_rl\": acc_knn_rl,",
    "        \"acc_knn_rl\": acc_knn_rl,\n        \"acc_knn_128d_rl\": acc_knn_128d_rl,",
    content
)
content = re.sub(
    r"        \"acc_hibrida_knn\": acc_hibrida_knn,",
    "        \"acc_hibrida_knn\": acc_hibrida_knn,\n        \"acc_hibrida_knn_128d\": acc_hibrida_knn_128d,",
    content
)
content = re.sub(
    r"        \"tempo_knn\": tempo_knn,",
    "        \"tempo_knn\": tempo_knn,\n        \"tempo_knn_128d\": tempo_knn_128d,",
    content
)
content = re.sub(
    r"        \"acc_por_classe_knn\": acc_por_classe_knn,",
    "        \"acc_por_classe_knn\": acc_por_classe_knn,\n        \"acc_por_classe_knn_128d\": acc_por_classe_knn_128d,",
    content
)

# Update imprimir_relatorio
old_imprimir = """def imprimir_relatorio(r):
    \"\"\"Imprime o relatório formatado para um cenário.\"\"\"
    logger.info("=" * 72)
    logger.info(f"  {r['nome'].upper()}")
    logger.info("=" * 72)
    logger.info(f"  Imagens: {r['n_total']} | Roteadas CNN: {r['n_cnn']} | Roteadas RL: {r['n_rl']}")
    logger.info("-" * 72)
    logger.info(f"  {'Métrica':<40} {'MLP (Q-Net)':>12} {'k-NN Bandit':>12}")
    logger.info("-" * 72)
    logger.info(f"  {'CNN Sozinha (sem árbitro):':<40} {r['acc_cnn']:>11.1f}% {r['acc_cnn']:>11.1f}%")
    logger.info(f"  {'Agente Isolado (todas as imagens):':<40} {r['acc_mlp_isolada']:>11.1f}% {r['acc_knn_isolada']:>11.1f}%")
    logger.info(f"  {'Agente no Subconjunto RL (roteado):':<40} {r['acc_mlp_rl']:>11.1f}% {r['acc_knn_rl']:>11.1f}%")
    logger.info(f"  {'SISTEMA HÍBRIDO GLOBAL:':<40} {r['acc_hibrida_mlp']:>11.1f}% {r['acc_hibrida_knn']:>11.1f}%")
    logger.info("-" * 72)
    logger.info(f"  {'Tempo de Inferência do Agente:':<40} {r['tempo_mlp']*1000:>9.1f} ms {r['tempo_knn']*1000:>9.1f} ms")
    logger.info("=" * 72)

    # Per-class breakdown (se houver imagens roteadas)
    if r['n_rl'] > 0:
        logger.info(f"\\n  Acc por Classe (Subconjunto RL, {r['n_rl']} imgs):")
        logger.info(f"  {'Dígito':<10} {'N':>6} {'MLP':>10} {'k-NN':>10} {'Δ (k-NN−MLP)':>14}")
        for d in range(10):
            n = int(r['count_por_classe'][d])
            a_mlp = r['acc_por_classe_mlp'][d]
            a_knn = r['acc_por_classe_knn'][d]
            delta = a_knn - a_mlp
            sinal = "+" if delta >= 0 else ""
            logger.info(f"  {d:<10} {n:>6} {a_mlp:>9.1f}% {a_knn:>9.1f}% {sinal}{delta:>12.1f}%")
    logger.info(\"\")"""

new_imprimir = """def imprimir_relatorio(r):
    \"\"\"Imprime o relatório formatado para um cenário.\"\"\"
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
        logger.info(f"\\n  Acc por Classe (Subconjunto RL, {r['n_rl']} imgs):")
        logger.info(f"  {'Dígito':<8} {'N':>6} {'MLP':>10} {'k-NN 10D':>10} {'k-NN 128D':>10} {'Δ (128D - MLP)':>16}")
        for d in range(10):
            n = int(r['count_por_classe'][d])
            a_mlp = r['acc_por_classe_mlp'][d]
            a_knn = r['acc_por_classe_knn'][d]
            a_knn_128d = r['acc_por_classe_knn_128d'][d]
            delta = a_knn_128d - a_mlp
            sinal = "+" if delta >= 0 else ""
            logger.info(f"  {d:<8} {n:>6} {a_mlp:>9.1f}% {a_knn:>9.1f}% {a_knn_128d:>9.1f}% {sinal}{delta:>15.1f}%")
    logger.info(\"\")"""
content = content.replace(old_imprimir, new_imprimir)

# Update gerar_grafico
old_grafico = """        labels_bar = ['CNN\\nSozinha', 'Agente\\nIsolado', 'Agente no\\nSubconjunto RL', 'Sistema\\nHíbrido']
        vals_mlp = [r['acc_cnn'], r['acc_mlp_isolada'], r['acc_mlp_rl'], r['acc_hibrida_mlp']]
        vals_knn = [r['acc_cnn'], r['acc_knn_isolada'], r['acc_knn_rl'], r['acc_hibrida_knn']]

        x = np.arange(len(labels_bar))
        w = 0.32

        bars_mlp = ax.bar(x - w/2, vals_mlp, w, label='MLP (Q-Network)', color=cores_mlp, alpha=0.85, edgecolor='white')
        bars_knn = ax.bar(x + w/2, vals_knn, w, label='k-NN Bandit', color=cores_knn, alpha=0.85, edgecolor='white')

        # CNN baseline é igual para ambos — pintar de azul
        bars_mlp[0].set_color(cores_cnn)
        bars_knn[0].set_color(cores_cnn)

        # Valores sobre as barras
        for bar in list(bars_mlp) + list(bars_knn):
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width()/2., h + 0.5,
                        f'{h:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')

        ax.set_title(r['nome'], fontsize=13, fontweight='bold', pad=12)
        ax.set_ylabel('Precisão (%)')
        ax.set_xticks(x)
        ax.set_xticklabels(labels_bar, fontsize=9)
        ax.set_ylim(0, 105)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(axis='y', alpha=0.3)

        # Destaque na diferença do sistema híbrido
        delta = r['acc_hibrida_knn'] - r['acc_hibrida_mlp']
        sinal = "+" if delta >= 0 else ""
        cor_delta = cores_knn if delta >= 0 else cores_mlp
        ax.annotate(f'Δ = {sinal}{delta:.1f}%',
                    xy=(3, max(r['acc_hibrida_mlp'], r['acc_hibrida_knn']) + 3),
                    fontsize=11, fontweight='bold', color=cor_delta, ha='center')"""

new_grafico = """        labels_bar = ['CNN\\nSozinha', 'Agente\\nIsolado', 'Agente no\\nSubconjunto RL', 'Sistema\\nHíbrido']
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
                    fontsize=10, fontweight='bold', color=cor_delta, ha='center')"""
content = content.replace(old_grafico, new_grafico)

content = content.replace(
    "    logger.info(\"[4/4] A carregar o Agente k-NN Bandit...\")",
    "    logger.info(\"[4/5] A carregar o Agente k-NN 10D...\")"
)
content = content.replace(
    "    # 6. Carregar Dados de Teste",
    "    logger.info(\"[5/5] A carregar o Agente k-NN 128D...\")\n    agent_knn_128d = KNNBanditAgent128D(k=30, n_actions=10)\n    agent_knn_128d.load(os.path.join(\"outputs\", \"knn_memory_bank_128d.npz\"))\n\n    # 6. Carregar Dados de Teste"
)

content = content.replace(
    "x_test_limpo, y_test, cnn, agent_mlp, agent_knn, mapa_ilhas",
    "x_test_limpo, y_test, cnn, agent_mlp, agent_knn, agent_knn_128d, mapa_ilhas"
)
content = content.replace(
    "x_test_ruido, y_test, cnn, agent_mlp, agent_knn, mapa_ilhas",
    "x_test_ruido, y_test, cnn, agent_mlp, agent_knn, agent_knn_128d, mapa_ilhas"
)

old_resumo = """    logger.info("=" * 72)
    logger.info("  RESUMO FINAL")
    logger.info("=" * 72)
    logger.info(f"  {'Cenário':<35} {'Híbrido+MLP':>14} {'Híbrido+k-NN':>14} {'Δ':>10}")
    logger.info("-" * 72)
    for r in [r_limpo, r_ruido]:
        d = r['acc_hibrida_knn'] - r['acc_hibrida_mlp']
        s = "+" if d >= 0 else ""
        logger.info(f"  {r['nome']:<35} {r['acc_hibrida_mlp']:>13.1f}% {r['acc_hibrida_knn']:>13.1f}% {s}{d:>8.1f}%")
    logger.info("=" * 72)

    # 10. Tabela de métricas em formato para copiar para o README
    logger.info("\\n--- TABELA PARA O README (Markdown) ---")
    logger.info("| Métrica | MLP (Q-Network) | k-NN Bandit |")
    logger.info("|---|---|---|")
    logger.info(f"| Híbrido (Limpas) | {r_limpo['acc_hibrida_mlp']:.1f}% | {r_limpo['acc_hibrida_knn']:.1f}% |")
    logger.info(f"| Híbrido (Ruído) | {r_ruido['acc_hibrida_mlp']:.1f}% | {r_ruido['acc_hibrida_knn']:.1f}% |")
    logger.info(f"| Agente Isolado (Limpas) | {r_limpo['acc_mlp_isolada']:.1f}% | {r_limpo['acc_knn_isolada']:.1f}% |")
    logger.info(f"| Agente Isolado (Ruído) | {r_ruido['acc_mlp_isolada']:.1f}% | {r_ruido['acc_knn_isolada']:.1f}% |")
    logger.info(f"| Inferência (ms) | {r_ruido['tempo_mlp']*1000:.1f} | {r_ruido['tempo_knn']*1000:.1f} |")
    logger.info(f"| Treino | 10 épocas + backprop | 0 épocas (memória) |")
    logger.info(f"| Parâmetros | ~8,714 (128→64→10) | 0 (non-parametric) |")"""

new_resumo = """    logger.info("=" * 85)
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
    logger.info("\\n--- TABELA PARA O README (Markdown) ---")
    logger.info("| Métrica | MLP (Q-Network) | k-NN 10D | k-NN Improved (128D) | Gap (128D vs MLP) |")
    logger.info("|---|---|---|---|---|")
    logger.info(f"| Híbrido (Limpas) | {r_limpo['acc_hibrida_mlp']:.1f}% | {r_limpo['acc_hibrida_knn']:.1f}% | {r_limpo['acc_hibrida_knn_128d']:.1f}% | {r_limpo['acc_hibrida_knn_128d'] - r_limpo['acc_hibrida_mlp']:+.1f}% |")
    logger.info(f"| Híbrido (Ruído σ=0.6) | {r_ruido['acc_hibrida_mlp']:.1f}% | {r_ruido['acc_hibrida_knn']:.1f}% | {r_ruido['acc_hibrida_knn_128d']:.1f}% | {r_ruido['acc_hibrida_knn_128d'] - r_ruido['acc_hibrida_mlp']:+.1f}% |")
    logger.info(f"| Agente Isolado (Limpas) | {r_limpo['acc_mlp_isolada']:.1f}% | {r_limpo['acc_knn_isolada']:.1f}% | {r_limpo['acc_knn_128d_isolada']:.1f}% | {r_limpo['acc_knn_128d_isolada'] - r_limpo['acc_mlp_isolada']:+.1f}% |")
    logger.info(f"| Agente Isolado (Ruído) | {r_ruido['acc_mlp_isolada']:.1f}% | {r_ruido['acc_knn_isolada']:.1f}% | {r_ruido['acc_knn_128d_isolada']:.1f}% | {r_ruido['acc_knn_128d_isolada'] - r_ruido['acc_mlp_isolada']:+.1f}% |")
    logger.info(f"| Inferência (ms) | {r_ruido['tempo_mlp']*1000:.1f} | {r_ruido['tempo_knn']*1000:.1f} | {r_ruido['tempo_knn_128d']*1000:.1f} | - |")
    logger.info(f"| Treino | 10 épocas + backprop | 0 (memória) | 0 (memória) | - |")
    logger.info(f"| Parâmetros | ~8,714 (128→64→10) | 0 (non-parametric) | 0 (non-parametric) | - |")"""
content = content.replace(old_resumo, new_resumo)

with open("benchmark_comparativo.py", "w", encoding="utf-8") as f:
    f.write(content)
