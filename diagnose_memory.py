"""Diagnóstico rápido da memória k-NN para identificar problemas de qualidade."""
import numpy as np

data = np.load("outputs/knn_memory_bank.npz")
states = data["states"]
actions = data["actions"]
rewards = data["rewards"]

print(f"Total experiências: {len(states):,}")
print(f"Recompensas positivas: {np.sum(rewards > 0):,} ({np.mean(rewards > 0)*100:.1f}%)")
print(f"Recompensas negativas: {np.sum(rewards < 0):,} ({np.mean(rewards < 0)*100:.1f}%)")

# Entropia média dos estados (quão "confusos" são os outputs da CNN)
# Softmax uniforme = alta entropia = CNN confusa
entropias = -np.sum(states * np.log(states + 1e-10), axis=1)
max_entropia = -np.log(1/10)  # Entropia máxima (uniforme)
print(f"\nEntropia média dos estados: {np.mean(entropias):.3f} / {max_entropia:.3f}")
print(f"% estados quase uniformes (entropia > 2.0): {np.mean(entropias > 2.0)*100:.1f}%")

# Confiança máxima média (o quanto a CNN está segura)
max_probs = np.max(states, axis=1)
print(f"Confiança máxima média da CNN: {np.mean(max_probs):.3f}")
print(f"% estados com confiança < 0.2: {np.mean(max_probs < 0.2)*100:.1f}%")

# Analisar os primeiros 60k (época 1) vs últimos 60k (época 3)
n_tercio = len(states) // 3
for i, nome in enumerate(["Época 1", "Época 2", "Época 3"]):
    sl = slice(i * n_tercio, (i + 1) * n_tercio)
    pct_pos = np.mean(rewards[sl] > 0) * 100
    ent_med = np.mean(entropias[sl])
    print(f"  {nome}: {pct_pos:.1f}% positivas | Entropia média: {ent_med:.3f}")
