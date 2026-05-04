with open("README.md", "r", encoding="utf-8") as f:
    content = f.read()

old_table = """### Benchmark Results

| Métrica | MLP (Q-Network) | k-NN 10D | k-NN Improved (128D) |
|---|---|---|---|
| **Híbrido (Limpas)** | 97.1% | **97.5%** | 96.8% |
| **Híbrido (Ruído σ=0.6)** | 86.6% | 75.7% | **88.9%** |
| Agente Isolado (Limpas) | 91.4% | 91.5% | 82.8% |
| Agente Isolado (Ruído) | 86.6% | 75.7% | **88.9%** |
| Inferência (ms) | **4.0** | 10105.8 | 3262.9 |
| Treino | 10 épocas + backprop | 0 (memória) | 0 (memória) |
| Parâmetros | ~8,714 (128→64→10) | 0 (non-parametric) | 0 (non-parametric) |"""

new_table = """### Benchmark Results

| Métrica | MLP (Q-Network) | k-NN 10D | k-NN Improved (128D) | Gap (128D vs MLP) |
|---|---|---|---|---|
| **Híbrido (Limpas)** | 97.1% | 97.5% | **97.8%** | +0.6% |
| **Híbrido (Ruído σ=0.6)** | 86.6% | 75.7% | **88.5%** | +1.9% |
| Agente Isolado (Limpas) | 91.4% | 91.5% | **97.6%** | +6.1% |
| Agente Isolado (Ruído) | 86.6% | 75.7% | **88.5%** | +1.9% |
| Inferência (ms) | **3.3** | 10088.5 | 3875.3 | - |
| Treino | 10 épocas + backprop | 0 (memória) | 0 (memória) | - |
| Parâmetros | ~8,714 (128→64→10) | 0 (non-parametric) | 0 (non-parametric) | - |"""
content = content.replace(old_table, new_table)

old_findings = """3. **The 128D Breakthrough:** By moving the k-NN agent from the 10D softmax probabilities to the **128D latent features**, we closed the 11% accuracy gap. The new **k-NN Improved (128D)** agent achieves **88.9%** on noisy images, beating the original MLP target (86.6%) by **+2.3%**, while strictly remaining a non-parametric model with zero backpropagation.
4. **Trade-off:** The 128D k-NN agent requires zero gradient-based training (the memory is populated in a single forward pass) and beats the MLP on accuracy, making it highly attractive. However, inference is slower than the MLP (3.2s vs 4ms for 10k images)."""

new_findings = """3. **The 128D Breakthrough:** By moving the k-NN agent from the 10D softmax probabilities to the **128D latent features** and populating the episodic memory with both clean and noisy images (Distribution Shift correction), we closed the 11% accuracy gap and completely surpassed the MLP. The new **k-NN Improved (128D)** agent achieves **88.5%** on noisy images and **97.8%** on clean images, beating the original MLP targets (86.6% and 97.1%) by **+1.9% and +0.6% respectively**, while strictly remaining a non-parametric model with zero backpropagation.
4. **Trade-off:** The 128D k-NN agent requires zero gradient-based training (the memory is populated in a single forward pass) and beats the MLP on accuracy across all scenarios, making it highly attractive. However, inference is slower than the MLP (3.8s vs 3ms for 10k images)."""
content = content.replace(old_findings, new_findings)

with open("README.md", "w", encoding="utf-8") as f:
    f.write(content)
