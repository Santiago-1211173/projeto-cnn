# Ficheiro: src/scratch/loss_activation.py
import torch
from src.scratch.activations import Activation_Softmax
from src.scratch.losses import Loss_CategoricalCrossentropy

class Activation_Softmax_Loss_CategoricalCrossentropy:
    """
    O ponto de ignição do Backward Pass.
    Combina o Softmax e a Loss para evitar cálculos redundantes e instabilidade.
    """
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossentropy()

    def forward(self, inputs, y_true):
        # Ida: Produz probabilidades e calcula o erro médio
        self.activation.forward(inputs)
        self.output = self.activation.output
        return self.loss.forward(self.output, y_true)

    def backward(self, dvalues, y_true):
        amostras = len(dvalues)
        
        # 1. O gradiente começa como uma cópia exata das probabilidades (Previsão)
        self.dinputs = dvalues.clone()
        
        # 2. A MAGIA VETORIZADA: Subtrair 1 APENAS na probabilidade da classe certa.
        # range(amostras) -> [0, 1, 2, ... 255] (Todas as imagens do batch)
        # y_true          -> [7, 2, 1, ... 4]   (A coluna da classe certa de cada imagem)
        self.dinputs[range(amostras), y_true] -= 1
        
        # 3. Normalização pelo Batch
        # Se somarmos 256 correções, o passo de atualização seria gigante.
        # Dividimos pelas amostras para manter o gradiente estável, independentemente do Batch Size.
        self.dinputs = self.dinputs / amostras