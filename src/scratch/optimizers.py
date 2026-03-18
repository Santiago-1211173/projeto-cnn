import tensorflow as tf
from typing import List, Tuple

class SGD:
    """Implementação 'from scratch' do Stochastic Gradient Descent."""
    def __init__(self, learning_rate: float = 0.01):
        self.lr = learning_rate

    def apply_gradients(self, variables: List[tf.Variable], gradients: List[tf.Tensor]) -> None:
        """Atualiza os tensores aplicando o gradiente negativo escalado pela LR."""
        for var, grad in zip(variables, gradients):
            if grad is not None:
                # assign_sub é a operação atómica do TensorFlow para: var = var - (lr * grad)
                # Isto acontece diretamente na memória da GPU sem ir ao CPU.
                var.assign_sub(self.lr * grad)