import tensorflow as tf
from typing import List

class SGD:
    """A 'from scratch' implementation of Stochastic Gradient Descent (SGD)."""
    def __init__(self, learning_rate: float = 0.01):
        self.lr = learning_rate

    def apply_gradients(self, variables: List[tf.Variable], gradients: List[tf.Tensor]) -> None:
        """Updates tensors by applying the negative gradient scaled by the learning rate."""
        for var, grad in zip(variables, gradients):
            if grad is not None:
                # assign_sub is TensorFlow's atomic operation for: var = var - (lr * grad)
                # This occurs directly in GPU memory without CPU involvement.
                var.assign_sub(self.lr * grad)