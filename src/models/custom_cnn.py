import tensorflow as tf
from src.scratch.layers import DenseLayer
from src.scratch.activations import relu, softmax

class RawModel(tf.Module):
    def __init__(self):
        super().__init__()
        # Exemplo simples: Flatten + 2 Camadas Densas "feitas à mão"
        self.flatten = tf.keras.layers.Flatten() # Usa keras apenas para fazer reshape
        self.dense1 = DenseLayer(28*28, 128, name="dense1")
        self.dense2 = DenseLayer(128, 10, name="classifier")

    def __call__(self, x: tf.Tensor) -> tf.Tensor:
        x = self.flatten(x)
        x = self.dense1(x)
        x = relu(x)
        x = self.dense2(x)
        return softmax(x)