import tensorflow as tf
from typing import List

class Adam:
    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-7):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}
        self.v = {}
        self.t = tf.Variable(0.0, trainable=False, dtype=tf.float32)

    def apply_gradients(self, variables: List[tf.Variable], gradients: List[tf.Tensor]) -> None:
        self.t.assign_add(1.0)
        lr_t = self.lr * (tf.sqrt(1.0 - tf.pow(self.beta2, self.t)) / (1.0 - tf.pow(self.beta1, self.t)))

        for var, grad in zip(variables, gradients):
            if grad is None:
                continue
            var_key = var.ref()
            if var_key not in self.m:
                self.m[var_key] = tf.Variable(tf.zeros_like(var), trainable=False)
                self.v[var_key] = tf.Variable(tf.zeros_like(var), trainable=False)

            m_var = self.m[var_key]
            v_var = self.v[var_key]

            m_var.assign(self.beta1 * m_var + (1.0 - self.beta1) * grad)
            v_var.assign(self.beta2 * v_var + (1.0 - self.beta2) * tf.square(grad))

            step = lr_t * m_var / (tf.sqrt(v_var) + self.epsilon)
            var.assign_sub(step)

w = tf.Variable([1.0, 2.0])
opt = Adam()

@tf.function
def step():
    opt.apply_gradients([w], [tf.constant([0.1, 0.2])])

step()
step()
print("Success! Updated w:", w.numpy())
