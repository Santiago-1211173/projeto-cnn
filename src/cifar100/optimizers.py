"""
Pure TensorFlow AdamW optimizer implemented from scratch with decoupled weight decay.
Supports both standard TF grads_and_vars tuples and legacy (variables, gradients) calling conventions.
"""

from typing import List, Tuple, Union, Optional
import tensorflow as tf


class Adam:
    """
    AdamW optimizer with decoupled weight decay (Loshchilov & Hutter, ICLR 2019).
    Applies weight decay exclusively to multi-dimensional weight matrices/tensors (dim > 1),
    sparing 1D biases and batch normalization scale/shift parameters.
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        weight_decay: float = 1e-4,
    ):
        self.lr = float(learning_rate)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.epsilon = float(epsilon)
        self.weight_decay = float(weight_decay)

        self.m = {}
        self.v = {}
        self.t = tf.Variable(0.0, trainable=False, dtype=tf.float32, name="adamw_step")

    @property
    def learning_rate(self) -> float:
        return self.lr

    @learning_rate.setter
    def learning_rate(self, value: float) -> None:
        self.lr = float(value)

    def apply_gradients(
        self,
        grads_and_vars: Optional[Union[List[Tuple[Optional[tf.Tensor], tf.Variable]], List[tf.Variable]]] = None,
        gradients: Optional[List[Optional[tf.Tensor]]] = None,
        variables: Optional[List[tf.Variable]] = None,
    ) -> None:
        """
        Applies gradients to variables.

        Accepts either:
            1. grads_and_vars: List[Tuple[tf.Tensor, tf.Variable]]
            2. variables: List[tf.Variable], gradients: List[tf.Tensor]
            3. Positional: apply_gradients(variables, gradients)
        """
        if variables is not None and gradients is not None:
            pairs = list(zip(gradients, variables))
        elif gradients is not None and grads_and_vars is not None:
            # Called positionally as apply_gradients(variables, gradients)
            pairs = list(zip(gradients, grads_and_vars))
        elif grads_and_vars is not None:
            # Called as apply_gradients(grads_and_vars)
            pairs = grads_and_vars
        else:
            raise ValueError("No gradients or variables provided to apply_gradients.")

        self.t.assign_add(1.0)
        lr_t = self.lr * (
            tf.sqrt(1.0 - tf.pow(self.beta2, self.t))
            / (1.0 - tf.pow(self.beta1, self.t))
        )

        for grad, var in pairs:
            if grad is None:
                continue

            var_key = var.ref()
            if var_key not in self.m:
                self.m[var_key] = tf.Variable(
                    tf.zeros_like(var), trainable=False
                )
                self.v[var_key] = tf.Variable(
                    tf.zeros_like(var), trainable=False
                )

            m_var = self.m[var_key]
            v_var = self.v[var_key]

            # 1st and 2nd moment estimates
            m_var.assign(self.beta1 * m_var + (1.0 - self.beta1) * grad)
            v_var.assign(self.beta2 * v_var + (1.0 - self.beta2) * tf.square(grad))

            step = lr_t * m_var / (tf.sqrt(v_var) + self.epsilon)

            # Decoupled weight decay: applied only to weight matrices (dim > 1), not 1D biases/BN params
            if self.weight_decay > 0.0 and len(var.shape) > 1:
                var.assign_sub(lr_t * self.weight_decay * var)

            var.assign_sub(step)


class SGDMomentum:
    """
    Stochastic Gradient Descent optimizer with Nesterov momentum and decoupled weight decay.
    Supports both standard TF grads_and_vars tuples and legacy (variables, gradients) conventions.
    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        momentum: float = 0.9,
        nesterov: bool = True,
        weight_decay: float = 5e-4,
    ):
        self.lr = float(learning_rate)
        self.momentum = float(momentum)
        self.nesterov = bool(nesterov)
        self.weight_decay = float(weight_decay)

        self.v = {}
        self.t = tf.Variable(0.0, trainable=False, dtype=tf.float32, name="sgd_step")

    @property
    def learning_rate(self) -> float:
        return self.lr

    @learning_rate.setter
    def learning_rate(self, value: float) -> None:
        self.lr = float(value)

    def apply_gradients(
        self,
        grads_and_vars: Optional[Union[List[Tuple[Optional[tf.Tensor], tf.Variable]], List[tf.Variable]]] = None,
        gradients: Optional[List[Optional[tf.Tensor]]] = None,
        variables: Optional[List[tf.Variable]] = None,
    ) -> None:
        if variables is not None and gradients is not None:
            pairs = list(zip(gradients, variables))
        elif gradients is not None and grads_and_vars is not None:
            pairs = list(zip(gradients, grads_and_vars))
        elif grads_and_vars is not None:
            pairs = grads_and_vars
        else:
            raise ValueError("No gradients or variables provided to apply_gradients.")

        self.t.assign_add(1.0)
        for grad, var in pairs:
            if grad is None:
                continue

            var_key = var.ref()
            if var_key not in self.v:
                self.v[var_key] = tf.Variable(
                    tf.zeros_like(var), trainable=False
                )

            v_var = self.v[var_key]
            v_var.assign(self.momentum * v_var + grad)

            if self.nesterov:
                step = self.momentum * v_var + grad
            else:
                step = v_var

            # Decoupled weight decay on multi-dimensional tensors (weights)
            if self.weight_decay > 0.0 and len(var.shape) > 1:
                var.assign_sub(self.lr * self.weight_decay * var)

            var.assign_sub(self.lr * step)
