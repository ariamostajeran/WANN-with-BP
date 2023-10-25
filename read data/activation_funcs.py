import numpy as np


class sigmoid:
    @classmethod
    def calc(cls, x):
        return 1 / (1 + np.exp(-x)) if x > 0 else np.exp(x) / (1+np.exp(x))

    @classmethod
    def grad(cls, x):
        """Gradient of sigmoid function. Assumes input has already passed through sigmoid!"""
        return x * (1 - x)


class relu:
    @classmethod
    def calc(cls, x):
        return max(0, x)

    @classmethod
    def grad(cls, x):
        """Gradient of relu function. Assumes input has already passed through relu!"""
        return 1 if x > 0 else 0


class tanh:
    @classmethod
    def calc(cls, x):
        return np.tanh(x)

    @classmethod
    def grad(cls, x):
        """Gradient of tanh function. Assumes input has already passed through tanh!"""
        return 1 - x**2


class softmax:  # can be used to convert output to probabilities (in classification)
    @classmethod
    def calc(cls, x):
        e_x = np.exp(x - np.max(x))  # Subtracting np.max(x) for numerical stability
        return e_x / e_x.sum()

    @classmethod
    def grad(cls, x, k):
        """x must be a softmax array already!"""
        # return np.outer(x, 1 - x)
        grad = -x[k] * (1 - x)
        grad[k] = x[k] * (1 - x[k])
        return grad


class sin:
    @classmethod
    def calc(cls, x):
        return np.sin(x)

    @classmethod
    def grad(cls, x):
        return np.cos(x)


class identity:
    @classmethod
    def calc(cls, x):
        return x

    @classmethod
    def grad(cls, x):
        return 1


class gauss:
    @classmethod
    def calc(cls, x):
        return np.exp(-x**2)

    @classmethod
    def grad(cls, x):
        return -2 * x * np.exp(-x**2)


activation_funcs = {  # pls update with other functions you add
    'sigmoid': sigmoid,
    'relu': relu,
    'tanh': tanh,
    'sin': sin,
    'identity': identity,
    'gauss': gauss,
}
