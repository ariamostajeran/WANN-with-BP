import numpy as np


class sigmoid:
    @classmethod
    def calc(cls, x):
        return 1 / (1 + np.exp(-x))

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
    def grad(cls, x):
        raise NotImplementedError


activation_funcs = {  # pls update with other functions you add
    'sigmoid': sigmoid,
    'relu': relu,
    'tanh': tanh,
}
