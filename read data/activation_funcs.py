import numpy as np


class sigmoid:
    def __call__(self, x):
        return 1 / (1 + np.exp(-x))

    def grad(self, x):
        sig_x = self(x)
        return sig_x * (1 - sig_x)


def relu(x):
    return max(0, x)


def softmax(x):  # can be used to convert output to probabilities (in classification)
    e_x = np.exp(x - np.max(x))  # Subtracting np.max(x) for numerical stability
    return e_x / e_x.sum()


activation_funcs = {  # pls update with other functions you add
    'sigmoid': sigmoid,
    'relu': relu,
    'softmax': softmax,
}