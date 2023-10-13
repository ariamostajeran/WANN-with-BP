import numpy as np
from reprlib import recursive_repr

from activation_funcs import softmax


class Node:
    def __init__(self, key=None, activation=None, learning_rate=0.1):
        self.weights = {}  # only incoming weights are stored! key=pre_node.key
        self.bias = 0  # TODO
        self.key = key
        self.pre_nodes = []
        self.post_nodes = []
        self.activation = activation() if activation else None

        self.input = None  # only used for input nodes
        self.output = 0
        self.target = None  # only used for output nodes
        self.loss = None  # only used for output nodes
        self.error = 0
        self.gradient = 0
        self.learning_rate = learning_rate
        self.net = None

    def add_pre(self, node, weight):
        if node not in self.pre_nodes:
            self.pre_nodes.append(node)
        if self not in node.post_nodes:
            node.post_nodes.append(self)
            node.weights[self] = weight

    def add_post(self, node, weight):
        if node not in self.post_nodes:
            self.post_nodes.append(node)
        if self not in node.pre_nodes:
            node.pre_nodes.append(self)
            self.weights[node] = weight

    def init_weights(self):
        new_weights = {}
        for pre_node in self.pre_nodes:
            new_weights[pre_node] = np.random.uniform(low=0.0, high=0.1)  # random float between 0 and 1
        self.weights = new_weights

    def forward(self):
        # Calculate the weighted sum of inputs
        # TODO not taking neat aggregation functions into account
        if self.pre_nodes:
            weighted_sum = sum(w * pre.output for pre, w in self.weights.items())
        else:
            weighted_sum = self.input

        # Apply an activation function (e.g., sigmoid)
        self.output = self.activation(weighted_sum) if self.activation else weighted_sum

        # Forward the output to connected neurons in the next layer
        return self.post_nodes

    def calculate_output_error(self):
        # multi-class categorical cross-entropy loss
        # Calculate the error at this neuron's output
        # self.error = (self.output - self.target) ** 2
        self.error = - np.sum(self.target * np.log(self.loss))

    def calculate_gradient(self):
        if not self.post_nodes:  # we are an output node
            self.gradient = np.clip(self.loss, 1e-15, 1 - 1e-15) - self.target
        elif not self.pre_nodes:  # we are an input node, won't need to update weights
            pass
        else:
            self.gradient = self.activation.grad(self.output)
        # self.gradient = self.error * self.output * (1 - self.output)

    def update_weights(self):
        # Update weights using gradient descent
        new_weights = {}
        for pre_node, weight in self.weights.items():
            new_weights[pre_node] = weight - (self.learning_rate * self.gradient * pre_node.output)
        self.weights = new_weights

    def backward(self):
        self.calculate_gradient()
        self.update_weights()

        # Backpropagate the error to connected neurons in the previous layer
        for pre_node in self.pre_nodes:
            pre_node.error += self.weights[pre_node] * self.gradient

        return self.pre_nodes

    @recursive_repr()
    def __repr__(self):
        pre_nodes = [node.key for node in self.pre_nodes]
        post_nodes = [node.key for node in self.post_nodes]
        weights = {k.key: v for k, v in self.weights.items()}
        dic = {k: v for k, v in self.__dict__.items() if k not in ['pre_nodes', 'post_nodes', 'weights', 'net']}
        dic['pre_nodes'] = pre_nodes
        dic['post_nodes'] = post_nodes
        dic['weights'] = weights
        return repr(dic)
