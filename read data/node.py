import numpy as np
from reprlib import recursive_repr
import sys


class Node:
    def __init__(self, key=None, activation=None, bias=0):
        self.weights = {}  # only incoming weights are stored! key=pre_node
        self.bias = bias
        self.key = key
        self.pre_nodes = []
        self.post_nodes = []
        self.activation = activation

        self.input = None  # only used for input nodes
        self.output = 0
        self.target = None  # only used for output nodes
        self.gradient = 0
        self.net = None
        self.new_weights = {}
        self.new_bias = None

    def add_pre(self, node, weight):
        if node not in self.pre_nodes:
            self.pre_nodes.append(node)
            self.weights[node] = weight
        if self not in node.post_nodes:
            node.post_nodes.append(self)

    def add_post(self, node, weight):
        if node not in self.post_nodes:
            self.post_nodes.append(node)
        if self not in node.pre_nodes:
            node.pre_nodes.append(self)
            node.weights[self] = weight

    def delete(self, pre_nodes=True, post_nodes=True):
        if pre_nodes:
            for node in self.pre_nodes:
                node.post_nodes.remove(self)
            self.pre_nodes = []
        if post_nodes:
            for node in self.post_nodes:
                node.pre_nodes.remove(self)
                node.weights.pop(self)
            self.post_nodes = []

    def init_weights(self):
        self.weights = {}
        for pre_node in self.pre_nodes:
            # TODO set high to 1.0? results seem to decrease if I do...
            self.weights[pre_node] = np.random.uniform(low=0.0, high=0.1)  # random float between 0 and 1

    def forward(self):
        # Calculate the weighted sum of inputs
        if self.pre_nodes:
            weighted_sum = np.clip(sum([weight * pre.output for pre, weight in self.weights.items()]), sys.float_info.min, sys.float_info.max)
        else:
            weighted_sum = self.input

        weighted_sum += self.bias

        # Apply an activation function (e.g., sigmoid)
        self.output = self.activation.calc(weighted_sum) if self.activation else weighted_sum

    def calculate_gradient(self):
        if not self.pre_nodes:  # we are an input node, won't need to update weights
            pass
        elif not self.post_nodes:  # we are an output node
            self.gradient = self.output - (self.target if self.target else 0)
        else:
            weighted_sum = sum([post_node.weights[self] * post_node.gradient for post_node in self.post_nodes])
            weighted_sum += self.bias * self.gradient  # Include bias in the gradient calculation
            if self.activation:
                self.gradient = np.clip(self.activation.grad(self.output) * weighted_sum, sys.float_info.min, sys.float_info.max)
            else:
                # If there's no activation function, assume it's linear
                self.gradient = weighted_sum

    def calculate_new_weights(self, lr):
        # Update weights using gradient descent
        self.new_bias = self.bias - (lr * self.gradient)  # Update the bias
        self.new_weights = {}
        for pre_node, weight in self.weights.items():
            self.new_weights[pre_node] = weight - (lr * self.gradient * pre_node.output)

    def update_weights(self):
        if (self.new_weights and self.new_bias is not None) or not self.pre_nodes:
            self.weights = self.new_weights
            self.bias = self.new_bias
            self.new_weights = {}
            self.new_bias = None
        else:
            raise Exception(f'No new weights to apply for node {self.key}, call .calculate_new_weights() first')

    def backward(self, lr):
        self.calculate_gradient()
        self.calculate_new_weights(lr)

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
