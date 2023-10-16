import numpy as np
from reprlib import recursive_repr


class Node:
    def __init__(self, key=None, activation=None, learning_rate=0.1):
        self.weights = {}  # only incoming weights are stored! key=pre_node
        self.bias = 0  # TODO
        self.key = key
        self.pre_nodes = []
        self.post_nodes = []
        self.activation = activation

        self.input = None  # only used for input nodes
        self.output = 0
        self.target = None  # only used for output nodes
        self.gradient = 0
        self.learning_rate = learning_rate
        self.net = None
        self.new_weights = {}

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

    def init_weights(self):
        self.weights = {}
        for pre_node in self.pre_nodes:
            # TODO
            self.weights[pre_node] = np.random.uniform(low=0.0, high=0.1)  # random float between 0 and 1

    def forward(self):
        # Calculate the weighted sum of inputs
        # TODO not taking neat aggregation functions into account
        if self.pre_nodes:
            weighted_sum = sum([weight * pre.output for pre, weight in self.weights.items()])
        else:
            weighted_sum = self.input

        weighted_sum += self.bias

        # Apply an activation function (e.g., sigmoid)
        self.output = self.activation.calc(weighted_sum) if self.activation else weighted_sum

        # Forward the output to connected neurons in the next layer
        return self.post_nodes

    def calculate_gradient(self):
        if not self.post_nodes:  # we are an output node
            self.gradient = self.output - self.target
        elif not self.pre_nodes:  # we are an input node, won't need to update weights
            pass
        else:
            weighted_sum = sum([post_node.weights[self] * post_node.gradient for post_node in self.post_nodes])
            if self.activation:
                self.gradient = self.activation.grad(self.output) * weighted_sum
            else:
                # If there's no activation function, assume it's linear
                self.gradient = weighted_sum

    def calculate_new_weights(self):
        # Update weights using gradient descent
        self.new_weights = {}
        for pre_node, weight in self.weights.items():
            self.new_weights[pre_node] = weight - (self.learning_rate * self.gradient * pre_node.output)

    def update_weights(self):
        if self.new_weights or not self.pre_nodes:
            self.weights = self.new_weights
            self.new_weights = {}
        else:
            raise Exception('No new weights to apply, call .calculate_new_weights() first')

    def backward(self):
        self.calculate_gradient()
        self.calculate_new_weights()

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
