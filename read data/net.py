from collections import deque
import numpy as np

from node import Node
from activation_funcs import *


class Net:
    def __init__(self, nodes=[]):
        self.nodes = nodes

        if self.nodes:
            for node in self.nodes:
                node.init_weights()
                node.net = self
            self.input_nodes = [node for node in self.nodes if not node.pre_nodes]
            self.output_nodes = [node for node in self.nodes if not node.post_nodes]

    def activate(self, inputs):
        if len(inputs) != len(self.input_nodes):
            raise ValueError(f'wrong input shape: expecting {len(self.input_nodes)} but got {len(inputs)}')
        q = deque()  # FIFO
        qs = set()
        for node, input in zip(self.input_nodes, inputs):
            node.input = input
            nxt_set = set(node.forward())
            diff = nxt_set.difference(qs)
            q.extend(diff)
            [qs.add(x) for x in diff]
        while len(q) != 0:
            node = q.popleft()
            nxt_set = set(node.forward())
            diff = nxt_set.difference(qs)
            q.extend(diff)
            [qs.add(x) for x in diff]

        return softmax([node.output for node in self.output_nodes])

    def __call__(self, inputs):
        return self.activate(inputs)

    def train(self, inputs, targets, epochs=1, verbose=True):
        max_target = max(targets)
        assert max_target+1 == len(self.output_nodes)
        for epoch in range(epochs):
            if verbose:
                print(f'training... (epoch {epoch})')
            for x, y in zip(inputs, targets):
                loss = self.activate(x)
                y_lst = np.zeros(max_target+1)
                y_lst[y] = 1
                q = deque()  # FIFO
                qs = set()
                for i, node in enumerate(self.output_nodes):
                    node.target = y_lst[i]
                    node.loss = loss[i]
                    node.calculate_output_error()
                    nxt_set = set(node.backward())
                    diff = nxt_set.difference(qs)
                    q.extend(diff)
                    [qs.add(x) for x in diff]
                while len(q) != 0:
                    node = q.popleft()
                    nxt_set = set(node.backward())
                    diff = nxt_set.difference(qs)
                    q.extend(diff)
                    [qs.add(x) for x in diff]

    @classmethod
    def from_genome(cls, genome, config):
        nodes = {}

        for key in config.genome_config.input_keys:
            nodes[key] = Node(key=key)

        for key, node_neat in genome.nodes.items():  # only includes hidden- and output-nodes
            nodes[key] = Node(  # TODO add other Node options (bias, etc.)
                key=key,
                activation=activation_funcs.get(node_neat.activation, None)
            )

        for con in genome.connections.values():
            pre = con.key[0]
            post = con.key[1]

            if con.enabled:
                nodes[pre].add_post(nodes[post], con.weight)

        return cls(list(nodes.values()))

    @classmethod
    def from_file(cls):
        # TODO implement regex from read_data.py here
        raise NotImplementedError
