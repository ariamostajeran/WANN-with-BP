import re
import pickle
import os
import time

import neat  # pip install neat-python

from node import Node
from activation_funcs import *
from utils import *
from run_neat import run_neat


class Net:
    def __init__(self, nodes=[], config=None):
        self.nodes = nodes

        if self.nodes:
            self.input_nodes = []
            self.output_nodes = []
            for node in self.nodes:
                node.init_weights()
                node.net = self
                if node.key in config.genome_config.input_keys if config else\
                        (not node.pre_nodes and node.key <= -1):  # input node
                    self.input_nodes.append(node)
                elif node.key in config.genome_config.output_keys if config else\
                        (not node.post_nodes and node.key >= 0):  # output node
                    self.output_nodes.append(node)
                    node.delete(pre_nodes=False, post_nodes=True)
                    node.activation = None
                elif (not node.pre_nodes) or (not node.post_nodes):  # orphan
                    node.delete()
                    self.nodes.remove(node)
            self.nodes_forward = self._topological_sort()  # ensure correct ordering of nodes
            self.nodes_backward = self.nodes_forward[::-1]  # ensure correct ordering of nodes

    def _topological_sort(self):
        # Ensure that the graph is a DAG! (no cycles)
        visited = set()
        stack = []

        def dfs(node):
            if node not in visited:
                visited.add(node)
                for successor in node.post_nodes:
                    dfs(successor)
                stack.append(node)

        for input_node in self.input_nodes:
            dfs(input_node)

        # Reverse the order to get the topological sort
        return stack[::-1]

    def _infer(self, x):
        if len(x) != len(self.input_nodes):
            raise ValueError(f'wrong input shape: expecting {len(self.input_nodes)} but got {len(x)}')

        inputs_done = 0
        for node in self.nodes_forward:
            if node in self.input_nodes:
                i = self.input_nodes.index(node)
                node.input = x[i]
                inputs_done += 1
            node.forward()

        assert inputs_done == len(x)  # sanity check

        softmax_outputs = softmax.calc([node.output for node in self.output_nodes])

        for i, node in enumerate(self.output_nodes):
            node.output = softmax_outputs[i]  # done for proper error calculation during backpropagation

        return softmax_outputs

    def activate(self, inputs, batch=False):
        """Forward pass"""
        if batch:
            return [self._infer(input) for input in inputs]
        else:
            return self._infer(inputs)

    def predict(self, inputs, batch=False):
        """Forward pass. Alias to .activate()"""
        return self.activate(inputs, batch)

    def __call__(self, inputs, batch=False):
        """Forward pass"""
        return self.activate(inputs, batch)

    def train(self, x_train, y_train, epochs=1, lr=0.1, verbose=True, save=False, x_test=None, y_test=None):
        """Train the network. Generator function yielding loss each epoch

        :param x_train: training input data
        :param y_train: training input labels
        :param int epochs: number of epochs to train for, defaults to 1
        :param float lr: learning rate, defaults to 0.1
        :param bool verbose: Whether to print the start of each epoch, defaults to True
        :param bool save: Whether to save a checkpoint of this model every 5 epochs, defaults to False
        :param x_test: test input data if test loss is desired, defaults to None
        :param y_test: test input labels if test loss is desired, defaults to None
        :return: Yields the train loss and the test loss or None
        :rtype: (float, float or None)
        """
        max_target = max(y_train)
        assert max_target + 1 == len(self.output_nodes), f'{max_target + 1} is not {len(self.output_nodes)}'  # sanity check
        if save:
            save_dir = f'models/{int(time.time())}'
            os.makedirs(save_dir)
        for epoch in range(epochs):
            if verbose:
                print(f'training... (epoch {epoch})')
            outputs = []
            for x, y in zip(x_train, y_train):
                output = self.activate(x)
                outputs.append(output)
                y_lst = np.zeros(max_target + 1)
                y_lst[y] = 1
                for node in self.nodes_backward:
                    if node in self.output_nodes:
                        i = self.output_nodes.index(node)
                        assert node.key == i  # sanity check
                        node.target = y_lst[i]
                    node.backward(lr)
                for node in self.nodes_backward:
                    node.update_weights()
            if save and epoch % 5 == 0:
                with open(f'{save_dir}/net_{epoch}.pkl', 'wb') as file:
                    pickle.dump(self, file)
            if x_test is not None and y_test is not None:
                yield loss_batch(outputs, y_train), loss_batch(self.activate(x_test, batch=True), y_test)
            else:
                yield loss_batch(outputs, y_train), None

    @classmethod
    def from_genome(cls, genome, config):
        nodes = {}

        for key in config.genome_config.input_keys:
            nodes[key] = Node(key=key)

        for key, node_neat in genome.nodes.items():  # only includes hidden- and output-nodes
            nodes[key] = Node(
                key=key,
                activation=activation_funcs.get(node_neat.activation, None),
                bias=node_neat.bias
            )

        for con in genome.connections.values():
            if con.enabled:
                pre = con.key[0]
                post = con.key[1]
                nodes[pre].add_post(nodes[post], con.weight)

        return cls(list(nodes.values()), config)

    @classmethod
    def from_checkpoint(cls, path):
        # pop = neat.Checkpointer.restore_checkpoint(path).population
        pop = neat.Checkpointer.restore_checkpoint(path)
        genome = pop.best_genome
        config = pop.config
        if not genome:
            genome = run_neat(pop, multithreading=False)
            print(f'No best_genome found, using genome {genome.key} with highest fitness {genome.fitness}')
            print(genome)
        while not genome:
            print('No best_genome found, please enter a genome key from the following selection')
            print(pop.population.keys())
            genome = pop.population.get(int(input('ENTER GENOME KEY: ')))

        return cls.from_genome(genome, config)

    @classmethod
    def from_file(cls, path, input_size):
        nodes = {}
        for i in range(input_size):
            key = -(i + 1)
            nodes[key] = Node(key=key)
        with open(path, 'r') as model_file:
            lines = model_file.readlines()
            flag = "node"
            for line in lines:
                if "Nodes" in line:
                    flag = "node"
                    continue
                elif "Connections" in line:
                    flag = "connection"
                    continue

                if flag == "node":
                    key = int(re.findall("key=(.*?), b", line)[0])
                    bias = float(re.findall("bias=(.*?), r", line)[0])
                    weight = float(re.findall('response=(.*?), ac', line)[0])
                    activation = re.findall("activation=(.*?), ag", line)[0]
                    nodes[key] = Node(
                        key=key,
                        activation=activation_funcs.get(activation, None),
                        bias=bias
                    )

                if flag == "connection":
                    enabled = bool(re.findall("enabled=(.*)\)", line)[0])
                    if enabled:
                        weight = float(re.findall('weight=(.*?),', line)[0])
                        pre = (int(re.findall("key=\((.*?), ", line)[0]))
                        post = (int(re.findall("key=.*, (.*?)\), w", line)[0]))
                        nodes[pre].add_post(nodes[post], weight)

        return cls(list(nodes.values()))
