class Node:
    def __init__(self, key, pre_nodes=[], post_nodes=[], activation="linear"):
        self.weight = 0
        self.bias = 0
        self.key = key
        self.pre_nodes = pre_nodes
        self.post_nodes = post_nodes
        self.activation = activation

    def add_pre(self, node):
        self.pre_nodes.append(node)

    def add_post(self, node):
        self.post_nodes.append(node)

