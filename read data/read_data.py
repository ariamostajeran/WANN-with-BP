import re
from nodes import Node

nodes = {}
model_file = open("model.txt", 'r')
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
        nodes[key] = Node(key=key, activation=activation)

    if flag == "connection":
        enabled = bool(re.findall("enabled=(.*)\)", line)[0])
        # print(enabled)
        if enabled:
            first_key = (int(re.findall("key=\((.*?), ", line)[0]))
            second_key = (int(re.findall("key=.*, (.*?)\), w", line)[0]))
            
            nodes[first_key].add_post(nodes[second_key])
            nodes[second_key].add_pre(nodes[first_key])









