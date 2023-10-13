import os

from net import Net
from activation_funcs import *


net = Net.from_file(os.path.normpath(os.path.join(os.path.dirname(__file__), 'model.txt')), 784)
print(net.output_nodes)
