import os

import mnist
import neat
import numpy as np

from net import Net
from activation_funcs import softmax

x_np, y = mnist.test_images()[:200], mnist.test_labels()[:200]
x = []
for item in x_np:
    x.append(item.flatten() / 255)

config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     os.path.join(os.path.dirname(__file__), 'config.cfg'))

genome = neat.DefaultGenome(69)
genome.configure_new(config.genome_config)
genome.connect_full_nodirect(config.genome_config)
print(genome.size())  # (nodes, connections)


net = Net.from_genome(genome, config)

print('## before training:')
# for i in range(10):
#     res = net(x[i])
#     print(f'pred: {res.argmax()} true: {y[i]} {res}')


net.train(x, y, epochs=10)

print('## after training:')
for i in range(10):
    res = net(x[i])
    print(f'pred: {res.argmax()} true: {y[i]} {res}')
