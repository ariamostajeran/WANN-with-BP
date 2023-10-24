import os

import mnist
import neat
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from net import Net
from utils import *

NUM_TRAIN = 2000
NUM_TEST = 2000
TRAIN_EPOCHS = 5
SAVE_MODELS = False
CHECKPOINT = 'neat_pop_2011'  # leave empty for test fully-connected network
# CHECKPOINT = None  # leave empty for test fully-connected network

x_train_np, y_train, x_test_np, y_test = mnist.train_images()[:NUM_TRAIN], mnist.train_labels()[:NUM_TRAIN], \
                                         mnist.test_images()[:NUM_TEST], mnist.test_labels()[:NUM_TEST]
x_train = []
x_test = []
for item in x_train_np:
    x_train.append(item.flatten() / 255)
for item in x_test_np:
    x_test.append(item.flatten() / 255)

if CHECKPOINT:
    net = Net.from_checkpoint(CHECKPOINT)
else:
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         os.path.join(os.path.dirname(__file__), 'config.cfg'))
    genome = neat.DefaultGenome(69)
    genome.configure_new(config.genome_config)
    genome.connect_full_nodirect(config.genome_config)
    print(genome.size())  # (nodes, connections)
    net = Net.from_genome(genome, config)


print('## before training:')
for i in range(10):
    res = net(x_train[i])
    print(f'pred: {res.argmax()} true: {y_train[i]}')

loss_list_train = []
loss_list_test = []
for train_loss, test_loss in net.train(x_train, y_train, epochs=TRAIN_EPOCHS, save=SAVE_MODELS, x_test=x_test,
                                       y_test=y_test):
    loss_list_train.append(train_loss)
    loss_list_test.append(test_loss)
    print(f'\tloss: {train_loss}')

print('## after training:')
for i in range(10):
    res = net(x_train[i])
    print(f'pred: {res.argmax()} true: {y_train[i]}')

outputs = []
for input in x_train:
    outputs.append(net(input))

matplotlib.use('TkAgg')  # temp fix for Pycharm
plt.plot(loss_list_train, label='Training loss')
plt.plot(loss_list_test, label='Test loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title(f'Train ac: {accuracy(outputs, y_train)} | Test ac: {accuracy([net(input) for input in x_test], y_test)}')
plt.legend()
plt.show()
