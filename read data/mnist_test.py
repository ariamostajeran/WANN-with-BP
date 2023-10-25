import os

import mnist
import neat
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import skimage as ski                       # pip install scikit-image
from scipy.ndimage import affine_transform  # pip install scipy

from net import Net
from utils import *

NUM_TRAIN = 20
NUM_TEST = 20
TRAIN_EPOCHS = 15
SAVE_MODELS = False
PROCESS = True
# CHECKPOINT = 'neat_pop_1159'  # leave empty for test fully-connected network
CHECKPOINT = None  # leave empty for test fully-connected network


def prep_dataset():
    def moments(image):
        c0, c1 = np.mgrid[:image.shape[0], :image.shape[1]]  # A trick in numPy to create a mesh grid
        totalImage = np.sum(image)  # sum of pixels
        m0 = np.sum(c0 * image) / totalImage  # mu_x
        m1 = np.sum(c1 * image) / totalImage  # mu_y
        m00 = np.sum((c0 - m0) ** 2 * image) / totalImage  # var(x)
        m11 = np.sum((c1 - m1) ** 2 * image) / totalImage  # var(y)
        m01 = np.sum((c0 - m0) * (c1 - m1) * image) / totalImage  # covariance(x,y)
        mu_vector = np.array([m0, m1])  # Notice that these are \mu_x, \mu_y respectively
        covariance_matrix = np.array([[m00, m01], [m01, m11]])  # Do you see a similarity between the covariance matrix
        return mu_vector, covariance_matrix

    def deskew(image):
        c, v = moments(image)
        alpha = v[0, 1] / v[0, 0]
        affine = np.array([[1, 0], [alpha, 1]])
        ocenter = np.array(image.shape) / 2.0
        offset = c - np.dot(affine, ocenter)
        img = affine_transform(image, affine, offset=offset)
        return (img - img.min(initial=0)) / (img.max(initial=1) - img.min(initial=0))

    for xt in x_train_np:
        if PROCESS:
            x_train.append(ski.transform.resize(deskew(xt), (16, 16), preserve_range=True, anti_aliasing=False).flatten())
        else:
            x_train.append(xt.flatten())
    for xt in x_test_np:
        if PROCESS:
            x_test.append(ski.transform.resize(deskew(xt), (16, 16), preserve_range=True, anti_aliasing=False).flatten())
        else:
            x_test.append(xt.flatten())


x_train_np, y_train, x_test_np, y_test = mnist.train_images()[:NUM_TRAIN], mnist.train_labels()[:NUM_TRAIN], \
                                         mnist.test_images()[:NUM_TEST], mnist.test_labels()[:NUM_TEST]
x_train = []
x_test = []
prep_dataset()

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
                                       y_test=y_test, lr=0.1):
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
