import os
from functools import partial
import numpy as np

import neat                        # pip install neat-python
import mnist                       # pip install mnist
from sklearn.utils import shuffle  # pip install scikit-learn
import skimage as ski                       # pip install scikit-image
from scipy.ndimage import affine_transform  # pip install scipy

PROCESS = True


def eval_genomes(genomes, config, resample=False, num_samples=2000):
    if resample:
        prep_dataset(num_samples)
    for genome_id, genome in genomes:
        eval_genome(genome, config)


def eval_genomes_async(genomes, config, resample=False, num_samples=2000):
    if resample:
        prep_dataset(num_samples)
    evaluator.evaluate(genomes, config)


def eval_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    correct = 0
    for xi, yi in xy:
        output = net.activate(xi)
        if max(output) == min(output):
            correct = 0
            break
        prediction = output.index(max(output))
        correct += 1 if prediction == yi else 0
    genome.fitness = correct / total


def eval_genome_async(genome, config, xy=None, total=None):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    correct = 0
    for xi, yi in xy:
        output = net.activate(xi)
        if max(output) == min(output):
            correct = 0
            break
        prediction = output.index(max(output))
        correct += 1 if prediction == yi else 0
    return correct / total


def prep_dataset(num_samples=2000):
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

    global xy, total
    x, y = shuffle(mnist.test_images(), mnist.test_labels(), n_samples=num_samples)
    total = len(y)
    xy = []
    for xi, yi in zip(x, y):
        if PROCESS:
            xi = ski.transform.resize(deskew(xi), (16, 16), preserve_range=True, anti_aliasing=False)
        xy.append((xi.flatten(), yi))


def run_neat(pop, num_samples=2000, resample=False, multithreading=False):
    global evaluator

    prep_dataset(num_samples)

    print('Running neat evaluation...')
    tuples = [(k, v) for k, v in pop.population.items()]
    if multithreading:
        print(f'### Using {os.cpu_count()} cpu cores for multithreading')
        evaluator = neat.parallel.ParallelEvaluator(num_workers=os.cpu_count(),
                                                    eval_function=partial(eval_genome_async, xy=xy, total=total))
        # winner = pop.run(partial(eval_genomes_async, resample=resample, num_samples=num_samples), num_generations)
        eval_genomes_async(tuples, pop.config, resample=resample, num_samples=num_samples)
    else:
        # winner = pop.run(partial(eval_genomes, resample=resample, num_samples=num_samples), num_generations)
        eval_genomes(tuples, pop.config, resample=resample, num_samples=num_samples)

    return max(pop.population.values(), key=lambda x: x.fitness)

