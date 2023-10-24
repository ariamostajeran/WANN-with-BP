import os
from functools import partial

import neat                        # pip install neat-python
import mnist                       # pip install mnist
from sklearn.utils import shuffle  # pip install scikit-learn


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


def prep_dataset(num_samples):
    global xy, total
    x, y = shuffle(mnist.test_images(), mnist.test_labels(), n_samples=num_samples)
    total = len(y)
    xy = []
    for xi, yi in zip(x, y):
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

