import datetime
import logging
import math

import numpy as np

from collections import defaultdict

logger = logging.getLogger('gclass')

logger.propagate = False
logger.setLevel(logging.INFO)
__fmt = logging.Formatter(fmt='%(asctime)s.%(msecs)03d: %(message)s', datefmt='%d/%m/%y %H:%M:%S')
__handler = logging.StreamHandler()
__handler.setFormatter(__fmt)
logger.addHandler(__handler)

default_alphabet = list(" abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789`~!@#$%^&*()_+-=[]{};'\\:\"|,./<>?")

def levenstein(seq1, seq2):
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = np.zeros((size_x, size_y))
    for x in range(size_x):
        matrix [x, 0] = x
    for y in range(size_y):
        matrix [0, y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x-1] == seq2[y-1]:
                matrix [x,y] = min(
                        matrix[x-1, y] + 1,
                        matrix[x-1, y-1],
                        matrix[x, y-1] + 1
                        )
            else:
                matrix [x,y] = min(
                        matrix[x-1,y] + 1,
                        matrix[x-1,y-1] + 1,
                        matrix[x,y-1] + 1
                        )

    return (matrix[size_x - 1, size_y - 1])


ACTION_GROW = 0
ACTION_SHRINK = 1
ACTION_MUTATE = 2
class GeneticAlgorithm:
    def __init__(self, target, alphabet=default_alphabet):
        self.target = target
        self.alphabet = alphabet

        self.start_time = datetime.datetime.now()

        self.mutation_rate = 0.01

    def generate_parent(self, length):
        return np.random.choice(self.alphabet, length, replace=True)

    def get_fitness_levenstein(self, guess):
        edits = levenstein(self.target, guess)
        return edits / len(self.target)

    def get_fitness(self, guess):
        sim = 0
        for expected, actual in zip(self.target, guess):
            if expected == actual:
                sim += 1
        return len(self.target) - sim

    def display(self, guess):
        time_diff = datetime.datetime.now() - self.start_time
        fitness = self.get_fitness(guess)

        logger.info('{}: fitness: {:.4f}'.format(str(time_diff), fitness))
        return fitness

    def mutate1(self, parent):
        mutations = 1
        locations = np.random.randint(0, len(parent), 1)
        letters = list(np.random.choice(self.alphabet, mutations))
        alterations = list(np.random.choice(self.alphabet, mutations))

        child = parent.copy()
        for loc, letter, alt in zip(locations, letters, alterations):
            if letter == child[loc]:
                child[loc] = alt
            else:
                child[loc] = letter

        return child

    def mutate(self, parent):
        child = parent.copy()

        action_set = [ACTION_GROW, ACTION_SHRINK, ACTION_MUTATE]
        #action_set = [ACTION_MUTATE]

        num_ops = math.ceil(len(child) * self.mutation_rate)
        actions = np.random.choice(action_set, num_ops, replace=True)

        anum = defaultdict(int)
        for a in actions:
            anum[a] += 1

        inserts = anum.get(ACTION_GROW, 0)
        if inserts > 0:
            locations = np.random.randint(0, len(child), inserts)
            locations = sorted(locations)
            letters = np.random.choice(self.alphabet, inserts)

            for idx, (loc, letter) in enumerate(zip(locations, letters)):
                loc += idx
                c0 = child[:loc]
                c1 = child[loc:]
                letter = np.array([letter])
                child = np.concatenate([c0, letter, c1], 0)

        deletions = anum.get(ACTION_SHRINK, 0)
        if deletions > 0:
            locations = np.random.randint(0, len(child), deletions)
            locations = sorted(locations)

            for idx, loc in enumerate(locations):
                loc -= idx
                c0 = child[:loc]
                c1 = child[loc+1:]
                child = np.concatenate([c0, c1], 0)

        mutations = anum.get(ACTION_MUTATE, 0)
        if mutations > 0:
            locations = np.random.randint(0, len(child), mutations)
            letters = np.random.choice(self.alphabet, mutations)
            alterations = np.random.choice(self.alphabet, mutations)

            for loc, letter, alt in zip(locations, letters, alterations):
                if letter == child[loc]:
                    child[loc] = alt
                else:
                    child[loc] = letter

        return child

def main():
    np.set_printoptions(formatter={'float': '{:0.4f}'.format, 'int': '{:4d}'.format}, linewidth=250, suppress=True, threshold=np.inf)

    target = "AnhsadflkjhsOHkjdsahf437hfsdlfknsdfkljLKJDhaszdasd3872r23dGHSAdasgkdhgaisdSADG*&sdKJhdja"
    train = GeneticAlgorithm(target)

    parent = train.generate_parent(len(target))
    best_fitness = train.display(parent)

    while best_fitness != 0:
        child = train.mutate(parent)
        fitness = train.get_fitness(child)

        if fitness < best_fitness:
            best_fitness = fitness
            parent = child
            fitness = train.display(child)

if __name__ == '__main__':
    main()
