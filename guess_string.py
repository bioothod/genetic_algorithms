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

default_alphabet = np.array(list(" abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789`~!@#$%^&*()_+-=[]{};'\\:\"|,./<>?"))

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

    def generate_random(self, length):
        return np.random.default_rng().choice(self.alphabet, length, replace=True)

    def get_fitness_levenstein(self, guess):
        edits = levenstein(self.target, guess)
        return edits / len(self.target)

    def get_fitness(self, genes):
        min_len = min(len(genes), len(self.target))
        max_len = max(len(genes), len(self.target))
        t = self.target[:min_len]
        g = genes[:min_len]
        equal = np.count_nonzero(t == g)
        return equal / max_len

class Organism:
    def __init__(self, genes, fitness, alphabet):
        self.genes = genes
        self.fitness = fitness
        self.alphabet = alphabet

        self.mutation_rate = 0.05

    def mate(self, other):
        min_len = min(len(other.genes), len(self.genes))
        o = other.genes[:min_len]
        g = self.genes[:min_len]
        rnd_index = np.random.rand(min_len)
        new_genes = np.where(rnd_index < 0.5, g, o)

        if len(other.genes) != len(self.genes):
            if np.random.rand() >= 0.5:
                if len(other.genes) > len(self.genes):
                    new_genes = np.concatenate([new_genes, other.genes[min_len:]])
                else:
                    new_genes = np.concatenate([new_genes, self.genes[min_len:]])

        return Organism(new_genes, 0, alphabet=self.alphabet)

    def mutate1(self):
        mutations = 1
        locations = np.random.randint(0, len(self.genes), mutations)
        letters = np.random.default_rng().choice(self.alphabet, mutations)

        self.genes[locations] = letters

    def mutate(self):
        action_set = [ACTION_GROW, ACTION_SHRINK, ACTION_MUTATE]
        #action_set = [ACTION_MUTATE]

        #actions = [ACTION_MUTATE]
        #anum = {ACTION_MUTATE: 1}

        num_ops = math.ceil(len(self.genes) * self.mutation_rate)
        actions = np.random.default_rng().choice(action_set, num_ops, replace=True)

        anum = defaultdict(int)
        for a in actions:
            anum[a] += 1

        inserts = anum.get(ACTION_GROW, 0)
        if inserts > 0:
            locations = np.random.randint(0, len(self.genes), inserts)
            locations = sorted(locations)
            letters = np.random.default_rng().choice(self.alphabet, inserts)

            for idx, (loc, letter) in enumerate(zip(locations, letters)):
                loc += idx
                c0 = self.genes[:loc]
                c1 = self.genes[loc:]
                letter = np.array([letter])
                self.genes = np.concatenate([c0, letter, c1], 0)

        deletions = anum.get(ACTION_SHRINK, 0)
        if deletions > 0:
            locations = np.random.randint(0, len(self.genes), deletions)
            locations = sorted(locations)

            for idx, loc in enumerate(locations):
                loc -= idx
                c0 = self.genes[:loc]
                c1 = self.genes[loc+1:]
                self.genes = np.concatenate([c0, c1], 0)

        mutations = anum.get(ACTION_MUTATE, 0)
        if mutations > 0:
            loc_idx_rnd = np.random.rand(len(self.genes))
            loc_idx = np.where(loc_idx_rnd < self.mutation_rate)
            locations = np.arange(len(self.genes))[loc_idx]
            letters = np.random.default_rng().choice(self.alphabet, locations.shape[0])

            self.genes[locations] = letters


class Population:
    def __init__(self, train, size, initial_size=10, mating_proportion=0.1):
        self.train = train

        self.population = []
        for i in range(size):
            p = self.train.generate_random(initial_size)
            fit = self.train.get_fitness(p)

            o = Organism(p, fit, alphabet=self.train.alphabet)
            self.population.append(o)

    def best_fitness_organism(self):
        bfo = None
        for o in self.population:
            if bfo == None or o.fitness > bfo.fitness:
                bfo = o

        return bfo

    def mate(self):
        eps = 1e-6
        fits = np.array([o.fitness + eps for o in self.population])
        fits_sum = np.sum(fits)
        probs = fits / fits_sum

        num_pairs = len(self.population)
        pairs = np.random.default_rng().choice(self.population, num_pairs*2, p=probs, replace=True)

        new_population = []
        for i in range(num_pairs):
            p0 = pairs[2*i+0]
            p1 = pairs[2*i+1]

            child = p0.mate(p1)
            child.mutate()

            child.fitness = self.train.get_fitness(child.genes)

            new_population.append(child)

        total_population = self.population + new_population
        sorted_population = sorted(total_population, reverse=True, key=lambda o: o.fitness)

        self.population = sorted_population[:len(self.population)]

def main():
    np.set_printoptions(formatter={'float': '{:0.4f}'.format, 'int': '{:4d}'.format}, linewidth=250, suppress=True, threshold=np.inf)

    target = np.array(list("AnhsadflkjhsOHkjdsahf437hfsdlfknsdfkljLKJDhaszdasd3872r23dGHSAdasgkdhgaisdSADG*&sdKJhdja"))
    #target = np.array(list("dififjIOPJew89djwefe"))
    train = GeneticAlgorithm(target)

    pop = Population(train, size=500, initial_size=len(target) * 10)
    bfo = pop.best_fitness_organism()
    best_fitness = bfo.fitness

    gen_num = 0
    logger.info('{}: best_fitness: {:.3f}, genes: {}/{}'.format(gen_num, bfo.fitness, len(bfo.genes), len(target)))

    while best_fitness != 1:
        pop.mate()
        gen_num += 1
        bfo = pop.best_fitness_organism()

        if bfo.fitness > best_fitness:
            best_fitness = bfo.fitness
            logger.info('{}: best_fitness: {:.3f}, genes: {}/{}'.format(gen_num, bfo.fitness, len(bfo.genes), len(target)))

        #logger.info('{}: fitness: {:.3f}, best_fitness: {:.3f}'.format(gen_num, fitness, best_fitness))

if __name__ == '__main__':
    main()
