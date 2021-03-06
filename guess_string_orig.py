import random

import numpy as np

geneSet = " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789`~!@#$%^&*()_+-=[]{};'\\:\"|,./<>?"
target = "AnhsadflkjhsOHkjdsahf437hfsdlfknsdfkljLKJDhaszdasd3872r23dGHSAdasgkdhgaisdSADG*&sdKJhdja"

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

def generate_parent(length):
    genes = []
    while len(genes) < length:
        sampleSize = min(length - len(genes), len(geneSet))
        genes.extend(random.sample(geneSet, sampleSize))
    return np.array(genes)

def get_fitness(guess):
    #edits = levenstein(target, guess)
    #return len(target) - edits

    sim = 0
    for expected, actual in zip(target, guess):
        if expected == actual:
            sim += 1
    return sim

xxx = np.array(list(geneSet))

def mutate(parent):
    mutations = 1
    if True:
        locations = np.random.randint(0, len(parent), mutations)
        #locations = [random.randrange(0, len(parent))]
        letters = np.random.default_rng().choice(xxx, mutations*2, replace=False)
        alterations = letters[mutations:]
        letters = letters[:mutations]
        child = parent.copy()
    else:
        locations = [random.randrange(0, len(parent))]
        letters, alterations = random.sample(geneSet, 2)
        child = list(parent)

    child[locations[0]] = alterations[0] \
        if letters[0] == child[locations[0]] \
        else letters[0]

    return child

def mutate1(parent):
    index = random.randrange(0, len(parent))
    childGenes = list(parent)
    newGene, alternate = random.sample(geneSet, 2)

    childGenes[index] = alternate \
        if newGene == childGenes[index] \
        else newGene
    return ''.join(childGenes)

import datetime


def main():
    random.seed()
    startTime = datetime.datetime.now()
    bestParent = generate_parent(len(target))
    bestFitness = get_fitness(bestParent)

    def display(guess):
        timeDiff = datetime.datetime.now() - startTime
        fitness = get_fitness(guess)
        guess_str = ''.join(guess)
        print("{0}\t{1}\t{2}".format(guess_str, fitness, str(timeDiff)))

    display(bestParent)
    while True:
        child = mutate(bestParent)
        childFitness = get_fitness(child)
        if bestFitness >= childFitness:
            continue
        display(child)
        if childFitness >= len(bestParent):
            break
        bestFitness = childFitness
        bestParent = child

if __name__ == '__main__':
    main()

