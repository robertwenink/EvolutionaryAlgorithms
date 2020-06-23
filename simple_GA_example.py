from simple_GA import GeneticAlgorithm
import numpy as np
from time import time

import os
from particle_swarm import BinaryParticleSwarmOptimization
from maxcut import MaxCut

instances_directory = 'instances/'
opt_directory = 'opts/'
sub_directory = ''

files = os.listdir(instances_directory + sub_directory)
instances = []
for f in files:
    if f[0] != '.':
        instance = MaxCut(f, instances_directory, opt_directory)
        instances.append(instance)



ga = GeneticAlgorithm(MaxCut = instances[3], population_size= 500, generations = 10, crossover_probability=0.8, mutation_probability = 0.2, opt= 8900)

ga.run()                                    # run the GA
print(ga.best_individual())              # print the GA's best solution