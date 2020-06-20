from maxcut import MaxCut
import os
from simple_GA import GeneticAlgorithm


instances_directory = 'simple_GA_instances/'
opt_directory = 'opts/'
sub_directory = ''

files = os.listdir(instances_directory + sub_directory)
print(files)
instances = []
for f in files:
    if f[0] != '.':
        instance = MaxCut(f, instances_directory, opt_directory)
        instances.append(instance)

#opt for 4 nodes is 14, 8 nodes is 106, 16 is 366, 32 is 1365, 64 4998

ga = GeneticAlgorithm(MaxCut = instances[3], population_size= 10000, generations = 10, crossover_probability=0.8, mutation_probability = 0.2, opt= 8900)

ga.run()                                    # run the GA
print(ga.best_individual())              # print the GA's best solution