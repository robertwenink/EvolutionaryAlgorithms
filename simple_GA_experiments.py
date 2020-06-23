from maxcut import MaxCut
import os
from simple_GA import GeneticAlgorithm
from GreyBox_simple_GA import GeneticAlgorithm_grey_box



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

ga_black = GeneticAlgorithm(MaxCut = instances[1], population_size= 500, generations = 30, crossover_probability=0.8, mutation_probability = 0.1, opt= 1300)
ga_black.run()                                    # run the GA
print(ga_black.best_individual())              # print the GA's best solution

ga_grey = GeneticAlgorithm_grey_box(MaxCut = instances[1], population_size= 20, generations = 20, crossover_probability=0.8, mutation_probability = 0.1, opt=1300, local_k = 10)

#

ga_grey.run()                                    # run the GA
print(ga_grey.best_individual())              # print the GA's best solution