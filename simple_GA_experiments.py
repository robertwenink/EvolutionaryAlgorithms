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

#opt for 4 nodes is 27, 8 nodes is 145, 16 is 412, 32 is 1365, 64 4998
problem = instances[5]
opt = 145
popsize = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
pop_black = 500
pop_grey = 10
for j in popsize:
    count = 0
    for i in range(10):
        ga_black = GeneticAlgorithm(MaxCut=problem, population_size=j, generations=5, crossover_probability=0.8,
                                    mutation_probability=0.1, opt=opt, limit_evals=1000000)
        ga_black.run()
        # print('Blackbox' + str(ga_black.best_individual()))            # print the GA's best solution
        # print(ga_black.best_individual()[3])
        if ga_black.best_individual()[0] == opt:
            count += 1
    if count == 10:
        print(j)
        pop_black = j
        break

popsize_grey = [2, 4, 8, 16, 32]
for j in popsize_grey:
    count = 0
    for i in range(10):
        ga_grey = GeneticAlgorithm_grey_box(MaxCut=problem, population_size=j, generations=5, crossover_probability=0.8,
                                            mutation_probability=0.1, opt=opt, local_k=j, limit_evals=1000000)
        ga_grey.run()
        # print('Blackbox' + str(ga_black.best_individual()))            # print the GA's best solution
        # print(ga_black.best_individual()[3])
        if ga_grey.best_individual()[0] == opt:
            count += 1
    if count == 10:
        print(j)
        pop_grey = j
        break



for i in range(10):
    ga_black = GeneticAlgorithm(MaxCut=problem, population_size=pop_black, generations=5, crossover_probability=0.8,
                                mutation_probability=0.1, opt=opt, limit_evals=1000000)
    ga_grey = GeneticAlgorithm_grey_box(MaxCut=problem, population_size=pop_grey, generations=5, crossover_probability=0.8,
                                        mutation_probability=0.1, opt=opt, local_k=pop_grey, limit_evals=1000000)
    ga_black.run()
    # run the GA
    print('Blackbox' + str(ga_black.best_individual()))            # print the GA's best solution

    ga_grey.run()                                    # run the GA
    print('Greybox' + str(ga_grey.best_individual()))            # print the GA's best solution