from FastGA.fastga.operators import FastMutationOperator
import copy
import numpy as np


class FastGA:

    def __init__(self, instance, pop_size=50, max_gen=500):
        '''
        Initialize the Binary Particle Swarm Optimization class

        '''

        # Instance
        self.instance = instance

        # fitness function
        self.calculate_fitness = instance.np_fitness

        # generator
        self.generate_random_genotype = instance.np_generate_random_genotype

        # set no max generations
        self.max_gen = max_gen

        # init best individual so far
        self.best_individual_pop = self.generate_random_genotype()

        # Set high value to best swarm error
        self.best_fitness_pop = 0

        # Init fit list
        self.pop_fit = []

        # init population and fitness
        self.population = [self.generate_random_genotype() for i in range(pop_size)]
        self.pop_fit = [self.instance.np_fitness(self.population[i]) for i in range(pop_size)]

        # Population size
        self.pop_size = pop_size



    def run(self):

        operator = FastMutationOperator(n=self.instance.length_genotypes, beta=1.5)

        for i in range(self.max_gen):
            print("## gen " + str(i) + " ##")
            for j in range(self.pop_size):
                offspring = copy.deepcopy(self.population[j])
                operator.mutate(offspring, inplace=True)
                fit_offspring = self.instance.np_fitness(offspring)
                if fit_offspring > self.pop_fit[j]:
                    self.population[j] = offspring
                    self.pop_fit[j] = fit_offspring
                    self.best_fitness_pop = max(fit_offspring, self.best_fitness_pop)

            print("Max fitness " + str(self.best_fitness_pop))