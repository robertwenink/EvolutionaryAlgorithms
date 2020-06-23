from FastGA.fastga.operators import FastMutationOperator
import copy
from FastGA.tabu.operators import TabuSearch
import numpy as np
import random


class FastGA:

    def __init__(self, instance, pop_size=50, max_gen=500, beta=3, alpha=3, opt=100000, gbo=False):
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

        # Init population
        self.population = [self.generate_random_genotype() for i in range(pop_size)]
        self.population = np.array(self.population)

        # Init fitness
        self.pop_fit = [self.instance.np_fitness(self.population[i]) for i in range(pop_size)]
        self.pop_fit = np.array(self.pop_fit)

        # Population size
        self.pop_size = pop_size

        # Set optimum value
        self.opt = opt

        # Use grey box optimization or not
        self.gbo = gbo

        # Set FastGA operator
        self.operator = FastMutationOperator(n=self.instance.length_genotypes, beta=beta)

        # Evaluations
        self.evaluations = 0

        # Set
        self.search = TabuSearch(self.instance, self, alpha=alpha)

    # Selection operator (Not used)
    def selection(self):
        # Get average fitness
        average_fit = np.average(self.pop_fit)
        # Choose individuals that have higher fitness than average
        survivors = self.population[self.pop_fit > average_fit]
        sur_fit = self.pop_fit[self.pop_fit > average_fit]
        for i in range(len(self.population)):
            try:
                index = random.randint(0, len(survivors)-1)
                self.population[i] = survivors[index]
                self.pop_fit[i] = sur_fit[index]
            except:
                None

    def inc_eval(self, evaluations=1):
        self.evaluations += evaluations

    # Black box FastGA
    def run_bbo(self):
        for j in range(self.pop_size):
            offspring = copy.deepcopy(self.population[j])
            self.operator.mutate(offspring, inplace=True)
            fit_offspring = self.instance.np_fitness(offspring)
            self.evaluations += 1
            if fit_offspring > self.pop_fit[j]:
                self.population[j] = offspring
                self.pop_fit[j] = fit_offspring

            # Set the maximum fitness
            if self.pop_fit[j] > self.best_fitness_pop:
                self.best_fitness_pop = self.pop_fit[j]
                self.best_individual_pop = self.population[j]

    # Grey Box FastGA with simple Tabu Search
    def run_gbo(self):
        for j in range(self.pop_size):

            # for k in range(100):
            offspring = copy.deepcopy(self.population[j])
            self.operator.mutate(offspring, inplace=True)
            fit_offspring = self.instance.np_fitness(offspring)
            self.evaluations += 1
            if fit_offspring > self.pop_fit[j]:
                self.population[j] = offspring
                self.pop_fit[j] = fit_offspring

            #  Tabu Search
            offspring = copy.deepcopy(self.population[j])
            offspring = self.search.tabu(offspring, inplace=False)
            self.population[j] = offspring
            self.pop_fit[j] = fit_offspring

            # Set the maximum fitness
            if self.pop_fit[j] > self.best_fitness_pop:
                self.best_fitness_pop = self.pop_fit[j]
                self.best_individual_pop = self.population[j]

    # Run the algorithm per generation
    def run(self):
        i = 0
        # for i in range(self.max_gen):
        while self.best_fitness_pop < self.opt:
            print("## gen " + str(i) + " ##")

            # Run grey box or black box optimization
            if self.gbo:
                self.run_gbo()
            else:
                self.run_bbo()

            # print("Avg fitness " + str(np.average(self.pop_fit)))
            print("Max fitness " + str(self.best_fitness_pop))
            print("Evaluations " + str(self.evaluations))
            i+=1

