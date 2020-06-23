"""
    pyeasyga module
"""

import random
import copy
from operator import attrgetter
import numpy as np

from six.moves import range


class GeneticAlgorithm_grey_box(object):
    """Genetic Algorithm class.
    This is the main class that controls the functionality of the Genetic
    Algorithm.
    A simple example of usage:
    # >>> # Select only two items from the list and maximise profit
    # >>> from pyeasyga.pyeasyga import GeneticAlgorithm
    # >>> input_data = [('pear', 50), ('apple', 35), ('banana', 40)]
    # >>> easyga = GeneticAlgorithm(input_data)
    # >>> def fitness (member, data):
    # >>>     return sum([profit for (selected, (fruit, profit)) in
    # >>>                 zip(member, data) if selected and
    # >>>                 member.count(1) == 2])
    # >>> easyga.fitness_function = fitness
    # >>> easyga.run()
    # >>> print easyga.best_individual()
    """

    def __init__(self,
                 MaxCut,
                 population_size,
                 generations,
                 crossover_probability,
                 mutation_probability,
                 opt,
                 local_k,
                 elitism=True,
                 maximise_fitness=True,
                 verbose=False,
                 random_state=None):
        """Instantiate the Genetic Algorithm.
        :param seed_data: input data to the Genetic Algorithm
        :type seed_data: list of objects
        :param int population_size: size of population
        :param int generations: number of generations to evolve
        :param float crossover_probability: probability of crossover operation
        :param float mutation_probability: probability of mutation operation
        :param int: random seed. defaults to None
        """

        self.maxcut = MaxCut
        self.population_size = population_size
        self.generations = generations
        self.crossover_probability = crossover_probability
        self.mutation_probability = mutation_probability
        self.elitism = elitism
        self.maximise_fitness = maximise_fitness
        self.verbose = verbose
        self.opt = opt
        self.evals = 0
        self.opt_evals = -1
        self.fittest_individual = -1
        self.local_k =local_k


        # seed random number generator
        self.random = random.Random(random_state)

        self.fitness_function = self.maxcut.np_fitness


        self.current_generation = []

        def create_individual(maxcut):
            """Create a candidate solution representation.
            e.g. for a bit array representation:
            return [random.randint(0, 1) for _ in range(len(data))]
            :param seed_data: input data to the Genetic Algorithm
            :type seed_data: list of objects
            :returns: candidate solution representation as a list
            """
            return [self.random.randint(0, 1) for _ in range(maxcut.length_genotypes)]

        def group_crossover(parent_1, parent_2, parent_3):
            
        def crossover(parent_1, parent_2):
            """Crossover (mate) two parents to produce two children.
            :param parent_1: candidate solution representation (list)
            :param parent_2: candidate solution representation (list)
            :returns: tuple containing two children
            """
            index = self.random.randrange(1, len(parent_1))
            child_1 = parent_1[:index] + parent_2[index:]
            child_2 = parent_2[:index] + parent_1[index:]
            return child_1, child_2

        def mutate(individual):
            """Reverse the bit of a random index in an individual."""
            mutate_index = self.random.randrange(len(individual))
            individual[mutate_index] = (0, 1)[individual[mutate_index] == 0]

        def random_selection(population):
            """Select and return a random member of the population."""
            return self.random.choice(population)

        def tournament_selection(population):
            """Select a random number of individuals from the population and
            return the fittest member of them all.
            """
            if self.tournament_size == 0:
                self.tournament_size = 2
            members = self.random.sample(population, self.tournament_size)
            members.sort(
                key=attrgetter('fitness'), reverse=self.maximise_fitness)
            return members[0]

        self.tournament_selection = tournament_selection
        self.tournament_size = self.population_size // 10
        print(self.tournament_size)
        self.random_selection = random_selection
        self.create_individual = create_individual
        self.crossover_function = crossover
        self.mutate_function = mutate
        self.selection_function = self.tournament_selection

    def create_initial_population(self):
        """Create members of the first population randomly.
        """
        initial_population = []
        for _ in range(self.population_size):
            genes = self.create_individual(self.maxcut)
            individual = Chromosome(genes)
            initial_population.append(individual)
        self.current_generation = initial_population

    def calculate_population_fitness(self):
        """Calculate the fitness of every member of the given population using
        the supplied fitness_function.
        """
        for individual in self.current_generation:
            individual.fitness = self.fitness_function(np.asarray(individual.genes))
            # print(str(individual.genes) + ' pre ' + str(individual.fitness))
            # individual.genes, individual.fitness = self.local_search(individual, 5)
            # print(str(individual.genes) + ' post ' + str(individual.fitness))

            if individual.fitness >= self.opt and self.opt_evals == -1:
                self.opt_evals = self.evals
            self.evals += 1

    def rank_population(self):
        """Sort the population by fitness according to the order defined by
        maximise_fitness.
        """
        self.current_generation.sort(
            key=attrgetter('fitness'), reverse=self.maximise_fitness)

    def create_new_population(self):
        """Create a new population using the genetic operators (selection,
        crossover, and mutation) supplied.
        """
        new_population = []
        elite = copy.deepcopy(self.current_generation[0])

        selection = self.selection_function

        while len(new_population) < self.population_size:
            parent_1 = copy.deepcopy(selection(self.current_generation))
            parent_2 = copy.deepcopy(selection(self.current_generation))

            child_1, child_2 = parent_1, parent_2
            child_1.fitness, child_2.fitness = 0, 0

            can_crossover = self.random.random() < self.crossover_probability
            can_mutate = self.random.random() < self.mutation_probability

            if can_crossover:
                child_1.genes, child_2.genes = self.crossover_function(
                    parent_1.genes, parent_2.genes)

            if can_mutate:
                self.mutate_function(child_1.genes)
                self.mutate_function(child_2.genes)

            child_1.fitness = self.fitness_function(np.asarray(child_1.genes))
            self.evals += 1



            #print(str(child_1.genes) + ' pre ' + str(child_1.fitness))
            child_1.genes, child_1.fitness = self.local_search(child_1)
            #print(str(child_1.genes) + ' post ' + str(child_1.fitness))

            if child_1.fitness >= self.opt and self.opt_evals == -1:
                self.opt_evals = self.evals

            #if child_1 not in new_population:
            new_population.append(child_1)

            if len(new_population) < self.population_size:
                child_2.fitness = self.fitness_function(np.asarray(child_2.genes))
                self.evals += 1
                #print(str(child_2.genes) + ' pre2 ' + str(child_2.fitness))
                child_2.genes, child_2.fitness = self.local_search(child_2)
                #print(str(child_2.genes) + ' post2 ' + str(child_2.fitness))
                #if child_2 not in new_population:
                new_population.append(child_2)

        if self.elitism:
            new_population.append(elite)

        self.current_generation = new_population

    def create_first_generation(self):
        """Create the first population, calculate the population's fitness and
        rank the population by fitness according to the order specified.
        """
        self.create_initial_population()
        self.calculate_population_fitness()
        self.rank_population()

    def create_next_generation(self):
        """Create subsequent populations, calculate the population fitness and
        rank the population by fitness in the order specified.
        """
        self.create_new_population()
        #self.calculate_population_fitness()
        self.rank_population()
        if self.verbose:
            print("Fitness: %f" % self.best_individual()[0])

    def run(self):
        """Run (solve) the Genetic Algorithm."""
        self.create_first_generation()

        for _ in range(1, self.generations):
            self.create_next_generation()

    def best_individual(self):
        """Return the individual with the best fitness in the current
        generation.
        """
        best = self.current_generation[0]
        return (best.fitness, best.genes, self.opt_evals)



    def last_generation(self):
        """Return members of the last generation as a generator function."""
        return ((member.fitness, member.genes) for member
                in self.current_generation)

    def local_search(self, individual):
        best_fitness = individual.fitness
        best_genes = copy.deepcopy(individual.genes)
        for j in range(self.local_k):
            round_max_fitness = best_fitness
            round_max_genes = copy.deepcopy(best_genes)
            temp_genes = copy.deepcopy(round_max_genes)
            for i in range(len(individual.genes)):
                genes = copy.deepcopy(temp_genes)
                if genes[i] == 1:
                    genes[i] = 0
                else:
                    genes[i] = 1
                new_fitness = self.fitness_function(np.asarray(genes))
                self.evals += 1
                if new_fitness > round_max_fitness:
                    round_max_fitness = new_fitness
                    round_max_genes = copy.deepcopy(genes)
            temp_genes = copy.deepcopy(round_max_genes)
            if round_max_fitness == best_fitness:
                break
            best_fitness = round_max_fitness
            best_genes = copy.deepcopy(round_max_genes)
        # better = Chromosome(best_genes)
        # better.fitness = best_fitness
        return best_genes, best_fitness




class Chromosome(object):
    """ Chromosome class that encapsulates an individual's fitness and solution
    representation.
    """
    def __init__(self, genes):
        """Initialise the Chromosome."""
        self.genes = genes
        self.fitness = 0

    def __repr__(self):
        """Return initialised Chromosome representation in human readable form.
        """
        return repr((self.fitness, self.genes))

    def __eq__(self, other):
        if self.fitness == other.fitness and self.genes == other.genes:
            return True
        return False