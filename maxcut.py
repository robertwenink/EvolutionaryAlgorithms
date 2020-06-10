import numpy as np
import os

class MaxCut:

    def __init__(self, filename, instances_directory, opt_directory, calc_opt=False):
        '''
        initialize MaxCut problem from filename \n
        param file : file containing problem instance

        '''

        self.edges_dict = {}
        self.edges_tuples = {}
        self.edges_list = []

        self.length_genotypes = 0

        with open(instances_directory + filename, "r") as f:
            nodes = int(f.readline())
            lines = f.readlines()

            self.length_genotypes = nodes
            # Faster numpy array for speedy fitness checking
            self.fast_fit = np.zeros((nodes, nodes))

            for i, line in enumerate(lines):
                edge = line.split()
                weight = int(edge[2])
                edge = sorted(edge[:2])
                node_1, node_2 = int(edge[0]), int(edge[1])

                if node_1 not in self.edges_dict:
                    self.edges_dict[node_1] = {}
                self.edges_dict[node_1][node_2] = weight

                self.edges_tuples[tuple([node_1, node_2])] = weight

                self.edges_list.append(tuple([node_1, node_2, weight]))

                self.fast_fit[node_1, node_2] = weight
                self.fast_fit[node_2, node_1] = weight 

        genotypes = self.np_generate_random_genotype_population(3)
        fit1 = self.np_fitness(genotypes[0])
        fit2 = self.np_fitness(genotypes[1])
        fit3 = self.np_fitness(genotypes[2])

        fitn = self.np_fitness_population(genotypes)
        
        if os.path.exists(opt_directory + filename):
            with open(opt_directory + filename, "r") as f2:
                self.opt = int(f2.readline())
        elif calc_opt:
            self.opt = self.brute_force_opt()
            with open(opt_directory + filename, "w") as f2:
                f2.write(str(self.opt))

    def np_fitness(self, genotype):
        '''
        Method for calculating fitness given genotype as input \n
        Input: list or np.array of bits \n
        Output: total weight of edges

        '''

        return np.dot(genotype, np.matmul(self.fast_fit, genotype == 0))


    def fitness(self, genotype):
        '''
        Method for calculating fitness given genotype as input
        Input: list or np.array of bits
        Output: tuple of objective values

        '''

        objective = 0
        for edge in self.edges_tuples:
            if genotype[edge[0]] != genotype[edge[1]]:
                objective += self.edges_tuples[edge]

        # Normalize objectives
        # objective = objective / float(self.opt)
        return np.int64(objective)

    def np_fitness_population(self, genotypes):
        '''
        Method for calculating fitness for numpy array of genotypes
        We can change the matrix multiplication because fast_fit is symmetric

        '''
        # between = 
        return np.matmul(genotypes, np.matmul(genotypes == 0, self.fast_fit))

    def np_generate_random_genotype(self):
        '''
        Method for generating a random genotype of length self.length_genotypes \n
        Ouput: random numpy array of bits

        '''
        return np.random.randint(2, size=self.length_genotypes)


    def np_generate_random_genotype_population(self, population_size):
        '''
        Method for generating a random genotype of length self.length_genotypes \n
        Ouput: random numpy array of bits

        '''
        return np.random.randint(2, size=(population_size, self.length_genotypes))


    def compare(self, genotype_1, genotype_2):
        '''
        Method for comparing 2 genotypes \n
        Ouput: True if fitness of genotype_1 >= fitness of genotype_2

        '''
        return self.np_fitness(genotype_1) >= self.np_fitness(genotype_2)


    def brute_force_opt(self):
        '''
        Method for brute forcing the optimum calculation \n
        Output: MaxCut problem solved

        '''
        # TODO:
        # look online for brute force solver of MaxCut problem?
        return -1