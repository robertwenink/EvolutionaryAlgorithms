import numpy as np
import os
from scipy.sparse.csgraph import minimum_spanning_tree

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
            self.fast_fit = np.zeros((nodes, nodes), dtype=np.int)

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

        # test1 = self.np_generate_random_genotype_population(100)
        # for gen in test1:
        #     print(self.fitness(np.logical_not(gen)) == self.np_fitness(gen))

        self.max_spanning_tree_genotype = self.calculate_max_spanning_tree_genotype()
        temp1 = self.np_fitness(self.max_spanning_tree_genotype)
        self.max_degree_greedy_genotype = self.calculate_max_degree_weight_genotype()
        temp2 = self.np_fitness(self.max_degree_greedy_genotype)

        print(temp1, temp2)

        self.count_evals = 0
    

    def np_fitness(self, genotype):
        '''
        Method for calculating fitness given genotype as input \n
        Input: list or np.array of bits \n
        Output: total weight of edges

        '''

        self.count_evals += 1
        return np.einsum('i, ik, k', genotype, self.fast_fit, genotype == 0)


    def fitness(self, genotype):
        '''
        Method for calculating fitness given genotype as input
        Input: list or np.array of bits
        Output: tuple of objective values

        '''

        self.count_evals += 1
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
        # old = np.diagonal(np.matmul(genotypes, np.matmul(self.fast_fit, np.transpose(genotypes == 0))))
        self.count_evals += genotypes.shape[0]
        return np.einsum('ij,jk,ik->i', genotypes, self.fast_fit, genotypes == 0)

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


    def np_local_search_genotype(self, genotype):
        '''
        Calculate local search optimum of genotype

        '''
        local_pop = self.np_generate_local_population(genotype)
        return local_pop[np.argmax(self.np_fitness_population(local_pop))]


    def np_local_search_population(self, population):
        '''
        Perform local search on current population

        '''
        for i, particle in enumerate(population):
            population[i] = self.np_local_search_genotype(particle)
        return population


    def np_generate_local_population(self, genotype):
        '''
        Generate off by 1 local population with size self.length_genotype

        '''
        return (np.diag(np.ones((self.length_genotypes))) + genotype) % 2


    def calculate_max_spanning_tree_genotype(self):
        '''
        Compute the genotype of the maximum spanning tree of the instance.

        '''
        max_spanning_tree = minimum_spanning_tree(self.fast_fit * -1).toarray().astype(np.int) < 0

        max_spanning_tree += max_spanning_tree.transpose()

        first_index = np.unravel_index(np.argmax(max_spanning_tree), max_spanning_tree.shape)

        return self.calculate_genotype_from_start_node(first_index[0], max_spanning_tree)


    def calculate_max_degree_weight_genotype(self):
        '''
        Compute the genotype of the maximum degree weight nodes

        '''
        
        self.degree_per_node = np.sum(self.fast_fit, axis=0)[np.newaxis,:]

        self.degree_per_node = np.append(self.degree_per_node, np.sum(self.fast_fit > 0, axis=0)[np.newaxis,:], axis=0)

        self.degree_per_node = np.append(self.degree_per_node, np.arange(self.length_genotypes)[np.newaxis,:], axis=0)

        self.degree_per_node = self.degree_per_node[:,np.argsort(self.degree_per_node[0,:])[::-1]]

        genotype = np.zeros((self.length_genotypes))
        genes_set = np.sum(self.fast_fit, axis=1) == 0
        for i in self.degree_per_node[2,:]:
            new_genotype = self.calculate_genotype_from_start_node(i, self.fast_fit, genotype.copy(), genes_set.copy())
            if self.np_fitness(genotype) < self.np_fitness(new_genotype):
                genotype = new_genotype
            genes_set[i] = True
            
        return genotype

    def calculate_genotype_from_start_node(self, node, adjacency_matrix, new_genotype = None, genes_set = None):
        '''
        Calculates the genotype from an adjacency matrix

        :param node: starting item
        :param adjacency_matrix: neighbouring matrix
        '''

        if new_genotype is None:
            new_genotype = np.zeros((self.length_genotypes))

        if genes_set is None:
            genes_set = np.sum(adjacency_matrix, axis=1) == 0

        pos = [node]
        neg = []
        while not np.all(genes_set):

            if len(pos) + len(neg) == 0:
                disconnected = np.nonzero(genes_set == 0)
                pos += [disconnected[0]]
                # pos = np.append(pos, disconnected[0])

            while len(pos) > 0:
                p = pos.pop(0)
                genes_set[p] = True
                new_genotype[p] = 1
                neg += [y for x in adjacency_matrix[p].nonzero() for y in x if y not in neg and not genes_set[y]]
                
            while len(neg) > 0:
                n = neg.pop(0)
                genes_set[n] = True
                pos += [y for x in adjacency_matrix[:, n].nonzero() for y in x if y not in pos and not genes_set[y]]

        return new_genotype