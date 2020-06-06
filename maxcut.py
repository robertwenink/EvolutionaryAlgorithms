import numpy as np

class MaxCut:

    def __init__(self, filename):
        '''
        initialize MaxCut problem from filename \n
        param create_reverse_edges : if True duplicates edges for reverse

        '''

        self.edges_dict = {}
        self.edges_tuples = {}
        self.edges_list = []

        self.length_genotypes = 0

        with open(filename, "r") as f:
            lines = f.readlines()

            self.length_genotypes = len(lines)
            # Faster numpy array for speedy fitness checking
            self.fast_fit = np.zeros((self.length_genotypes))

            for i, line in enumerate(lines):
                edge = line.split()
                node_1, node_2, weight = edge[0], edge[1], edge[2]

                if node_1 not in self.edges_dict:
                    self.edges_dict[node_1] = {}
                self.edges_dict[node_1][node_2] = weight

                self.edges_tuples[tuple(sorted([node_1, node_2]))] = weight

                self.edges_list.append(tuple(node_1, node_2, weight))

                self.fast_fit[i] = weight


    def np_fitness(self, genotype):
        '''
        Method for calculating fitness given genotype as input \n
        Input: list or np.array of bits \n
        Output: total weight of edges

        '''

        return np.sum(genotype * self.fast_fit)


    def np_generate_random_genotype(self):
        '''
        Method for generating a random genotype of length self.length_genotypes \n
        Ouput: random numpy array of bits

        '''
        return np.random.rand(2, size=self.length_genotypes)