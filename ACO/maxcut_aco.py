import copy
from random import randint
import random as rand

class Ant:
    def __init__(self):


class ACO:
    def __init__(self,instance,numAnts,max_its,rho,ph_max=1,ph_min=0,Q=100,alpha=1,beta=1):
        """"
        Initialise base BBO Ant Colony Optimization for Max-Cut
        :param instance: the max-cut instance, providing the nodes and edges including weights
        :param numAnts: number of Ants to run the problem with
        :param max_its: maximum number of iterations allowed
        :param rho: evaporation constant
        :param ph_max: maximum possible pheromone on an edge
        :param ph_min: minimum possible pheromone on an edge
        :param Q: constant scaling the increase of pheromone as 1/Q

        If we do Gray Box Optimalization and weights are known we additionally have:
        :param alpha: pheromone weight factor (how much do we value the pheromone)
        :param beta: local heuristic information factor (how much do we use the heuristic 'just take edges with large weight')
        """
        self.instance = instance
        self.numAnts = numAnts
        self.max_its = max_its
        self.rho = rho
        self.ph_max = ph_max
        self.ph_min = ph_min
        self.Q = Q

        self.previousBestCut = 0
        self.currentBestCut = 0
        
    def updateIndividualPheromone(self,ph):
        newph = (1-self.rho)*ph + 1/(self.Q+currentBestCut) # - previousBestCut?
        if newph < self.ph_min:
            return self.ph_min
        elif newph > self.ph_max:
            return self.ph_max
        else:
            return newph

def max_fitness(population):
    max_fitness = 0
    # TODO: change this loop bc its really bad
    for individual in population:
        if fitness(individual) > max_fitness:
            max_fitness = fitness(individual)
    return max_fitness

def avg_fitness(population):
    avg_fitness = 0
    for individual in population:
        avg_fitness += fitness(individual)
    return avg_fitness / len(population)

def get_cut_edges(individual):
    edge_list = []

    for edge in E:
        u = int(edge[0])
        v = int(edge[1])
        if bit_string[u-1] != bit_string[v-1]:
            edge_list.append((u, v))
            edge_list.append((v, u))

    return edge_list

# Loading single instance
with open('../instances/maxcut_10_05_20_instance_01.txt', 'r') as f:
    size = int(next(f))
    E = [line.split() for line in f]


# MAKE NODE DICT
node_dict = {}
for edge in E:
    u = int(edge[0])
    v = int(edge[1])
    # For u
    if u in node_dict:
        node_dict[u].append(v)
    else:
        node_dict[u] = [v]

    # For v
    if v in node_dict:
        node_dict[v].append(u)
    else:
        node_dict[v] = [u]

# MAKE EDGE DICT
# TODO: initialization
edge_dict = {}
for edge in E:
    u = int(edge[0])
    v = int(edge[1])
    w = int(edge[2])
    edge_dict[(u,v)] = w
    edge_dict[(v,u)] = w

# MAKE PHEROMONE DICT - T_i_j
pheromone_dict = {}
for edge in E:
    u = int(edge[0])
    v = int(edge[1])
    pheromone_dict[(u,v)] = 1.0
    pheromone_dict[(v,u)] = 1.0


# INITIALIZATION: TODO: Horrible, horrible code
pop_size = 20
population = []
max_generations = 30
form = '{0:0' + str(size) + 'b}'
for i in range(pop_size):
    bit_string = form.format(randint(0, 2**10))
    population.append(list(map(int, list(bit_string))))

# Pheromone booster
boost = 1.05
evap = 0.9

# OPTIMIZATION
for i in range(max_generations):
    print("## gen " + str(i) + " ##")
    # For every ant
    for j in range(pop_size):
        # For every bit
        bit_string = population[j]
        for k in range(len(bit_string)):
            node = bit_string[k]
            conn_nodes = node_dict[node]
            total_w = 0
            conn_w = 0
            # get edge probabilities
            for conn_node in conn_nodes:
                # Multiply this edge with pheromone factor
                edge_w = edge_dict[(node, conn_node)] * pheromone_dict[(node, conn_node)]
                total_w += edge_w
                if bit_string[node-1] == bit_string[conn_node-1]:
                    conn_w += edge_w

            probability = conn_w/total_w
            # Decide where to go
            if rand.uniform(0, 1) <= probability:
                bit_string[k] = int(bit_string[k]) ^ 1

    # Evaporate pheromones
    for key, value in pheromone_dict.items():
        pheromone_dict[key] = value * evap
    #
    # For every ant
    avg_fit = avg_fitness(population)
    for j in range(pop_size):
        # Reinforce path
        bit_string = population[j]
        if fitness(bit_string) > avg_fit:
            edge_list = get_cut_edges(bit_string)
            for edge in edge_list:
                pheromone_dict[edge] = pheromone_dict[edge] * boost
    print("Max fitness " + str(max_fitness(population)))



