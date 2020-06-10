from fastga import FastMutationOperator
import copy
from random import randint
import random as rand

def fitness(bit_string):
    cut = 0

    for edge in E:
        u = int(edge[0])
        v = int(edge[1])
        w = int(edge[2])
        if bit_string[u-1] != bit_string[v-1]:
            cut += w

    return cut

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



