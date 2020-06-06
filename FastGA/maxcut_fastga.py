from fastga import FastMutationOperator
import copy
from random import randint

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

# Loading single instance
with open('maxcut_4x4_1_1_donut','r') as f:
    size = int(next(f))
    E = [line.split() for line in f]

# INITIALIZATION: TODO: Horrible, horrible code
pop_size = 20
population = []
max_generations = 30
form = '{0:0' + str(size) + 'b}'
for i in range(pop_size):
    bit_string = form.format(randint(0, 2**10))
    population.append(list(map(int, list(bit_string))))


operator = FastMutationOperator(n=size, beta=1.5)

# OPTIMIZATION
for i in range(max_generations):
    print("## gen " + str(i) + " ##")
    for i in range(pop_size):
        offspring = copy.deepcopy(population[i])
        operator.mutate(offspring, inplace=True)
        if fitness(offspring) > fitness(population[i]):
            population[i] = offspring

    print("Max fitness " + str(max_fitness(population)))


