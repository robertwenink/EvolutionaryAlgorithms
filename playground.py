import itertools
import random

def group_crossover(parents_list):
    all_parents = []
    for parent in parents_list:
        solution_zeros = set([i for i, e in enumerate(parent) if e != 1])
        solution_ones = set([i for i, e in enumerate(parent) if e != 0])
        all_parents.append([solution_zeros, solution_ones])
    permutations = ["".join(seq) for seq in itertools.product("01", repeat=len(parents_list))]
    int_perms = []
    for i in permutations:
        int_perms.append([int(char) for char in i])
    max_set = {}
    for i in int_perms:
        intersection = set(range(len(parents_list[0])))
        for index, j in enumerate(i):
            intersection = intersection & all_parents[index][j]
        if len(intersection) > len(max_set):
            max_set = intersection
    for i in max_set:
        for parent in all_parents:
            if i in parent[0]:
                parent[0].remove(i)
            if i in parent[1]:
                parent[1].remove(i)
    max_set_2 = {}
    for i in int_perms:
        intersection = set(range(len(parents_list[0])))
        for index, j in enumerate(i):
            intersection = intersection & all_parents[index][j]
        if len(intersection) > len(max_set_2):
            max_set_2 = intersection
    for i in range(len(parents_list[0])):
        if i not in max_set and i not in max_set_2:
            if random.uniform(0, 1)> 0.5:
                max_set.add(i)
            else:
                max_set_2.add(i)
    new_chromosome = [0]*len(parents_list[0])
    for i in max_set:
        new_chromosome[i] = 1

    return new_chromosome
a = [0, 0, 0, 1, 1, 1, 1]
b = [0, 1, 0, 0, 1, 1, 1]
c = [0, 0, 0, 0, 0, 1, 1]
all = []
all.append(a)
all.append(b)
all.append(c)
group_crossover(all)