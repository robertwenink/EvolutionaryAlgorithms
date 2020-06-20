import graphgenerator
from graphgenerator import generateRandomGraph



directory = 'C:/Users/wybek/Documents/school/Master/evolutionary/final_project/simple_GA_instances/'
nodes = [4, 8, 16, 32, 64]
maxWeight = 20
edgeProb = 0.2

for i in range(len(nodes)):
    generateRandomGraph(directory, nodes[i], maxWeight, edgeProb, problemId=str(i))