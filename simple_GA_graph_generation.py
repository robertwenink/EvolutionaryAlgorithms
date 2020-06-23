import graphgenerator
from graphgenerator import generateRandomGraph



directory = 'C:/Users/wybek/Documents/school/Master/evolutionary/final_project/simple_GA_instances_weights'
#nodes = [4, 8, 16, 32, 64]
maxWeight = [1, 10, 100, 1000]
#edgeProb = [0.2, 0.4, 0.6, 0.8, 1]

for i in range(4):
    generateRandomGraph(directory, 16, maxWeight[i], 0.2, problemId=str(i))

#generateRandomGraph(directory, 128, maxWeight, edgeProb, problemId=5)