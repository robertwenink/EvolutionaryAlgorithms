import maxcut
import Ant

import json
import copy
import numpy as np
from random import randint
import maxcut_aco
import math
from operator import attrgetter

if __name__ == "__main__":
    np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})
    instances_directory = 'instances/'
    opt_directory = 'opts/'
    instance = "maxcut_4x4_1_1_donut.txt"
    # instance = "maxcut_2x2_1_1_donut.txt"

    instance = maxcut.MaxCut(instance, instances_directory, opt_directory)


    numAnts = 3
    max_its = 150

    # higher rho is more 'forgetting', so more exploration
    # I believe, if a solution is never good, we should be able to remove it almost completely (up to 0.1) within 1/4 the iterations.
    # so then is rho = 1 - 0.1**(1/(n/2))
    rho = 1 - 0.01**(1/(max_its/4))
    ph_max=10
    ph_min=1

    # likewise, d_ph 
    d_ph = rho*ph_max/2
    # d_ph = 100 # higher means smaller steps of pheromone, lower means more aggresive towards current best and alltime best
    alpha = 2
    beta = 1

    print("press key to start with: rho = %f, d_ph = %f" % (rho,d_ph))
    input()
    ACO = maxcut_aco.ACO(instance,numAnts,max_its,rho,ph_max,ph_min,d_ph)
    ACO.run()

class ACO:
    def __init__(self,instance,numAnts,max_its,rho,ph_max=1,ph_min=0,d_ph=100,alpha=1,beta=1):
        """"
        Initialise base BBO Ant Colony Optimization for Max-Cut
        :param instance: the max-cut instance, providing the nodes and edges including weights
        :param numAnts: number of Ants to run the problem with
        :param max_its: maximum number of iterations allowed
        :param rho: evaporation constant
        :param ph_max: maximum possible pheromone on an edge
        :param ph_min: minimum possible pheromone on an edge
        :param d_ph: constant scaling the increase of pheromone as 1/d_ph

        If we do Gray Box Optimalization and weights are known we additionally have:
        :param alpha: pheromone weight factor (how much do we value the pheromone)
        :param beta: local heuristic information factor (how much do we use the heuristic 'just take edges with large weight')
        """
        self.instance = instance
        self.edges_dict = self.instance.edges_dict\

        self.numAnts = numAnts
        self.max_its = max_its
        self.rho = rho
        self.ph_max = ph_max
        self.ph_min = ph_min
        self.d_ph = d_ph
        self.alpha = alpha
        self.beta = beta

        self.AntColony = self.initialiseAntPopulation(numAnts)

        self.previousElitist = copy.deepcopy(self.AntColony[0])
        self.elitist = copy.deepcopy(self.AntColony[0])
        self.alltimeElitist = copy.deepcopy(self.AntColony[0])
    
    def initialiseAntPopulation(self,numAnts):
        AntColony = []
        for i in range(numAnts):
            AntColony.append(Ant.Ant(self.instance.length_genotypes))
        return np.array(AntColony)

    def findElitist(self):
        # save previous elitist before finding the new one, to be used in case we want to try ohter options at updatePheromone
        self.previousElitist = copy.deepcopy(self.elitist)

        self.elitist = max(self.AntColony, key=attrgetter('fitness'))
        if self.elitist.fitness > self.alltimeElitist.fitness:
            self.alltimeElitist = copy.deepcopy(self.elitist)

    def getInitPheromoneDict(self):
        """
        Set the initial pheromone dictionary using the same representation structure as the edge dictionary with all values set to ph_max
        """
        pheromone_dict = copy.deepcopy(self.edges_dict)
        for k1 in pheromone_dict:
            for k2 in pheromone_dict[k1]:
                pheromone_dict[k1][k2] = self.ph_max
        return pheromone_dict
        

    def updatePheromone(self,ph,k1,k2):
        """
        Return updated pheromone of single pheromone count
        """
        newph = (1-self.rho)*ph
        # check if the edge combination is cut by checking if the corresponding genotypes have different 0/1 values
        if (self.elitist.cutVector[k1] == -self.elitist.cutVector[k2]):# or (self.alltimeElitist.cutVector[k1] == -self.alltimeElitist.cutVector[k2]):
            # newph += 1/(1/self.d_ph + self.alltimeElitist.fitness - self.elitist.fitness)
            newph += self.d_ph 


        # respect the pheromone bounds
        if newph < self.ph_min:
            return self.ph_min
        elif newph > self.ph_max:
            return self.ph_max
        else:
            return newph

    def updateGlobalPheromoneDict(self):
        """
        Update the pheromone if it belongs to the set of edges cut by the elitists
        """
        for k1 in self.ph_dict:
            for k2 in self.ph_dict[k1]:
                self.ph_dict[k1][k2] = self.updatePheromone(self.ph_dict[k1][k2],k1,k2)

    def evaluateAntFitness(self,ant):
        ant.transformCutvectorToGenotype()
        ant.fitness = int(self.instance.np_fitness(ant.genotype))

    def run(self):
        """
        Running the GBO algorithm 'AntCut'
        """
        # create similar one with pheromone weights, iniitialise at ph_max
        self.ph_dict = self.getInitPheromoneDict()

        # run it for all ants in the population, up to max_its iterations or 'walks'
        for it in range(self.max_its):
            for ant in self.AntColony:
                ant.run(self.edges_dict,self.ph_dict,self.alpha,self.beta)
                self.evaluateAntFitness(ant)
                # print(ant.genotype)
                # print(ant.cutVector)
                # print(ant.deltaF)
                # print(ant.fitness)
                # print("------------------------------------")

            self.findElitist()
            self.updateGlobalPheromoneDict()

            # printing the ph dict
            # print(json.dumps(self.ph_dict,sort_keys=True, indent=4))
            
            print(self.elitist.P)
            print(self.elitist.fitness)
            print(self.elitist.genotype)
            print("------------------------------------")
            # print(self.ph_dict)
            # print("it : %d, with best cut: %f" % (it,self.currentBestCut))
            # print(self.elitist.cutVector)     
        print("------------------------------------")
        print(self.alltimeElitist.P)   
        print(self.alltimeElitist.fitness)
        print(self.alltimeElitist.genotype)

    def runBBO(self):
        """
        With BBO where we assume to know nothing except the length of the genotype, we assign pheromones to the nodes instead of edges.
        Nodes with higher pheromone count should be more easily included in the 1-set. 
        Therefore, for 
        Is this the same as GBO but with differently constructed edges with only weights 1?
        """
        pass

