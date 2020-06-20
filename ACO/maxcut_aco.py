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
    # instances_directory = 'from_assignment_3/'
    opt_directory = 'opts/'
    instancename = "maxcut_4x4_1_1_donut"
    # instancename = "maxcut_2x2_1_1_donut"
    # instancename = 'newL25_2' #opt = 530

    instance = maxcut.MaxCut(instancename+".txt", instances_directory, opt_directory)

    numAnts = 4
    max_its = 50

    # I believe, if a solution is never good, we should be able to remove it almost completely (up to 0.1 ph_max) within 1/4 the iterations.
    # so then is rho = 1 - 0.1**(1/(n/2))
    rho = 1 - 0.01**(1/(max_its/4))
    rho *= 2
    ph_max=2
    ph_min=1
    alpha = 1
    beta = 1

    print("press key to start with: rho = %f" % (rho))
    # input()
    ACO = maxcut_aco.ACO(instance,numAnts,max_its,rho,ph_max,ph_min)
    ACO.run()

class ACO:
    def __init__(self,instance,numAnts,max_its,rho,ph_max=1,ph_min=0,alpha=1,beta=1):
        """"
        Initialise base Ant Colony Optimization for Max-Cut
        :param instance: the max-cut instance, providing the nodes and edges including weights
        :param numAnts: number of Ants to run the problem with
        :param max_its: maximum number of iterations allowed
        :param rho: evaporation constant
        :param ph_max: maximum possible pheromone on an edge
        :param ph_min: minimum possible pheromone on an edge

        If we do Gray Box Optimalization and weights are known we additionally have:
        :param alpha: pheromone weight factor (how much do we value the pheromone)
        :param beta: local heuristic information factor (how much do we use the heuristic 'just take edges with large weight')
        """
        self.instance = instance 
        self.edges_dict = self.getEdgedict(instance)

        # check edges dict before run
        print(json.dumps(self.edges_dict,sort_keys=True, indent=4))
        # input()
        self.numAnts = numAnts
        self.max_its = max_its
        self.rho = rho
        self.ph_max = ph_max
        self.ph_min = ph_min

        self.alpha = alpha
        self.beta = beta

        self.AntColony = self.initialiseAntPopulation(numAnts)

        self.num_edges = len(self.edges_dict)/2 # undirected graph
        self.elitist = copy.deepcopy(self.AntColony[0])
        self.archiveElitist = copy.deepcopy(self.AntColony[0])

    def run(self):
        """
        Running the GBO algorithm 'AntCut'
        """
        # create similar one with pheromone weights, iniitialise at ph_max
        self.ph_dict,self.dtau_dict = self.getInitPheromoneDict()

        # run it for all ants in the population, up to max_its iterations or 'walks'
        for it in range(self.max_its):
            for ant in self.AntColony:
                ant.run(self.edges_dict,self.ph_dict,self.alpha,self.beta)

            ## local search
            for ant in self.AntColony:
                ant.run(self.edges_dict,self.ph_dict,0,1,False)
                
                ant.transformCutvectorToGenotype()
                self.evaluateAntFitness(ant)
            
            self.findElitist()
            self.updatePheromone()

            # printing the ph dict
            # print(json.dumps(self.ph_dict,sort_keys=True, indent=4))
            # print(self.elitist.deltaF)
            # print(self.elitist.P)
            print(self.elitist.fitness)
            print(self.elitist.genotype)
            print("------------------------------------")

        print("------------------------------------")
        # print(self.archiveElitist.P)   
        print(self.archiveElitist.fitness)
        print(self.archiveElitist.genotype)

    def getEdgedict(self,instance):
        return instance.edges_dict

    #####################################################################
    ########### PHEROMONE OPERATIONS ####################################
    #####################################################################
    
    def evaporatePheromone(self):
        for k1 in self.ph_dict:
            for k2 in self.ph_dict[k1]:
                # and evaporate
                self.ph_dict[k1][k2] = (1-self.rho)*(self.ph_dict[k1][k2])

    def getdtau(self):
        for k1 in self.ph_dict:
            for k2 in self.ph_dict[k1]:
                # establish new dtau
                self.dtau_dict[k1][k2] = self.rho*self.ph_dict[k1][k2]/self.numAnts * self.num_edges/self.C_elitist

    def updatePheromone(self):
        """
        If the fitness of an ant is above average, drop pheromone on the edges that he cut
        """
        avg_fitness = self.getAveragePopulationFitness()
        
        self.getdtau()
        self.evaporatePheromone()

        for ant in self.AntColony:
            for k1 in self.ph_dict:
                for k2 in self.ph_dict[k1]:
                    if (ant.cutVector[k1] != ant.cutVector[k2]):
                        dph = self.dtau_dict[k1][k2]*(1+(ant.fitness - avg_fitness)/avg_fitness)
                        self.ph_dict[k1][k2] = self.respectPheromoneBounds(self.ph_dict[k1][k2]+dph)


    def respectPheromoneBounds(self,newph):        
        # respect the pheromone bounds
        if newph < self.ph_min:
            return self.ph_min
        elif newph > self.ph_max:
            return self.ph_max
        else:
            return newph



    #####################################################################
    ########### FITNESS OPERATIONS ######################################
    #####################################################################

    def evaluateAntFitness(self,ant):
        ant.transformCutvectorToGenotype()
        ant.fitness = int(self.instance.np_fitness(ant.genotype))

    def getAveragePopulationFitness(self):
        avg = 0
        for ant in self.AntColony:
            avg += ant.fitness
        return avg/np.shape(self.AntColony)[0]

    def findElitist(self):
        self.elitist = max(self.AntColony, key=attrgetter('fitness'))
        # self.C_elitist = self.findNumCutEdges(self.elitist)
        if self.elitist.fitness > self.archiveElitist.fitness:
            self.archiveElitist = copy.deepcopy(self.elitist)
            self.C_elitist = self.findNumCutEdges(self.archiveElitist)
            

    def findNumCutEdges(self,ant):
        C = 0
        for k1 in self.ph_dict:
            for k2 in self.ph_dict[k1]:
                if (ant.cutVector[k1] != ant.cutVector[k2]):
                    C+=1
        return C

    #####################################################################
    ########### INITIALIZATION OPERATIONS ###############################
    #####################################################################

    def getInitPheromoneDict(self):
        """
        Set the initial pheromone dictionary using the same representation structure as the edge dictionary with all values set to ph_max
        """
        pheromone_dict = copy.deepcopy(self.edges_dict)
        dtau_dict = copy.deepcopy(self.edges_dict)
        for k1 in pheromone_dict:
            for k2 in pheromone_dict[k1]:
                pheromone_dict[k1][k2] = (self.ph_max-self.ph_min)
                dtau_dict[k1][k2] = 0
        return pheromone_dict,dtau_dict

    def initialiseAntPopulation(self,numAnts):
        AntColony = []
        for i in range(numAnts):
            AntColony.append(Ant.Ant_GBO(self.instance.length_genotypes))
        return np.array(AntColony)

    # def initialiseAntGenotypesFiftyFifty(self):
    #     N = self.instance.length_genotypes
    #     for ant in self.AntColony:
    #         ant.genotype = ant.rand_bin_array(math.ceil(N/2),N)
            

