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
    # instances_directory = 'instances/'
    instances_directory = 'from_assignment_3/'
    opt_directory = 'opts/'
    # instancename = "maxcut_4x4_1_1_donut"
    # instancename = "maxcut_2x2_1_1_donut"
    instancename = 'newL25_2'

    instance = maxcut.MaxCut(instancename+".txt", instances_directory, opt_directory)

    numAnts = 20
    max_its = 150

    # higher rho is more 'forgetting', so more exploration
    # I believe, if a solution is never good, we should be able to remove it almost completely (up to 0.1 ph_max) within 1/4 the iterations.
    # so then is rho = 1 - 0.1**(1/(n/2))
    rho = 1 - 0.01**(1/(max_its/4))
    rho *= 2
    ph_max=2
    ph_min=1

    # likewise, d_ph should scale with the amount we might be able to subtract in order to arrive at a stable solution
    # higher means smaller steps of pheromone, lower means more aggresive towards current best and alltime best
    # globally seen, if we always choose a solution, we require rho*ph_max for stable behaviour towards ph_max; 
    # as we add pheromone per ant, we divide by numants
    d_ph = rho*ph_max/(numAnts) 

    # alpha: scaling the pheromone contribution, beta: scaling the 'take high weights' heuristic
    alpha = 2
    beta = 1

    # Are we doing BBO (true) or GBO (false)?
    BBO = False

    print("press key to start with: rho = %f, d_ph = %f" % (rho,d_ph))
    input()
    ACO = maxcut_aco.ACO(instance,numAnts,max_its,rho,ph_max,ph_min,d_ph,BBO)
    ACO.run()

class ACO:
    def __init__(self,instance,numAnts,max_its,rho,ph_max=1,ph_min=0,d_ph=100,BBO=True,alpha=1,beta=1):
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
        self.edges_dict = self.instance.edges_dict

        # check edges dict before run
        print(json.dumps(self.edges_dict,sort_keys=True, indent=4))
        input()
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

    def run(self):
        """
        Running the GBO algorithm 'AntCut'
        """
        # create similar one with pheromone weights, iniitialise at ph_max
        self.ph_dict = self.getInitPheromoneDict()

        self.initialiseAntGenotypesFiftyFifty()

        # run it for all ants in the population, up to max_its iterations or 'walks'
        for it in range(self.max_its):
            for ant in self.AntColony:
                ant.run(self.edges_dict,self.ph_dict,self.alpha,self.beta)
                self.evaluateAntFitness(ant)

            
            self.findElitist()
            self.updatePheromone()

            # printing the ph dict
            # print(json.dumps(self.ph_dict,sort_keys=True, indent=4))
            print(self.elitist.deltaF)
            print(self.elitist.P)
            print(self.elitist.fitness)
            print(self.elitist.genotype)
            print("------------------------------------")
        print("------------------------------------")
        print(self.alltimeElitist.P)   
        print(self.alltimeElitist.fitness)
        print(self.alltimeElitist.genotype)

    #####################################################################
    ########### PHEROMONE OPERATIONS ####################################
    #####################################################################
    
    def evaporatePheromone(self):
        for k1 in self.ph_dict:
            for k2 in self.ph_dict[k1]:
                self.ph_dict[k1][k2] = (1-self.rho)*(self.ph_dict[k1][k2])

    def updatePheromone(self):
        """
        If the fitness of an ant is above average, drop pheromone on the edges that he cut
        """
        self.evaporatePheromone()

        avg_fitness = self.getAveragePopulationFitness()
        alltimebest = self.alltimeElitist.fitness

        for ant in self.AntColony:
            for k1 in self.ph_dict:
                for k2 in self.ph_dict[k1]:
                    # second part better retains the alltimebest in the pheromone trail!
                    if (ant.cutVector[k1] != ant.cutVector[k2]) or (self.alltimeElitist.cutVector[k1] != self.alltimeElitist.cutVector[k2] and ant.fitness > avg_fitness):
                        dph = self.d_ph*(1+(ant.fitness - avg_fitness)/avg_fitness) # avg dph = self.d_ph !
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
        # save previous elitist before finding the new one, to be used in case we want to try ohter options at updatePheromone
        self.previousElitist = copy.deepcopy(self.elitist)

        self.elitist = max(self.AntColony, key=attrgetter('fitness'))
        if self.elitist.fitness > self.alltimeElitist.fitness:
            self.alltimeElitist = copy.deepcopy(self.elitist)

    #####################################################################
    ########### INITIALIZATION OPERATIONS ###############################
    #####################################################################

    def getInitPheromoneDict(self):
        """
        Set the initial pheromone dictionary using the same representation structure as the edge dictionary with all values set to ph_max
        """
        pheromone_dict = copy.deepcopy(self.edges_dict)
        for k1 in pheromone_dict:
            for k2 in pheromone_dict[k1]:
                pheromone_dict[k1][k2] = self.ph_max
        return pheromone_dict

    def initialiseAntPopulation(self,numAnts):
        AntColony = []
        for i in range(numAnts):
            AntColony.append(Ant.Ant(self.instance.length_genotypes))
        return np.array(AntColony)

    def initialiseAntGenotypesFiftyFifty(self):
        N = self.instance.length_genotypes
        for ant in self.AntColony:
            ant.genotype = rand_bin_array(math.ceil(N),N)
            
def rand_bin_array(K, N):
    """
    K the number of ones
    N the genotype length
    """
    arr = np.zeros(N)
    arr[:K]  = 1
    np.random.shuffle(arr)
    return arr
