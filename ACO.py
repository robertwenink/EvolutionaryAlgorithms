import maxcut as mc
import Ant

import json
import copy
import numpy as np
from random import randint
import ACO
import math
from operator import attrgetter
import matplotlib.pyplot as plt

if __name__ == "__main__":
    np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})
    instances_directory = 'instances/'
    instances_directory = "testinstances/"
    # instances_directory = 'from_assignment_3/'
    opt_directory = 'opts/'
    # # instancename = "maxcut_4x4_1_1_donut"
    # # instancename = "maxcut_2x2_1_1_donut"
    # instancename = 'newL25_2' #opt = 530
    # # instancename = 'newL12_1' # opt = 127
    # # instancename = 'newL12_2' # opt = 124
    # instancename = "newL50_1_opt2078"
    # # instancename = "newL50_2_opt2056"
    instancenamelist = ["maxcut_random_16_0.4_100_instance_0","maxcut_random_32_0.4_100_instance_1","maxcut_random_64_0.4_100_instance_2","maxcut_random_128_0.4_100_instance_3","maxcut_random_256_0.4_100_instance_4","maxcut_random_512_0.4_100_instance_5"]
    instancename = instancenamelist[2]

    instance = mc.MaxCut(instancename+".txt", instances_directory, opt_directory)
    numAnts = 20
    max_evals = 1000
    max_gens = 100
    # rho = 1 - 0.01**(1/(max_evals/4))

    # 
    rho = 0.21
    ph_max=10
    ph_min=0.1
    alpha = 0.7

    fig, ax = plt.subplots()
    fig2, ax2 = plt.subplots()

    ACO_BBO = ACO.ACO_BBO(instance,numAnts,max_evals,max_gens,rho,ph_max,ph_min,alpha,ax)
    ACO_BBO.run()

    ACO_BBO2 = ACO.ACO_BBO2(instance,numAnts,max_evals,max_gens,rho,ph_max,ph_min,alpha,ax)
    ACO_BBO2.run()

    # GBO
    rho = 0.41
    alpha = 1.5
    beta = 1

    # ACO_GBO = ACO.ACO_GBO(instance,numAnts,max_evals,max_gens,rho,ph_max,ph_min,alpha,beta,ax)
    # ACO_GBO.run()

    # ACO_GBO2 = ACO.ACO_GBO2(instance,numAnts,max_evals,max_gens,rho,ph_max,ph_min,alpha,beta,ax)
    # ACO_GBO2.run()

    ax2.plot(ACO_BBO.numEvalsList,ACO_BBO.archiveElitistList,color="black",label="Black Box archive elitist fitness")
    ax2.plot(ACO_BBO.numEvalsList,ACO_BBO.averageFitnessList,'-*',color="black",label="Black Box average population fitness")

    ax2.plot(ACO_BBO2.numEvalsList,ACO_BBO2.archiveElitistList,color="blue",label="Black Box archive elitist fitness + local search")
    ax2.plot(ACO_BBO2.numEvalsList,ACO_BBO2.averageFitnessList,'-*',color="blue",label="Black Box average population fitness + local search")

    # ax2.plot(ACO_GBO.numEvalsList,ACO_GBO.archiveElitistList,color="green",label="Grey Box archive elitist fitness")
    # ax2.plot(ACO_GBO.numEvalsList,ACO_GBO.averageFitnessList,'-*',color="green",label="Grey Box average population fitness")

    # ax2.plot(ACO_GBO2.numEvalsList,ACO_GBO2.archiveElitistList,color="silver",label="Grey Box archive elitist fitness + local search")
    # ax2.plot(ACO_GBO2.numEvalsList,ACO_GBO2.averageFitnessList,'-*',color="silver",label="Grey Box average population fitness + local search")
    ax2.legend()

    print(instancename+".txt :")
    print("Black Box elitist:")
    ACO_BBO.printArchiveElitist()
    print("Black Box 2 elitist:")
    ACO_BBO2.printArchiveElitist()
    # print("Grey Box elitist:")
    # ACO_GBO.printArchiveElitist()
    # print("Grey Box 2 elitist:")
    # ACO_GBO2.printArchiveElitist()

    ax.legend(loc='lower left')
    plt.show()


class ACO:
    def __init__(self,instance,numAnts,max_evals,max_gens,rho,ph_max=1,ph_min=0,alpha=1,ax=None):
        """"
        Initialise base Ant Colony Optimization class
        :param instance: the max-cut instance, providing the nodes and edges including weights
        :param numAnts: number of Ants to run the problem with
        :param max_evals: maximum number of iterations allowed
        :param max_gens: maximum number of generations allowed, not mutually exclusive with max_evals
        :param rho: evaporation constant
        :param ph_max: maximum possible pheromone on an edge
        :param ph_min: minimum possible pheromone on an edge
        :param alpha: pheromone weight factor (how much do we value the pheromone)
        :param ax: provide matplotlib axis for optional plotting of elitist trend
        """
        self.instance = instance 
        self.edges_dict = self.getEdgedict(instance)
        self.Lg = self.instance.length_genotypes

        # check edges dict before run
        # print(json.dumps(self.edges_dict,sort_keys=True, indent=4))
        # input()

        self.numAnts = numAnts
        self.max_evals = max_evals
        self.max_gens = max_gens
        self.rho = rho
        self.ph_max = ph_max
        self.ph_min = ph_min
        self.alpha = alpha

        self.ax = ax
        self.hasnolabel = True

        self.initAgain()

        # number of edges in original graph
        self.num_edges = len(instance.edges_dict)/2 # undirected graph

    def initAgain(self):
        """
        Function to initialise the 'usables' of the ACO class, such that running the same instance from a clear class only requires calling this function beforehand.
        """
        self.AntColony = self.initialiseAntPopulation(self.numAnts)
        self.elitist = copy.deepcopy(self.AntColony[0])
        self.archiveElitist = copy.deepcopy(self.AntColony[0])

        self.numEvals = 0

        self.archiveElitistList = []
        self.averageFitnessList = []
        self.numEvalsList = []

    def run(self):
        raise NotImplementedError

    def initialiseAntPopulation(self,numAnts):
        raise NotImplementedError

    def getEdgedict(self,instance):
        raise NotImplementedError

    def updatePheromone(self):
        raise NotImplementedError

    def respectPheromoneBounds(self,newph):  
        """
        Bound the pheromone to be within desired bounds.
        """      
        # respect the pheromone bounds
        if newph < self.ph_min:
            return self.ph_min
        elif newph > self.ph_max:
            return self.ph_max
        else:
            return newph

    def evaporatePheromone(self):
        """
        Evaporates pheromone globally.
        """
        for k1 in self.ph_dict:
            for k2 in self.ph_dict[k1]:
                # and evaporate
                self.ph_dict[k1][k2] = (1-self.rho)*(self.ph_dict[k1][k2])

    def evaluateAntFitness(self,ant):
        """
        Evaluating the fitness of the genotype using the centrally defined fitness function of the problem instance.
        """
        ant.fitness = int(self.instance.np_fitness(ant.genotype))
        self.numEvals += 1

    def getAveragePopulationFitness(self):
        """
        Calculate current population fitness average.
        """
        avg = 0
        for ant in self.AntColony:
            avg += ant.fitness
        return avg/np.shape(self.AntColony)[0]

    def getInitPheromoneDict(self,edges_dict):
        """
        Set the initial pheromone dictionary using the same representation structure as the edge dictionary
        """
        pheromone_dict = copy.deepcopy(edges_dict)
        dtau_dict = copy.deepcopy(edges_dict)
        for k1 in pheromone_dict:
            for k2 in pheromone_dict[k1]:
                pheromone_dict[k1][k2] = (self.ph_max-self.ph_min)/2 + self.ph_min
                dtau_dict[k1][k2] = 0
        return pheromone_dict,dtau_dict

    def findElitist(self):
        """
        Standard implementation of finding the elitist in current population. Updates the alltime archive elitist if found.
        """
        self.elitist = max(self.AntColony, key=attrgetter('fitness'))
        if self.elitist.fitness > self.archiveElitist.fitness:
            self.archiveElitist = copy.deepcopy(self.elitist)

    def doLocalSearch(self,ant):
        # local search part, on initialisation does 1 full evaluation, 
        # afterwards each update is 1/#edges evaluation although this *lazily* is not implemented as such
        L = self.instance.np_calculate_all_bits_flip_value(ant.genotype)
        self.numEvals += 1
        while np.any(L<0):
            # get maximum bit to switch
            k = np.argmin(L)

            # switch bit within both the genotype as trail for proper pheromone update6
            ant.genotype[k] = 1 - ant.genotype[k]
            k2 = 0
            if k < self.Lg:
                ant.trail[k] = k + self.Lg
                k2 = k
            else:
                ant.trail[k] = k - self.Lg
                k2 = k - self.Lg

            L = self.instance.np_calculate_all_bits_flip_value(ant.genotype)
            self.numEvals += 1/self.num_edges*len(self.instance.edges_dict[k2])

    def np_calculate_all_bits_flip_value(self, genotype):
        '''
        Calculates bit flip value of entire genotype, basically a local search
        :param genotype: genotype
        :return: array of values of bitflip
        '''
        return np.sum(self.instance.fast_fit * (genotype != genotype[:, np.newaxis]) - self.instance.fast_fit * (genotype == genotype[:, np.newaxis]), axis=1)

    def logPlottingData(self):
        """
        Log the new data after evaluation and after finding the elitist
        """
        self.archiveElitistList.append(self.archiveElitist.fitness)
        self.numEvalsList.append(self.numEvals)
        self.averageFitnessList.append(self.getAveragePopulationFitness())

    def printElitist(self):
        print(self.elitist.fitness)
        print(self.elitist.genotype)
        print("------------------------------------")

    def printArchiveElitist(self):
        print("------------------------------------") 
        print(self.archiveElitist.fitness)
        print(self.archiveElitist.genotype)
        print("------------------------------------\n") 
    
    def printDict(self,dicttoprint):
        print(json.dumps(dicttoprint,sort_keys=True, indent=4))

    def plotAgainstNumevals(self,data,text,col):
        if self.hasnolabel:
            t = text 
        else:
            t = None

        if self.ax is not None:
            self.ax.scatter(self.numEvals,data, label= t , color = col)
            self.hasnolabel = False

            self.ax.set_xlabel("Number of evaluations")
            self.ax.set_ylabel("Fitness")

##################################################################
###################### Black Box specific ########################
##################################################################
class ACO_BBO(ACO):
    def __init__(self,instance,numAnts,max_evals,max_gens,rho,ph_max=1,ph_min=0,alpha=1,ax=None):
        ACO.__init__(self,instance,numAnts,max_evals,max_gens,rho,ph_max,ph_min,alpha,ax)

    def run(self):
        """
        Running the BBO
        """
        print("########## Starting BBO ############")
        # create similar one with pheromone weights, iniitialise at ph_max
        self.ph_dict,_ = self.getInitPheromoneDict(self.edges_dict)
        self.adict = self.createAndUpdateRelativePheromoneDict(self.ph_dict,self.alpha)
        self.pdict = self.createAndUpdatePdict(self.adict)

        # run it for all ants in the population, up to max_evals iterations or 'walks'
        genr = 0
        while self.numEvals <= self.max_evals and genr < self.max_gens:
            for ant in self.AntColony:
                ant.run(self.pdict)
                
                ant.transformTrailToGenotype()
                self.evaluateAntFitness(ant)
            
            self.findElitist()
            self.updatePheromone() #updates ph_dict without explicit return

            self.adict = self.createAndUpdateRelativePheromoneDict(self.ph_dict,self.alpha)
            self.pdict = self.createAndUpdatePdict(self.adict)

            # for data plotting
            self.logPlottingData()
            # self.printElitist()
            self.plotAgainstNumevals(self.elitist.fitness,"BBO elitist fitness","black")
            genr += 1

        self.printArchiveElitist()


    def getEdgedict(self,instance):
        """
        Knowing only the amount of nodes, create an artificial graph in which all edge weights are 1; where a node is split up into a path including and path excluding it, creating a directed graph.
        The include nodes are stored at node#, the 'do not include nodes' at node# + length_genotype.
        """
        new_dict = dict() 
        edge_dict = self.instance.edges_dict
        N = len(edge_dict)
        for k in edge_dict: # I only make use of the genotype node names!
            new_dict[k] = {}
            new_dict[k+N] = {}
        
        first = min(new_dict, key=int)
        last = max(new_dict, key=int)
        
        # add a singular starting node leading to the two options of the first node
        new_dict[-1] = {}
        new_dict[-1][first] = int(1)
        new_dict[-1][first+N] = int(1)

        for k in edge_dict:
            if(k == last):
                # last nodes are endstates so do nothing
                pass
            else: 
                # non-include node
                new_dict[k][k+1] = int(1) # edge to non-include
                new_dict[k][k+1+N] = int(1) # edge to include

                # include node
                new_dict[k+N][k+1] = int(1) # edge to non-include
                new_dict[k+N][k+1+N] = int(1) # edge to include
        return new_dict

    def createAndUpdateRelativePheromoneDict(self,ph_dict,alpha):
        """
        Initialize the dict containing the portion of pheromone an edge has compared to all nodes outgoing edges`.
        """
        adict = copy.deepcopy(ph_dict)
        for k1 in adict:
            s = np.sum(np.power(list(ph_dict[k1].values()),alpha))
            for k2 in adict[k1]:
                adict[k1][k2] = ph_dict[k1][k2]**alpha / s
        return adict

    def createAndUpdatePdict(self,adict):
        """
        Update transition probabilities dictionary
        """
        pdict = copy.deepcopy(adict)
        for k1 in adict:
            s = sum(adict[k1].values())
            for k2 in adict[k1]:
                pdict[k1][k2] = pdict[k1][k2] / s
        return pdict
    
    def updatePheromone(self):
        """
        If the fitness of an ant is above average, drop pheromone on the edges that he traversed
        """
        avg_fitness = self.getAveragePopulationFitness()
        
        self.evaporatePheromone()

        for ant in self.AntColony:
            if ant.fitness > avg_fitness:
                dt = ant.fitness / avg_fitness

                # drop on first edge
                k1 = -1
                k2 = ant.trail[0]
                self.ph_dict[k1][k2] = self.respectPheromoneBounds(self.ph_dict[k1][k2]+dt)
                for i in range(self.Lg-1):
                    k1 = ant.trail[i]
                    k2 = ant.trail[i+1]
                    self.ph_dict[k1][k2] = self.respectPheromoneBounds(self.ph_dict[k1][k2]+dt)

    def initialiseAntPopulation(self,numAnts):
        AntColony = []
        for i in range(numAnts):
            AntColony.append(Ant.Ant_BBO(self.Lg))
        return np.array(AntColony)


##################################################################
#################### Black/Grey Box 2 specific ###################
##################################################################

class ACO_BBO2(ACO_BBO):
    """
    This class of 'BBO' includes local search, thus actually grey box but with the model of the bbo
    """
    def __init__(self,instance,numAnts,max_evals,max_gens,rho,ph_max=1,ph_min=0,alpha=1,ax=None):
        ACO_BBO.__init__(self,instance,numAnts,max_evals,max_gens,rho,ph_max,ph_min,alpha,ax)

    def run(self):
        """
        Running the BBO with local search for maxcut
        """
        print("########## Starting BBO with local search ############")
        # create similar one with pheromone weights, iniitialise at ph_max
        self.ph_dict,_ = self.getInitPheromoneDict(self.edges_dict)
        self.adict = self.createAndUpdateRelativePheromoneDict(self.ph_dict,self.alpha)
        self.pdict = self.createAndUpdatePdict(self.adict)

        # run it for all ants in the population, up to max_evals iterations or 'walks'
        genr = 0
        while self.numEvals <= self.max_evals and genr < self.max_gens:
            for ant in self.AntColony:
                ant.run(self.pdict)
                
                ant.transformTrailToGenotype()

                self.doLocalSearch(ant)

                self.evaluateAntFitness(ant)
            
            self.findElitist()
            self.updatePheromone() #updates ph_dict without explicit return

            self.adict = self.createAndUpdateRelativePheromoneDict(self.ph_dict,self.alpha)
            self.pdict = self.createAndUpdatePdict(self.adict)

            # for data plotting
            self.logPlottingData()
            # self.printElitist()
            self.plotAgainstNumevals(self.elitist.fitness,"BBO elitist fitness","black")
            genr += 1

        self.printArchiveElitist()



##################################################################
###################### Grey Box specific #########################
##################################################################

class ACO_GBO(ACO):
    def __init__(self,instance,numAnts,max_evals,max_gens,rho,ph_max=1,ph_min=0,alpha=1,beta=1,ax=None):
        """
        Initialise base Ant Colony Optimization for Max-Cut
        Doing Gray Box Optimalization with weights and problem structure known we additionally have:
        :param beta : local heuristic information factor for weights (how much do we use the heuristic 'just take edges with large weight')
        """
        ACO.__init__(self,instance,numAnts,max_evals,max_gens,rho,ph_max,ph_min,alpha,ax)
        self.beta = beta

    def run(self):
        """
        Running the GBO algorithm 'AntCut'
        """
        print("########## Starting GBO ############")
        # create similar one with pheromone weights, iniitialise at ph_max
        self.ph_dict,self.dtau_dict = self.getInitPheromoneDict(self.edges_dict)

        # run it for all ants in the population, up to max_evals iterations or 'walks'
        genr = 0
        while self.numEvals <= self.max_evals and genr < self.max_gens:
            for ant in self.AntColony:
                ant.run(self.edges_dict,self.ph_dict,self.alpha,self.beta)
                ant.transformCutvectorToGenotype()

                self.evaluateAntFitness(ant)
            
            self.findElitist()
            self.updatePheromone()

            # for data plotting
            self.logPlottingData()
            # self.printElitist()
            self.plotAgainstNumevals(self.elitist.fitness,"GBO elitist fitness","silver")
            genr += 1

        self.printArchiveElitist()

    def getEdgedict(self,instance):
        """
        Return the full edge dictionary of the maxcut problem instance
        """
        return instance.edges_dict

    def getdtau(self):
        """
        Calculate the dictionary containing desired amount of pheromone to add per edge cut by an ant.
        """
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
            if ant.fitness > avg_fitness:
                for k1 in self.ph_dict:
                    for k2 in self.ph_dict[k1]:
                        if (ant.cutVector[k1] != ant.cutVector[k2]):
                            # dph = self.dtau_dict[k1][k2]*(1+(ant.fitness - avg_fitness)/avg_fitness)
                            dph = ant.fitness/avg_fitness
                            self.ph_dict[k1][k2] = self.respectPheromoneBounds(self.ph_dict[k1][k2]+dph)

    def findElitist(self):
        """
        Overriding the standard implementation as for the used update rule the number of cut edges of an elitist is used and I like to update it once here
        """
        self.elitist = max(self.AntColony, key=attrgetter('fitness'))
        # self.C_elitist = self.findNumCutEdges(self.elitist)
        if self.elitist.fitness > self.archiveElitist.fitness:
            self.archiveElitist = copy.deepcopy(self.elitist)
            self.C_elitist = self.findNumCutEdges(self.archiveElitist)
            
    def findNumCutEdges(self,ant):
        """
        Find the number of edges that are cut by a given individual ant.
        This number is used in defining the amount of pheromone to drop.
        """
        C = 0
        for k1 in self.ph_dict:
            for k2 in self.ph_dict[k1]:
                if (ant.cutVector[k1] != ant.cutVector[k2]):
                    C+=1
        return C

    def initialiseAntPopulation(self,numAnts):
        """
        Initialise population with the GBO ant variant
        """
        AntColony = []
        for i in range(numAnts):
            AntColony.append(Ant.Ant_GBO(self.Lg))
        return np.array(AntColony)




class ACO_GBO2(ACO_GBO):
    def __init__(self,instance,numAnts,max_evals,max_gens,rho,ph_max=1,ph_min=0,alpha=1,beta=1,ax=None):
        """
        Initialise base Ant Colony Optimization for Max-Cut GBO including local search
        """
        ACO_GBO.__init__(self,instance,numAnts,max_evals,max_gens,rho,ph_max,ph_min,alpha,beta,ax)

    def run(self):
        """
        Running the GBO algorithm 'AntCut'
        """
        print("########## Starting GBO ############")
        # create similar one with pheromone weights, iniitialise at ph_max
        self.ph_dict,self.dtau_dict = self.getInitPheromoneDict(self.edges_dict)

        # run it for all ants in the population, up to max_evals iterations or 'walks'
        genr = 0
        while self.numEvals <= self.max_evals and genr < self.max_gens:
            for ant in self.AntColony:
                ant.run(self.edges_dict,self.ph_dict,self.alpha,self.beta)
                ant.transformCutvectorToGenotype()

                L = self.instance.np_calculate_all_bits_flip_value(ant.genotype)
                self.numEvals += 1
                while np.any(L<0):
                    # get maximum bit to switch
                    k = np.argmin(L)

                    ant.genotype[k] = 1 - ant.genotype[k]

                    L = self.instance.np_calculate_all_bits_flip_value(ant.genotype)
                    self.numEvals += 1/self.num_edges

                self.evaluateAntFitness(ant)
            
            self.findElitist()
            self.updatePheromone()

            # for data plotting
            self.logPlottingData()
            # self.printElitist()
            self.plotAgainstNumevals(self.elitist.fitness,"GBO elitist fitness","silver")
            genr += 1

        self.printArchiveElitist()