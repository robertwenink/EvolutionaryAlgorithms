import numpy as np

class Ant:
    def __init__(self,length_genotypes):
        self.genotype = np.zeros(length_genotypes)
        self.fitness = int(0)
        
        # initialize cutvector; initially all in same set, for migrate to other set *-1, X in {-1,1}
        self.cutVector = np.ones(length_genotypes)

        # inits for procedure of getting the changes to switch set membership. 
        self.deltaF = None
        self.P = np.ones(length_genotypes) 

    def initializeDeltaF(self,edges_dict,ph_dict,alpha,beta):
        """
        Initialize the vector indicating expected increase of cut value. Tracked seperately by each individual ant.
        """
        deltaF = np.zeros(np.shape(self.cutVector))
        for k1 in ph_dict:
            for k2 in ph_dict[k1]:
                deltaF[k1] += self.cutVector[k1]*self.cutVector[k2] * ph_dict[k1][k2]**alpha * edges_dict[k1][k2]**beta
        self.deltaF = deltaF

    def updateDeltaF(self,k,edges_dict,ph_dict,alpha,beta):
        """
        Evidently, when changing the set membership of a single node (making change in cutVector), we only have to change the deltaF corresponding to that node and the nodes it is connected to
        """
        # The profit we gained is exactly opposite to what we will lose if we change again, so:
        self.deltaF[k] =-self.deltaF[k]
        
        # Likewise, if we had a positive gain, it is now negative, so add that number twice to get update value of all vj, and vice versa
        for k2 in ph_dict[k]:
            value = 2 * self.cutVector[k]*self.cutVector[k2] * ph_dict[k2][k]**alpha * edges_dict[k2][k]**beta
            self.deltaF[k] += value

        # update the canditates for updating P later (after checking there are more than 0 candidates)
        self.Candidates = (self.deltaF > 0)

    def updateP(self):
        """
        Update value of p by weighting the benefit of changing  (+ or -) over the total benefit gain possible. 
        This implies that we only take into account positive influences. We can keep the algorithm per ant going as long as we have a positive p left.
        Call this function only if there is a candidate left.
        """
        self.P[self.Candidates] = np.divide(self.deltaF[self.Candidates],np.sum(self.deltaF[self.Candidates]))

    def updateCutvector(self,k):
        self.cutVector[k] = -self.cutVector[k]

    def transformCutvectorToGenotype(self):
        """
        Cutvector is expressing in {-1,1} while genotype expressed as {0,1}
        """
        self.genotype = ((self.cutVector + 1)/2).astype('int32') 

    def run(self,edges_dict,ph_dict,alpha,beta):
        """
        Run the procedure for an ant for a single iteration
        """
        self.cutVector = np.ones(np.shape(self.cutVector))
        self.Candidates = np.ones(np.shape(self.cutVector),dtype=bool)
        self.initializeDeltaF(edges_dict,ph_dict,alpha,beta)
        self.updateP()

        # while we can still make a positive change and thus the candidate set is not empty ...
        while(np.any(self.Candidates>0)):
            # randomly choose an index/node in the candidate set and decide using p if we are going to switch it
            a = np.where(self.Candidates==True)[0]
            k = np.random.choice(a)
            # print(self.P)
            if self.P[k] >= np.random.uniform(0,1,1):
                self.updateCutvector(k)
                self.updateDeltaF(k,edges_dict,ph_dict,alpha,beta)
                self.updateP()