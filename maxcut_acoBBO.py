def createBBOedgedict(self,instance):
        """
        Knowing only the amount of nodes, create an artificial graph in which all edge weights are 1; and edges link back to itself.
        This makes that the GBO algorithm behave like a BBO as it effectively can only assign added pheromone to the nodes.
        """
        edge_dict = copy.deepcopy(self.instance.edges_dict)
        new_dict = dict() 
        for k in edge_dict:
            new_dict[k] = {}
            # new_dict[k][k] = 1
        
        maxkey = max(new_dict, key=int)
        minkey = min(new_dict, key=int)
        previouskey = minkey
        for k in new_dict:
            if(k == minkey):
                new_dict[k][maxkey] = int(1)
                new_dict[maxkey][k] = int(1)
            else: 
                new_dict[k][k-1] = int(1)
                new_dict[k-1][k] = int(1)
                # new_dict[int(k)][previous] = 1 # if for some reason the node numbering is not continous
                # new_dict[previous][k] = 1
            previouskey = k
        return new_dict



### dit is rotzooi
    def updatePheromone(self,ph,k1,k2):
        """
        Return updated pheromone of single pheromone count.
        For highly connected graphs this works very poorly, as then just all egdes` weights get updated without distinction.
        """
        # check if the edge combination is cut by checking if the corresponding genotypes have different 0/1 values
        if (self.alltimeElitist.cutVector[k1] == -self.alltimeElitist.cutVector[k2]):# or (self.elitist.cutVector[k1] == -self.elitist.cutVector[k2])
            # newph += 1/(1/self.d_ph + self.alltimeElitist.fitness - self.elitist.fitness)
            # newph += self.d_ph 
            newph += self.d_ph * self.elitist.fitness/(self.alltimeElitist.fitness)

        return self.respectPheromoneBounds(newph)
        
    def updateGlobalPheromoneDict(self):
        """
        Update the pheromone if it belongs to the set of edges cut by the elitists
        """
        for k1 in self.ph_dict:
            for k2 in self.ph_dict[k1]:
                self.ph_dict[k1][k2] = self.evaporatePheromone(self.ph_dict[k1][k2])
                # self.ph_dict[k1][k2] = self.updatePheromone(self.ph_dict[k1][k2],k1,k2)

        self.updatePheromone()