import numpy as np

class ParticleSwarmOptimization:
    
    w = 0.729
    c1 = 1.49445
    c2 = 1.49445
    lr = 0.01

    def __init__(self, instance, numOfBoids, numOfEpochs):
        self.instance = instance

        self.swarm_list = [self.Particle(instance, self) for i in range(numOfBoids)]
        self.numOfEpochs = numOfEpochs

        self.best_swarm_position = np.random.uniform(low=-500, high=500, size=genes)
        self.best_swarm_error = 1e80  # Set high value to best swarm error

    class Particle:

        def __init__(self, instance, swarm, genes):
            self.instance = instance

            self.position = np.random.randint(2, size=instance.length_genotypes)
            self.velocity = np.random.uniform(2, size=instance.length_genotypes)
            self.best_part_pos = self.position.copy()

            self.error = error(self.position)
            self.best_part_err = self.error.copy()

        def setPos(self, pos):
            self.position = pos
            self.error = error(pos)
            if self.error < self.best_part_err:
                self.best_part_err = self.error
                self.best_part_pos = pos

    
    def optimize(self):
        for i in range(self.numOfEpochs):

            for j in range(len(self.swarm_list)):

                current_particle = self.swarm_list[j]  # get current particle

                Vcurr = grad_error(current_particle.position)  # calculate current velocity of the particle

                deltaV = self.w * Vcurr \
                        + self.c1 * (current_particle.best_part_pos - current_particle.position) \
                        + self.c2 * (self.best_swarm_position - current_particle.position)  # calculate delta V

                new_position = self.swarm_list[j].position - self.lr * deltaV  # calculate the new position

                self.swarm_list[j].setPos(new_position)  # update the position of particle

                if error(new_position) < self.best_swarm_error:  # check the position if it's best for swarm
                    self.best_swarm_position = new_position
                    self.best_swarm_error = error(new_position)

            print('Epoch: {0} | Best position: [{1},{2}] | Best known error: {3}'.format(i,
                                                                                        self.best_swarm_position[0],
                                                                                        self.best_swarm_position[1],
                                                                                        self.best_swarm_error))
