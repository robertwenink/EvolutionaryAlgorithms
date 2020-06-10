import numpy as np

class BinaryParticleSwarmOptimization:

    def __init__(self, instance, num_particles=50, numOfEpochs=500):
        '''
        Initialize the Binary Particle Swarm Optimization class

        '''

        # fitness function 
        self.calculate_fitness = instance.np_fitness

        # generator
        self.generate_random_genotype = instance.np_generate_random_genotype

        # set no epochs
        self.numOfEpochs = numOfEpochs

        # init best global position so far
        self.best_position_swarm = self.generate_random_genotype()

        # Set high value to best swarm error
        self.best_fitness_swarm = self.calculate_fitness(self.best_position_swarm)

        # init swarm
        self.swarm_list = [Particle(self) for i in range(num_particles)]

        self.np_swarm_position = np.random.randint(2, size=(num_particles, self.length_genotypes))
        self.np_swarm_velocity = np.random.uniform(low=0, high=1, size=len(num_particles, self.position))


    def update_best_swarm(self, particle):
        '''
        Update the global elite fitness after particle best has been updated

        '''
        if particle.best_fitness_particle > self.best_fitness_swarm:
            self.best_position_swarm = particle.best_position_particle
            self.best_fitness_swarm = particle.best_fitness_particle

    
    def calculate_velocity(self, particle):
        '''
        Calculate the velocity of the Particle

        '''
        phi_1 = np.random.rand()
        phi_2 = np.random.rand()
        return particle.velocity + phi_1 * (particle.best_position_particle - particle.position) + \
                    phi_2 * (self.best_position_swarm - particle.position)


    def sigmoid(self, arr):
        '''
        Calculate sigmoid function

        '''
        return 1/(1 + np.exp(-arr)) 

    
    def run(self):
        '''
        Run the optimization algorithm

        '''
        for epoch in range(self.numOfEpochs):

            for i, particle in enumerate(self.swarm_list):

                # calculate new velocity of the particle
                new_velocity = self.calculate_velocity(particle)

                # calculate new position
                rand = np.random.uniform(low=0, high=1, size=len(particle.position))
                new_position = (rand < self.sigmoid(new_velocity)).astype(np.int64)
                
                # update velocity and position
                particle.update(new_position, new_velocity)

            if epoch % 50 == 0:
                print(f'Epoch: {epoch} | Best position: {self.best_position_swarm} | Best known fitness: {self.best_fitness_swarm}')

        print(f'\nFinal Epoch: {epoch} | Best position: {self.best_position_swarm} | Best known fitness: {self.best_fitness_swarm}')

class Particle:
    '''
    Inner Particle class belonging to Binary Particle Swarm

    '''

    def __init__(self, swarm):
        '''
        Particle initializer

        '''
        # pointer to swarm
        self.swarm = swarm

        # random position
        self.position = self.swarm.generate_random_genotype()

        # random velocity
        self.velocity = np.random.uniform(low=0, high=1, size=len(self.position))

        # calc fitness
        self.fitness = self.swarm.calculate_fitness(self.position)

        # set best position
        self.best_position_particle = self.position.copy()

        # set best fitness
        self.best_fitness_particle = self.fitness.copy()

        # update swarm
        self.swarm.update_best_swarm(self)


    def update(self, position, velocity):
        '''
        Set position of particle and update best position particle

        '''
        self.position = position
        self.velocity = velocity
        self.fitness = self.swarm.calculate_fitness(self.position)

        if self.fitness > self.best_fitness_particle:
            self.best_position_particle = position
            self.best_fitness_particle = self.fitness
            self.swarm.update_best_swarm(self)