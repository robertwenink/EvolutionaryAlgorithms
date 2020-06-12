import numpy as np

class BinaryParticleSwarmOptimization:

    def __init__(self, instance, num_particles=50, num_of_epochs=500):
        '''
        Initialize the Binary Particle Swarm Optimization class

        '''

        # fitness function 
        self.calculate_fitness = instance.np_fitness

        # matrix fitness function
        self.calculate_fitness_population = instance.np_fitness_population

        # generator
        self.generate_random_genotype = instance.np_generate_random_genotype

        # set no epochs
        self.num_of_epochs = num_of_epochs

        # set no particles
        self.num_particles = num_particles

        # set length genotypes
        self.length_genotypes = instance.length_genotypes

        # init best global position so far
        self.best_position_swarm = self.generate_random_genotype()

        # Set high value to best swarm error
        self.best_fitness_swarm = self.calculate_fitness(self.best_position_swarm)

        # init swarm
        self.swarm_list = [Particle(self) for i in range(num_particles)]


    def np_run(self):
        '''
        SUPER-MEGA-FAST-PSO-RUNNER

        '''
        # All current positions
        self.np_swarm_position = np.random.randint(2, size=(self.num_particles, self.length_genotypes))

        # Particles best historic position
        self.np_swarm_best_position = self.np_swarm_position.copy()

        # All particles current velocity
        self.np_swarm_velocity = np.random.rand(self.num_particles, self.length_genotypes)

        # All particles current fitness
        self.np_swarm_fitness = self.calculate_fitness_population(self.np_swarm_position)

        # Particles best historic fitness
        self.np_swarm_best_fitness = self.np_swarm_fitness.copy()

        for epoch in range(self.num_of_epochs):

            self.np_swarm_velocity = self.np_swarm_velocity + \
                np.einsum('i, ik -> ik', np.random.rand(self.num_particles), self.np_swarm_best_position - self.np_swarm_position) + \
                    np.einsum('i, ik -> ik', np.random.rand(self.num_particles), self.np_swarm_best_position[np.argmax(self.np_swarm_best_fitness)] - self.np_swarm_position)

            self.np_swarm_position = (np.random.rand(self.num_particles, self.length_genotypes) < self.sigmoid(self.np_swarm_velocity)).astype(np.int64)
            
            self.np_swarm_fitness = self.calculate_fitness_population(self.np_swarm_position)

            self.np_swarm_best_position = np.where(np.repeat( \
                (self.np_swarm_fitness > self.np_swarm_best_fitness)[:, np.newaxis], self.length_genotypes, axis=1), \
                    self.np_swarm_position, self.np_swarm_best_position)

            self.np_swarm_best_fitness = np.where(self.np_swarm_fitness > self.np_swarm_best_fitness, \
                self.np_swarm_fitness, self.np_swarm_best_fitness)

            if epoch % 250 == 0:
                print(f'Epoch: {epoch} | Best position: {self.np_swarm_best_position[np.argmax(self.np_swarm_best_fitness)]} | Best known fitness: {self.np_swarm_best_fitness[np.argmax(self.np_swarm_best_fitness)]}')

        print(f'\nFinal Epoch: {epoch+1} | Best position: {self.np_swarm_best_position[np.argmax(self.np_swarm_best_fitness)]} | Best known fitness: {self.np_swarm_best_fitness[np.argmax(self.np_swarm_best_fitness)]}')


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
        for epoch in range(self.num_of_epochs):

            for i, particle in enumerate(self.swarm_list):

                # calculate new velocity of the particle
                new_velocity = self.calculate_velocity(particle)

                # calculate new position
                rand = np.random.uniform(low=0, high=1, size=len(particle.position))
                new_position = (rand < self.sigmoid(new_velocity)).astype(np.int64)
                
                # update velocity and position
                particle.update(new_position, new_velocity)

            if epoch % 250 == 0:
                print(f'Epoch: {epoch} | Best position: {self.best_position_swarm} | Best known fitness: {self.best_fitness_swarm}')

        print(f'\nFinal Epoch: {epoch+1} | Best position: {self.best_position_swarm} | Best known fitness: {self.best_fitness_swarm}')

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
        self.velocity = np.random.rand(len(self.position))

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