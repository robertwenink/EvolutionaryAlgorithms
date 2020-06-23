import numpy as np

# Ideas:
# Restriction - all ideas must be writable in Einstein notation for speed
# 1: Add local search
#       Create list of single bit mutations of length genotype_length
#       Assign best value to current.
# 2: Add mutations against convergence
#       Add random mutations each bit in a spring has a probability p 
#       of changing dependend on the fitness distance to the best solution 
#       the best solution changes is bits with p = 0, the worst 
#       solution with p = 0.5.
# 3: Add mutations based on path distance
#       Path distance is calculated as the 


class BPSO:

    def __init__(self, instance, metrics, run, num_particles, num_of_epochs):
        '''
        Initialize the Binary Particle Swarm Optimization class

        '''

        # instance
        self.instance = instance

        # metrics object
        self.metrics = metrics

        # run
        self.run = run

        # set no epochs
        self.num_of_epochs = num_of_epochs

        # set no particles
        self.num_particles = num_particles

        # set length genotypes
        self.length_genotypes = instance.length_genotypes

        # evaluations counter
        self.counter = 0

    def sigmoid(self, arr):
        '''
        Calculate sigmoid function

        '''
        return 1/(1 + np.exp(-arr)) 

class VectorizedBinaryParticleSwarmOptimization(BPSO):


    def __init__(self, instance, metrics, num_particles, run, num_of_epochs):
        '''
        Initialize the vectorized version of Particle swarm optimization
        
        '''

        super().__init__(instance, metrics, run, num_particles, num_of_epochs)
        # init the get fitness function of the metrics class
        metrics.best_fitness_function = self.get_swarm_best_fitness

    def get_swarm_best_fitness(self):
        '''
        Return the best swarm position fitness

        '''

        self.np_swarm_best_position[np.argmax(self.np_swarm_best_fitness)]


    def np_run(self, GBO_init=False, GBO=False, local_search=False, norm_velo=False):
        '''
        SUPER-MEGA-FAST-PSO-RUNNER

        '''

        # All current positions
        self.np_swarm_position = np.random.randint(2, size=(self.num_particles, self.length_genotypes))

        if GBO_init:
            fitness = self.instance.np_fitness_population(self.np_swarm_position, self.counter)
            minargi = np.argsort(fitness)

            # Add Gray Box genotype based on max spanning tree
            self.np_swarm_position[minargi[0]] = self.instance.calculate_max_spanning_tree_genotype()

            # Add Gray Box genotype based on greedy edge weight search
            self.np_swarm_position[minargi[1]] = self.instance.calculate_max_degree_weight_genotype()
            
            # Perform local search
            self.np_swarm_position = self.instance.np_local_search_population(self.np_swarm_position, self.counter)

        # Particles best historic position
        self.np_swarm_best_position = self.np_swarm_position.copy()

        # All particles current velocity
        self.np_swarm_velocity = np.random.rand(self.num_particles, self.length_genotypes)

        # All particles current fitness
        self.np_swarm_fitness = self.instance.np_fitness_population(self.np_swarm_position, self.counter)

        # Particles best historic fitness
        self.np_swarm_best_fitness = self.np_swarm_fitness.copy()

        for epoch in range(self.num_of_epochs):

            # Re-calculate swarm velocity
            self.np_swarm_velocity = self.np_swarm_velocity + \
                np.einsum('i, ik -> ik', np.random.rand(self.num_particles), self.np_swarm_best_position - self.np_swarm_position) + \
                    np.einsum('i, ik -> ik', np.random.rand(self.num_particles), self.np_swarm_best_position[np.argmax(self.np_swarm_best_fitness)] - self.np_swarm_position)

            # If gray-box optimization, use bit flip value for velocity.
            if GBO: 
                bit_flip_vals = self.instance.np_calculate_all_bits_flip_value_population(self.np_swarm_position)
                bit_flip_vals_norm = (bit_flip_vals / np.max(np.abs(bit_flip_vals),axis=1)[:, np.newaxis] + 1) / 2
                # self.np_swarm_velocity += np.random.rand(self.num_particles)[:, np.newaxis] * \
                #     (bit_flip_vals_norm * (self.np_swarm_position == 0) - \
                #         bit_flip_vals_norm * (self.np_swarm_position == 1))
                self.np_swarm_velocity += np.ones(self.num_particles)[:, np.newaxis] * \
                    (bit_flip_vals_norm * (self.np_swarm_position == 0) - \
                        bit_flip_vals_norm * (self.np_swarm_position == 1))

            # if local search, perform local search to increase fitness for all position with hamming distance 1
            if local_search:
                self.np_swarm_position = self.instance.np_local_search_population(self.np_swarm_position, self.counter)

            # if number use velocity normalization every norm_velo epochs.
            if isinstance(norm_velo, int) and norm_velo is not False:
                if (epoch + 1) % norm_velo == 0:
                    self.np_swarm_velocity /= np.max(np.abs(self.np_swarm_velocity), axis=1)[:, np.newaxis]

            # Use sigmori
            self.np_swarm_position = (np.random.rand(self.num_particles, self.length_genotypes) < self.sigmoid(self.np_swarm_velocity)).astype(np.int64)
            
            self.np_swarm_fitness = self.instance.np_fitness_population(self.np_swarm_position, self.counter)

            update = self.np_swarm_fitness > self.np_swarm_best_fitness
            self.np_swarm_best_position[update] = self.np_swarm_position[update]

            self.np_swarm_best_fitness[update] = self.np_swarm_fitness[update]

            if epoch % 250 == 0:
                print(f'Epoch: {epoch} | Best position: {self.np_swarm_best_position[np.argmax(self.np_swarm_best_fitness)]} | Best known fitness: {self.np_swarm_best_fitness[np.argmax(self.np_swarm_best_fitness)]}')

        print(f'\nFinal Epoch: {epoch + 1} | Best position: {self.np_swarm_best_position[np.argmax(self.np_swarm_best_fitness)]} | Best known fitness: {self.np_swarm_best_fitness[np.argmax(self.np_swarm_best_fitness)]}')



# class BinaryParticleSwarmOptimization(BPSO):

#     def __init__(self, instance, metrics, run, num_particles=50, num_of_epochs=500):
#         '''
#         Initialize the Binary Particle Swarm Optimization class

#         '''

#         super().__init__(instance, metrics, run, num_particles, num_of_epochs)

#         # fitness function 
#         self.calculate_fitness = instance.np_fitness

#         # generator
#         self.generate_random_genotype = instance.np_generate_random_genotype

#         # init best global position so far
#         self.best_position_swarm = self.generate_random_genotype()

#         # Set high value to best swarm error
#         self.best_fitness_swarm = self.calculate_fitness(self.best_position_swarm)

#         # init swarm
#         self.swarm_list = [Particle(self) for i in range(num_particles)]





#     def update_best_swarm(self, particle):
#         '''
#         Update the global elite fitness after particle best has been updated

#         '''
#         if particle.best_fitness_particle > self.best_fitness_swarm:
#             self.best_position_swarm = particle.best_position_particle
#             self.best_fitness_swarm = particle.best_fitness_particle

    
#     def calculate_velocity(self, particle):
#         '''
#         Calculate the velocity of the Particle

#         '''
#         phi_1 = np.random.rand()
#         phi_2 = np.random.rand()
#         return particle.velocity + phi_1 * (particle.best_position_particle - particle.position) + \
#                     phi_2 * (self.best_position_swarm - particle.position)


    
#     def run(self):
#         '''
#         Run the optimization algorithm

#         '''
#         for epoch in range(self.num_of_epochs):

#             for i, particle in enumerate(self.swarm_list):

#                 # calculate new velocity of the particle
#                 new_velocity = self.calculate_velocity(particle)

#                 # calculate new position
#                 rand = np.random.uniform(low=0, high=1, size=len(particle.position))
#                 new_position = (rand < self.sigmoid(new_velocity)).astype(np.int64)
                
#                 # update velocity and position
#                 particle.update(new_position, new_velocity)

#             if epoch % 250 == 0:
#                 print(f'Epoch: {epoch} | Best position: {self.best_position_swarm} | Best known fitness: {self.best_fitness_swarm}')

#         print(f'\nFinal Epoch: {epoch+1} | Best position: {self.best_position_swarm} | Best known fitness: {self.best_fitness_swarm}')


# class Particle:
#     '''
#     Inner Particle class belonging to Binary Particle Swarm

#     '''

#     def __init__(self, swarm):
#         '''
#         Particle initializer

#         '''
#         # pointer to swarm
#         self.swarm = swarm

#         # random position
#         self.position = self.swarm.generate_random_genotype()

#         # random velocity
#         self.velocity = np.random.rand(len(self.position))

#         # calc fitness
#         self.fitness = self.swarm.calculate_fitness(self.position)

#         # set best position
#         self.best_position_particle = self.position.copy()

#         # set best fitness
#         self.best_fitness_particle = self.fitness.copy()

#         # update swarm
#         self.swarm.update_best_swarm(self)


#     def update(self, position, velocity):
#         '''
#         Set position of particle and update best position particle

#         '''
#         self.position = position
#         self.velocity = velocity
#         self.fitness = self.swarm.calculate_fitness(self.position)

#         if self.fitness > self.best_fitness_particle:
#             self.best_position_particle = position
#             self.best_fitness_particle = self.fitness
#             self.swarm.update_best_swarm(self)