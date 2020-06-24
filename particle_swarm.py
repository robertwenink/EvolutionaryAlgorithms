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

    def __init__(self, instance, metrics, run, num_particles):
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
        # self.num_of_epochs = num_of_epochs

        # set no particles
        self.num_particles = num_particles

        # set length genotypes
        self.length_genotypes = instance.length_genotypes

    def sigmoid(self, arr):
        '''
        Calculate sigmoid function

        '''
        return 1/(1 + np.exp(-arr)) 

class VectorizedBinaryParticleSwarmOptimization(BPSO):


    def __init__(self, instance, metrics, num_particles, run):
        '''
        Initialize the vectorized version of Particle swarm optimization
        
        '''

        super().__init__(instance, metrics, run, num_particles)
        # init the get fitness function of the metrics class
        metrics.best_fitness_function = self.get_swarm_best_fitness

    def get_swarm_best_fitness(self):
        '''
        Return the best swarm position fitness

        '''

        return self.np_swarm_best_fitness[np.argmax(self.np_swarm_best_fitness)]


    def np_run(self, GBO=False, local_search=False, norm_velo=False):
        '''
        SUPER-MEGA-FAST-PSO-RUNNER

        '''

        # All current positions
        self.np_swarm_position = np.random.randint(2, size=(self.num_particles, self.length_genotypes))

        # if GBO_init:
        #     fitness = self.instance.np_fitness_population(self.np_swarm_position, self.metrics, self.run)
        #     minargi = np.argsort(fitness)

        #     # Add Gray Box genotype based on max spanning tree
        #     self.np_swarm_position[minargi[0]] = self.instance.calculate_max_spanning_tree_genotype()

        #     # Add Gray Box genotype based on greedy edge weight search
        #     self.np_swarm_position[minargi[1]] = self.instance.calculate_max_degree_weight_genotype()
            
        #     # Perform local search
        #     self.np_swarm_position = self.instance.np_local_search_population(self.np_swarm_position, self.metrics, self.run)

        # Particles best historic position
        self.np_swarm_best_position = self.np_swarm_position.copy()

        # All particles current velocity
        self.np_swarm_velocity = np.random.rand(self.num_particles, self.length_genotypes)

        # All particles current fitness
        self.np_swarm_fitness = self.instance.np_fitness_population(self.np_swarm_position, self.metrics, self.run)

        # Particles best historic fitness
        self.np_swarm_best_fitness = self.np_swarm_fitness.copy()

        # for epoch in range(self.num_of_epochs):
        epoch = 0
        done = False
        while not done:

            # Re-calculate swarm velocity
            self.np_swarm_velocity = self.np_swarm_velocity + \
                np.einsum('i, ik -> ik', np.random.rand(self.num_particles), self.np_swarm_best_position - self.np_swarm_position) + \
                    np.einsum('i, ik -> ik', np.random.rand(self.num_particles), self.np_swarm_best_position[np.argmax(self.np_swarm_best_fitness)] - self.np_swarm_position)

            # If gray-box optimization, use bit flip value for velocity.
            if GBO: 
                bit_flip_vals = self.instance.np_calculate_all_bits_flip_value_population(self.np_swarm_position)
                bit_flip_vals_norm = (bit_flip_vals / np.max(np.abs(bit_flip_vals),axis=1)[:, np.newaxis] + 1) / 2
                self.np_swarm_velocity += np.random.rand(self.num_particles)[:, np.newaxis] * \
                    (bit_flip_vals_norm * (self.np_swarm_position == 0) - \
                        bit_flip_vals_norm * (self.np_swarm_position == 1))

            # Use sigmoid for new position
            self.np_swarm_position = (np.random.rand(self.num_particles, self.length_genotypes) < self.sigmoid(self.np_swarm_velocity)).astype(np.int)
            
            # update fitness of swarm
            self.np_swarm_fitness = self.instance.np_fitness_population(self.np_swarm_position, self.metrics, self.run)

            done = self.update_best_position()

            # if local search, perform local search to increase fitness for all position with hamming distance 1
            if local_search and not done:
                for i in len(self.num_particles):
                    self.np_swarm_position[i], self.np_swarm_fitness[i] = self.local_search_genotype(self.np_swarm_position[i])
                    done = self.update_best_position()

            # if number use velocity normalization every norm_velo epochs.
            if isinstance(norm_velo, int) and norm_velo is not False:
                if (epoch + 1) % norm_velo == 0:
                    self.np_swarm_velocity /= np.max(np.abs(self.np_swarm_velocity), axis=1)[:, np.newaxis]

            if (epoch) % 100 == 0:
                print(f'Epoch: {epoch} | Best position: {self.np_swarm_best_position[np.argmax(self.np_swarm_best_fitness)]} | Best known fitness: {self.np_swarm_best_fitness[np.argmax(self.np_swarm_best_fitness)]} | No evals: {self.metrics.evaluations}')
            epoch += 1

        print(f'\nFinal Epoch: {epoch + 1} | Best position: {self.np_swarm_best_position[np.argmax(self.np_swarm_best_fitness)]} | Best known fitness: {self.np_swarm_best_fitness[np.argmax(self.np_swarm_best_fitness)]} | No evals: {self.metrics.evaluations}\n')

    def update_best_position(self):
        '''
        Update best position and fitness swarm

        '''

        # update swarm best position and fitness
        update = self.np_swarm_fitness > self.np_swarm_best_fitness

        self.np_swarm_best_position[update] = self.np_swarm_position[update]

        self.np_swarm_best_fitness[update] = self.np_swarm_fitness[update]

        # update metrics
        return self.metrics.update_metrics(self.run, None, True)

    def local_search_genotype(self, genotype):
        
        x_best = genotype.copy()
        x_best_fit = self.instance.np_fitness(x_best, self.metrics, self.run)
        x = genotype.copy()
        flag = True
        while(flag):
            flag = False
            F = np.ones((self.length_genotypes), dtype=np.bool)
            gains = self.instance.np_calculate_all_bits_flip_value(x, self.metrics, self.run)
            for k in range(np.int(self.length_genotypes/10)):

                g_a = np.argmax(gains[F])
                x[g_a] = x[g_a] == 0
                F[g_a] = False
                gains = self.instance.np_calculate_all_bits_flip_value(x, self.metrics, self.run)
                x_fit = self.instance.np_fitness(x, self.metrics, self.run)
                if x_fit > x_best_fit:
                    x_best = x.copy()
                    x_best_fit = x_fit
                    flag = True
                
                g_b = np.argmax(gains[np.logical_and(F, x == x[g_a])])
                x[g_b] = x[g_b] == 0
                F[g_b] = False
                gains = self.instance.np_calculate_all_bits_flip_value(x, self.metrics, self.run)
                x_fit = self.instance.np_fitness(x, self.metrics, self.run)
                if x_fit > x_best_fit:
                    x_best = x.copy()
                    x_best_fit = x_fit
                    flag = True

        return x_best, x_best_fit
        







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