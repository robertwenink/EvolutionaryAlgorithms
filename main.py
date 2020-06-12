import os
from time import time
from maxcut import MaxCut
from particle_swarm import BinaryParticleSwarmOptimization
# from FastGA.fastGA import FastGA

instances_directory = 'instances/'
opt_directory = 'opts/'
sub_directory = ''

def main(instances_directory, opt_directory, sub_directory):
    files = os.listdir(instances_directory + sub_directory)
    instances = []
    for f in files:
        if f[0] != '.':
            instance = MaxCut(f, instances_directory, opt_directory)
            instances.append(instance)

    BPSO = BinaryParticleSwarmOptimization(instances[2], 2000, 500)

    t = time()
    BPSO.np_run()
    print(f'{time() - t} seconds\n\n')

    t = time()
    BPSO.run()
    print(f'{time() - t} seconds')

    # TODO
    # Call solvers
    # FGA = FastGA(instances[2], 48, 40)
    # t = time()
    # FGA.run()
    # print(f'{time() - t} seconds')
    

if __name__ == '__main__':
    main(instances_directory, opt_directory, sub_directory)