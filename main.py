import os
from time import time
from maxcut import MaxCut
from particle_swarm import BinaryParticleSwarmOptimization

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

    BPSO = BinaryParticleSwarmOptimization(instances[2], 48, 20)
    t = time()
    BPSO.run()
    print(f'{time() - t} seconds')

    # TODO
    # Call solvers
    

if __name__ == '__main__':
    main(instances_directory, opt_directory, sub_directory)