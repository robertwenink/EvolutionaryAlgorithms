import os
from maxcut import MaxCut
from particle_swarm import BinaryParticleSwarmOptimization

instances_directory = 'instances/'
opt_directory = 'opts/'
sub_directory = ''

def main(instances_directory, opt_directory, sub_directory):
    files = os.listdir(instances_directory + sub_directory)
    instances = []
    for f in files:
        instance = MaxCut(f, instances_directory, opt_directory)
        instances.append(instance)

    BPSO = BinaryParticleSwarmOptimization(instances[3], 50, 50)
    BPSO.run()

    # TODO
    # Call solvers
    

if __name__ == '__main__':
    main(instances_directory, opt_directory, sub_directory)