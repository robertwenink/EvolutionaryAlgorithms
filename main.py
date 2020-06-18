import os
from time import time
from maxcut import *
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

    # BPSO = BinaryParticleSwarmOptimization(instances[2], 200, 1000)

    # t = time()
    # BPSO.np_run(True)
    # print(f'{time() - t} seconds\n')

    # t = time()
    # BPSO.np_run()
    # print(f'{time() - t} seconds\n')

    # TODO
    # Call solvers
    # FGA = FastGA(instances[2], 48, 40)
    # t = time()
    # FGA.run()
    # print(f'{time() - t} seconds')


def random_instance():

    instance = NP_MaxCut_Random(nodes=500, max_weight=200, edge_prob=0.5)

    instance.write_to_file('testname', 'testdirectory')

    instance2 = NP_MaxCut_From_File('testname', 'testdirectory')

    # BPSO_GBO = BinaryParticleSwarmOptimization(instance, 200, 1000).np_run(GBO=True)
    BPSO = BinaryParticleSwarmOptimization(instance, 200, 1000).np_run()
    

if __name__ == '__main__':
    main(instances_directory, opt_directory, sub_directory)
    # random_instance()
    



