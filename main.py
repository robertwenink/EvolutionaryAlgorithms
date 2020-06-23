import os
from time import time
from maxcut import *
from particle_swarm import *
from metrics import Metrics
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

    
    BPSO = BinaryParticleSwarmOptimization(instances[2], 300, 2000)

    t = time()
    BPSO.np_run(False, False, False, 5)
    print(f'{time() - t} seconds\n')

    t = time()
    BPSO.np_run()
    print(f'{time() - t} seconds\n')

    # TODO
    # Call solvers
    # FGA = FastGA(instances[2], 48, 40)
    # t = time()
    # FGA.run()
    # print(f'{time() - t} seconds')

def filename_generator(n, w, p, r, prefix=''):
    return f'n-{n}-w-{w}-p-{p}-r-{r}' if prefix == '' else f'{prefix}-n-{n}-w-{w}-p-{p}-r-{r}'

def get_params_from_filename(filename):
    temp = filename
    r = int(temp.split('-r-')[1])
    temp = temp.split('-r-')[0]
    p = float(temp.split('-p-')[1])
    temp = temp.split('-p-')[0]
    w = int(temp.split('-w-')[1])
    temp = temp.split('-w-')[0]
    n = int(temp.split('-n-')[1])
    temp = temp.split('-n-')[0]
    return n, w, p ,r

def get_random_instances(directory):

    instances, names = [], []

    for r in range(10):
        for n in [250, 500, 1000, 2000]:
            for w in [200]:
                for p in [0.25, 0.5, 0.75]:

                    filename = filename_generator(n, w, p, r)
                    if os.path.exists(directory + filename + '.npy'):
                        instances.append(NP_MaxCut_From_File(filename, directory))
                    else:
                        instances.append(NP_MaxCut_Random(nodes=n, max_weight=w, edge_prob=p))
                        instances[-1].write_to_file(filename, directory)
                    names.append(filename)

    return instances, names
    
def run_instances(names, instances):

    NO_RUNS = 10
    NO_EVALS = 1e5
    NO_EPOCHS = 1e3
    NO_PARTICLES = [100, 250, 500, 1000]

    for no_particles in NO_PARTICLES:
        for i, instance in enumerate(instances):
            n, w, p, r = get_params_from_filename(names[i])

            GBO_init = False
            GBO = False
            local_search = False
            norm_velo = False

            hyper_params = {
                'num_particles' : no_particles,
                'nodes': n,
                'maxweight': w,
                'edgeprob': p,
                'GBO': GBO,
                'GBO_init': GBO_init,
                'local_search': local_search,
                'norm_velo': norm_velo
            }
            metrics = Metrics('BPSO_' + names[i], NO_RUNS, NO_EVALS, NO_EPOCHS, hyper_params)
            for run in range(NO_RUNS):
                VectorizedBinaryParticleSwarmOptimization(instance, metrics, no_particles, 
                    r, NO_EPOCHS).np_run(GBO, GBO_init, local_search, norm_velo)

        

if __name__ == '__main__':
    # main(instances_directory, opt_directory, sub_directory)

    names, instances, names = get_random_instances('experiment/')
    
    run_instances(names, instances)
