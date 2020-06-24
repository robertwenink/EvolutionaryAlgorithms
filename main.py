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

def instance_filename_generator(n, w, p, prefix=''):
    return f'n-{n}-w-{w}-p-{p}' if prefix == '' else f'{prefix}-n-{n}-w-{w}-p-{p}'

def get_params_from_instance_filename(filename):
    temp = filename
    p = float(temp.split('-p-')[1])
    temp = temp.split('-p-')[0]
    w = int(temp.split('-w-')[1])
    temp = temp.split('-w-')[0]
    n = int(temp.split('n-')[1])
    return n, w, p

def metrics_filename_generator(hyper_params):
    string = ''
    for k, v in hyper_params.items():
        string += f'{k}[{v}]'
    return string

def get_params_from_metrics_filename(filename):
    hyper_params = {}
    temp = filename
    if len(temp.split('.')) > 1:
        temp = temp.split('.')[0]
    for par in temp.split(']'):
        hyper_params[par.split('[')[0]] = par.split('[')[1]
    return hyper_params

def get_random_instances(directory):

    instances, names = [], []

    for n in [50, 100, 250, 500]:
        for w in [100]:
            for p in [0.5]:

                filename = instance_filename_generator(n, w, p)
                if os.path.exists(directory + filename + '.npy'):
                    instances.append(NP_MaxCut_From_File(filename, directory))
                else:
                    instances.append(NP_MaxCut_Random(nodes=n, max_weight=w, edge_prob=p))
                    instances[-1].write_to_file(filename, directory)
                names.append(filename)

    return names, instances
    
def run_instances(names, instances):

    NO_RUNS = 5
    NO_EVALS = 15000
    NO_PARTICLES = [25, 50, 100, 250]

    for i, instance in enumerate(instances):
        for no_particles in NO_PARTICLES:
            n, w, p = get_params_from_instance_filename(names[i])

            GBO = False
            local_search = False
            norm_velo = False

            hyper_params = {
                'num_particles' : no_particles,
                'nodes': n,
                'maxweight': w,
                'edgeprob': p,
                'GBO': GBO,
                'local_search': local_search,
                'norm_velo': norm_velo
            }
            metrics = Metrics('BPSO', NO_RUNS, NO_EVALS, hyper_params)
            for run in range(NO_RUNS):
                metrics.run = run
                VectorizedBinaryParticleSwarmOptimization(instance, metrics, no_particles, 
                    run).np_run(GBO, local_search, norm_velo)

            metrics_file_prefix = metrics_filename_generator(hyper_params)
            
            metrics.write_to_file('metrics-PSO', [metrics], metrics_file_prefix)

        

if __name__ == '__main__':
    # main(instances_directory, opt_directory, sub_directory)

    names, instances = get_random_instances('experiment/')
    
    run_instances(names, instances)
