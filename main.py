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

    
    # BPSO = BinaryParticleSwarmOptimization(instances[2], 300, 2000)

    # t = time()
    # BPSO.np_run(False, False, False, 5)
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

    for n in [100]:
    # for n in [50, 100, 250, 500]:
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
    NO_EVALS = 20000

    tests = [
        {'c1' : [0.5, 1.0, 2, 5]},
        {'c2' : [0.5, 1.0, 2, 5]}
    ]

    test_params = []
    for k1, v1 in tests[0].items():
        for val1 in v1:
            if len(tests) > 1:
                for k2, v2 in tests[1].items():
                    for val2 in v2:
                        if len(tests) > 2:
                            for k3, v3 in tests[2].items():
                                for val3 in v3:
                                    test_params.append({k1: val1, k2: val2, k3: val3})
                        else:
                            test_params.append({k1: val1, k2: val2})
            else:
                test_params.append({k1: val1})


    for i, instance in enumerate(instances):

        n, w, p = get_params_from_instance_filename(names[i])

        experiment_params = []

        for t in test_params:

            params = {                  # DEFAULT VALUES
                'num_particles' : 25,
                'c1': 1.0,
                'c2': 1.0,
                'v_bound' : 6.0,
                'nodes': n,
                'maxweight': w,
                'edgeprob': p,
                'GBO': False,
                'local_search': False
            }

            for key, value in t.items():
                params[key] = value
            experiment_params.append(params)


        for hyper_params in experiment_params:

            no_particles = hyper_params['num_particles']
            c1 = hyper_params['c1']
            c2 = hyper_params['c2']
            v_bound = hyper_params['v_bound']
            GBO = hyper_params['GBO']
            local_search = hyper_params['local_search']

            metrics = Metrics('BPSO', NO_RUNS, NO_EVALS, hyper_params)
            
            for run in range(NO_RUNS):
                metrics.run = run
                VectorizedBinaryParticleSwarmOptimization(instance, metrics, no_particles, 
                    run, c1, c2, v_bound).np_run(GBO, local_search)

            metrics_file_prefix = metrics_filename_generator(hyper_params)
            
            metrics.write_to_file('metrics-PSO', [metrics], metrics_file_prefix)

        

if __name__ == '__main__':
    # main(instances_directory, opt_directory, sub_directory)

    names, instances = get_random_instances('experiment/')
    
    run_instances(names, instances)
