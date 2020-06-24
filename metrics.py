import glob
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import pathlib

class Metrics:

    def __init__(self, name, no_runs, no_evals, hyper_params={}):
        '''
        Metrics class that remembers the fitness per evaluation

        '''

        self.name = name

        self.num_evaluations = no_evals

        self.fitness = np.zeros((no_runs, self.num_evaluations))

        # Needs to be set 
        self.best_fitness_function = ()

        self.params = hyper_params

        self.evaluations = np.zeros((no_runs), dtype=np.int)

        self.last_eval = 0
        self.last_val = 0

        
    def update_metrics(self, run, no_eval = None, update_prev = False):
        '''
        Updates the metrics

        :param no_eval: if not given defaults to self.evaluations[run]
        :param update_prev: if true, updates all previous fitness per evaluation
        :return: True if the evaluation exceeded the max number of evaluations -> Done
        '''

        if no_eval == None:
            no_eval = self.evaluations[run]
        
        new_val = self.best_fitness_function()

        if update_prev:
            for i in range(min(self.last_eval, self.num_evaluations), min(no_eval, self.num_evaluations)):
                self.fitness[run, i] = self.last_val

        if no_eval >= self.num_evaluations:
            return True

        self.fitness[run, no_eval] = new_val
        self.last_eval = no_eval
        self.last_val = new_val
        return False


    @staticmethod
    def write_to_file(directory, list_of_metric_objects = [], prefix='metric'):
        '''
        Write the metrics objects to file, uses all metrics given in mean_metrics

        '''

        mean_metrics = {
            'fitness' : ['evaluation', 'fitness'],
        }

        if directory == '' or len(list_of_metric_objects) == 0:
            raise KeyError

        BASE_PATH = pathlib.Path(__file__).parent.absolute()
        os.chdir(BASE_PATH)
        if not os.path.exists(directory):
            os.mkdir(directory)

        for metric, headers in mean_metrics.items():

            os.chdir(BASE_PATH)
            os.chdir(directory)

            filename = f'{metric}.dat' if prefix == '' else f'{prefix}metric[{metric}].dat'

            if os.path.exists(filename):
                os.remove(filename)

            header = headers[0] + '\t' + '\t'.join(f'{obj.name}_{headers[1]}_mean\t{obj.name}_{headers[1]}_std_up\t{obj.name}_{headers[1]}_std_down' for obj in list_of_metric_objects)

            with open(filename, "w") as f:
                f.write(header + '\n')
                for i in range(len(getattr(list_of_metric_objects[0], metric)[0])):
                    data = '\t\t'.join(f'{np.round(np.mean(getattr(obj, metric), axis=0)[i], 5)}\t\t' + \
                        f'{np.round(np.mean(getattr(obj, metric), axis=0)[i] + np.std(getattr(obj, metric), axis=0)[i], 5)}\t\t' + \
                            f'{np.round(np.mean(getattr(obj, metric), axis=0)[i] - np.std(getattr(obj, metric), axis=0)[i], 5)}' \
                                for obj in list_of_metric_objects)
                    f.write(f'{i+1}\t\t{data}\n')
            f.close()

    # @staticmethod
    # def subplots(metrics=[self], metric='fitness', hyper_param=[[None], [None]], begin=0, end=-1):

    #     if len(metrics) > 0:

    #         ######################################################
    #         if metric not in ['fitness', 'epoch_fitness']:
    #             raise KeyError

    #         if hyper_param == [[None], [None]]:
    #             raise KeyError


    #         xlabel = {
    #             'fitness': 'Evaluations',
    #             'epoch_fitness': 'Epochs',
    #             'evals_epoch_fitness': 'Evaluations'
    #         }[metrics]

    #         param_rows = hyper_param[0]
    #         param_cols = hyper_param[1]

    #         data = {}

    #         for met in metrics:
    #             if met.name not in data:
    #                 data[met.name] = {}
    #             for r in param_rows:
    #                 if r not in data[met.name]:
    #                     data[met.name][r] = {}
    #                 for c in param_cols:
    #                     if c not in data[met.name][r]:
    #                         data[met.name][r][c] = {}
                        
    #                     if 'mean' not in data[met.name][r][c]:
    #                         data[met.name][r][c]['mean'] = met.
                        

    #         NUM_ROWS = len(param_rows) # offset
    #         NUM_COLS = len(param_cols) # rho

    #         font_size = 14

    #         #Options
    #         params = {'font.size' : font_size,
    #                 'font.family' : 'serif',
    #                 }
    #         plt.rcParams.update(params) 


    #         fig, axs = plt.subplots(
    #             NUM_ROWS, NUM_COLS, sharex=True, sharey=True)
    #         for r in param_rows:
    #             for c in param_cols:
                    
    #                 max_opt = mediator_max_opt[metric][r][c]['mean'][:NUM_RESULTS]
    #                 random = mediator_random[metric][r][c]['mean'][:NUM_RESULTS]
    #                 mbie_data = mbie[metric]['mean'][:NUM_RESULTS]
    #                 mbie_eb_data = mbie_eb[metric]['mean'][:NUM_RESULTS]
    #                 index = range(NUM_RESULTS)

    #                 max_opt_std_up = mediator_max_opt[metric][r][c]['std_up'][:NUM_RESULTS]
    #                 max_opt_std_down = mediator_max_opt[metric][r][c]['std_down'][:NUM_RESULTS]
    #                 random_std_up = mediator_random[metric][r][c]['std_up'][:NUM_RESULTS]
    #                 random_std_down = mediator_random[metric][r][c]['std_down'][:NUM_RESULTS]
    #                 mbie_data_std_up = mbie[metric]['std_up'][:NUM_RESULTS]
    #                 mbie_data_std_down = mbie[metric]['std_down'][:NUM_RESULTS]
    #                 mbie_eb_data_std_up = mbie_eb[metric]['std_up'][:NUM_RESULTS]
    #                 mbie_eb_data_std_down = mbie_eb[metric]['std_down'][:NUM_RESULTS]

    #                 if NUM_ROWS == 1 and NUM_COLS == 1:
    #                     # Plot the sds.
    #                     axs.fill_between(
    #                         index, max_opt_std_down, max_opt_std_up, color = 'red', alpha = 0.1)
    #                     axs.fill_between(
    #                         index, random_std_down, random_std_up, color = 'orange', alpha = 0.1)
    #                     axs.fill_between(
    #                         index, mbie_data_std_down, mbie_data_std_up, color = 'blue', alpha = 0.1)
    #                     axs.fill_between(
    #                         index, mbie_eb_data_std_down, mbie_eb_data_std_up, color = 'cyan', alpha = 0.1)

    #                     # Plot the means.
    #                     sns.lineplot(x = index, y = max_opt, ax = axs, color = 'red', alpha = 0.6, label='Max-Opt')
    #                     sns.lineplot(x = index, y = random, ax = axs, color = 'orange', alpha = 0.6, label='Random')
    #                     sns.lineplot(x = index, y = mbie_data, ax = axs, color = 'blue', alpha = 0.6, label='MBIE')
    #                     sns.lineplot(x = index, y = mbie_eb_data, ax = axs, color = 'cyan', alpha = 0.6, label='MBIE-EB')
    #                     subplot_title = r'($\kappa=$' + f'{r}' + r', $\rho=$' + f'{c}' + ')'
    #                     axs.set_title(subplot_title, fontsize = font_size)
    #                     axs.get_legend().remove()

    #                     handles, labels = axs.get_legend_handles_labels()


    #                 if NUM_ROWS == 1:
    #                     # Plot the sds.
    #                     axs[c].fill_between(
    #                         index, max_opt_std_down, max_opt_std_up, color = 'red', alpha = 0.1)
    #                     axs[c].fill_between(
    #                         index, random_std_down, random_std_up, color = 'orange', alpha = 0.1)
    #                     axs[c].fill_between(
    #                         index, mbie_data_std_down, mbie_data_std_up, color = 'blue', alpha = 0.1)
    #                     axs[c].fill_between(
    #                         index, mbie_eb_data_std_down, mbie_eb_data_std_up, color = 'cyan', alpha = 0.1)

    #                     # Plot the means.
    #                     sns.lineplot(x = index, y = max_opt, ax = axs[c], color = 'red', alpha = 0.6, label='Max-Opt')
    #                     sns.lineplot(x = index, y = random, ax = axs[c], color = 'orange', alpha = 0.6, label='Random')
    #                     sns.lineplot(x = index, y = mbie_data, ax = axs[c], color = 'blue', alpha = 0.6, label='MBIE')
    #                     sns.lineplot(x = index, y = mbie_eb_data, ax = axs[c], color = 'cyan', alpha = 0.6, label='MBIE-EB')
    #                     subplot_title = r'($\kappa=$' + f'{r}' + r', $\rho=$' + f'{c}' + ')'
    #                     axs[c].set_title(subplot_title, fontsize = font_size)
    #                     axs[c].get_legend().remove()

    #                     handles, labels = axs[c].get_legend_handles_labels()

    #                 else:
    #                     # Plot the sds.
    #                     axs[r, c].fill_between(
    #                         index, max_opt_std_down, max_opt_std_up, color = 'red', alpha = 0.1)
    #                     axs[r, c].fill_between(
    #                         index, random_std_down, random_std_up, color = 'orange', alpha = 0.1)
    #                     axs[r, c].fill_between(
    #                         index, mbie_data_std_down, mbie_data_std_up, color = 'blue', alpha = 0.1)
    #                     axs[r, c].fill_between(
    #                         index, mbie_eb_data_std_down, mbie_eb_data_std_up, color = 'cyan', alpha = 0.1)

    #                     # Plot the means.
    #                     sns.lineplot(x = index, y = max_opt, ax = axs[r, c], color = 'red', alpha = 0.6, label='Max-Opt')
    #                     sns.lineplot(x = index, y = random, ax = axs[r, c], color = 'orange', alpha = 0.6, label='Random')
    #                     sns.lineplot(x = index, y = mbie_data, ax = axs[r, c], color = 'blue', alpha = 0.6, label='MBIE')
    #                     sns.lineplot(x = index, y = mbie_eb_data, ax = axs[r, c], color = 'cyan', alpha = 0.6, label='MBIE-EB')
    #                     subplot_title = r'($\kappa=$' + f'{r}' + r', $\rho=$' + f'{c}' + ')'
    #                     axs[r, c].set_title(subplot_title, fontsize = font_size)
    #                     axs[r, c].get_legend().remove()

    #                     handles, labels = axs[r, c].get_legend_handles_labels()
    #         fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1), ncol=4, fontsize=f'{font_size}')

    #         plt.subplots_adjust(top=0.92, right=0.98)

    #         plt.ticklabel_format(style='scientific', axis='y', scilimits=(0, 3))
    #         fig.tight_layout(pad=0.003)
    #         # fig.suptitle(, size=font_size, y=0.95)
    #         fig.text(0.5, 0.005, 'Number of steps', ha='center', fontsize = font_size)
    #         fig.text(0.003, 0.5, xlabel, va='center', rotation='vertical', fontsize = font_size)
    #         plt.show()
