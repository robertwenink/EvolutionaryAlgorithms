import glob
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

hp_example = {
    'num_particles' : 0,
    'c1' : 0.0,
    'c2' : 0.0,
    'v_bound' : 0.0,
    'nodes': 0,
    'maxweight': 0,
    'edgeprob': 0.0,
    'GBO': False,
    'local_search': False
}

def get_params_from_metrics_filename(filename):
    hyper_params = {}
    temp = filename
    temp = temp.replace('.dat', '')
    if len(temp.split('/')) > 1:
        temp = temp.split('/')[-1]
    for par in temp.split(']'):
        if len(par) > 0:
            if par.split('[')[0] in hp_example:
                t = type(hp_example[par.split('[')[0]])
                if t == bool:
                    hyper_params[par.split('[')[0]] = par.split('[')[1] == 'True'
                else:
                    hyper_params[par.split('[')[0]] = t(par.split('[')[1])
            else:
                hyper_params[par.split('[')[0]] = par.split('[')[1]
    return hyper_params

def fill_dict_with_dat(dict, data, columns):
    if metric not in dict:
        dict[metric] = {}
    if hyper_params[row_params] not in dict[metric]:
        dict[metric][hyper_params[row_params]] = {}
    if hyper_params[col_params] not in dict[metric][hyper_params[row_params]]:
        dict[metric][hyper_params[row_params]][hyper_params[col_params]] = {}
    if hyper_params[lin_params] not in dict[metric][hyper_params[row_params]][hyper_params[col_params]]:
        dict[metric][hyper_params[row_params]][hyper_params[col_params]][hyper_params[lin_params]] = {}

    dict[metric][hyper_params[row_params]][hyper_params[col_params]][hyper_params[lin_params]]['mean'] = np.array(data[columns[1]], dtype=float)[slices]
    dict[metric][hyper_params[row_params]][hyper_params[col_params]][hyper_params[lin_params]]['std_up'] = np.array(data[columns[2]], dtype=float)[slices]
    dict[metric][hyper_params[row_params]][hyper_params[col_params]][hyper_params[lin_params]]['std_down'] = np.array(data[columns[3]], dtype=float)[slices]

DATA_ROOT = 'metrics-PSO/'

NUM_RESULTS = 20000

stepsize = 100

NUM_RESULTS_LEFT = int(NUM_RESULTS/stepsize)

slices = np.arange(NUM_RESULTS) % stepsize == 0

BBO = {}
GBO = {}
local_search = {}
GBO_local_search = {}

###################### files #####################

baseline_files = glob.glob(DATA_ROOT + '*.dat')

metric = 'fitness'
ylabel = 'Max Cut Fitness'
xlabel = 'Evaluations'

# ADAPT FOR YOUR PARAMS
row_params = 'nodes'
row_values = [100]
row_labels = r'$n=$'

col_params = 'c1'
col_values = [0.5, 1.0, 2, 5]
col_labels = r'$c_1=$'

lin_params = 'c2'
lin_values = [0.5, 1.0, 2, 5]
lin_labels = r'$c_2=$'
lin_colors = ['red'] if len(lin_values) == 1 else ['red', 'orange', 'cyan', 'blue']

for file in baseline_files:

    hyper_params = get_params_from_metrics_filename(file)

    data = pd.read_csv(file, sep = r'\t+', engine = 'python')
    columns = np.array(data.columns)

    # BBO
    if not hyper_params['GBO'] and not hyper_params['local_search']:

        fill_dict_with_dat(BBO, data, columns)

    # GBO
    elif hyper_params['GBO'] and not hyper_params['local_search']:

        fill_dict_with_dat(GBO, data, columns)

    # local_search
    elif not hyper_params['GBO'] and hyper_params['local_search']:

        fill_dict_with_dat(local_search, data, columns)

    # GBO_local_search
    elif hyper_params['GBO'] and hyper_params['local_search']:
        
        fill_dict_with_dat(GBO_local_search, data, columns)

######################################################


NUM_ROWS = len(row_values)
NUM_COLS = len(col_values)
NUM_LINS = len(lin_values)

font_size = 14

#Options
params = {'font.size' : font_size,
          'font.family' : 'serif',
          }
plt.rcParams.update(params) 


fig, axs = plt.subplots(
    NUM_ROWS, NUM_COLS, sharey='row') #, sharey=True
for r in range(NUM_ROWS):
    for c in range(NUM_COLS):
        for l in range(NUM_LINS):

            r_val = row_values[r]
            c_val = col_values[c]
            l_val = lin_values[l]
            l_clr = lin_colors[l]

            if metric in BBO:
                BBO_mean = BBO[metric][r_val][c_val][l_val]['mean'][:NUM_RESULTS_LEFT]
                BBO_std_up = BBO[metric][r_val][c_val][l_val]['std_up'][:NUM_RESULTS_LEFT]
                BBO_std_down = BBO[metric][r_val][c_val][l_val]['std_down'][:NUM_RESULTS_LEFT]
                BBO_label = f'BBO {lin_labels}{l_val}'
            
            if metric in GBO:
                GBO_mean = GBO[metric][r_val][c_val][l_val]['mean'][:NUM_RESULTS_LEFT]
                GBO_std_up = GBO[metric][r_val][c_val][l_val]['std_up'][:NUM_RESULTS_LEFT]
                GBO_std_down = GBO[metric][r_val][c_val][l_val]['std_down'][:NUM_RESULTS_LEFT]
                GBO_label = f'GBO {lin_labels}{l_val}'

            if metric in local_search:
                local_search_mean = local_search[metric][r_val][c_val][l_val]['mean'][:NUM_RESULTS_LEFT]
                local_search_data_std_up = local_search[metric][r_val][c_val][l_val]['std_up'][:NUM_RESULTS_LEFT]
                local_search_data_std_down = local_search[metric][r_val][c_val][l_val]['std_down'][:NUM_RESULTS_LEFT]
                local_search_label = f'LS {lin_labels}{l_val}'

            if metric in GBO_local_search:
                GBO_local_search_mean = GBO_local_search[metric][r_val][c_val][l_val]['mean'][:NUM_RESULTS_LEFT]
                GBO_local_search_data_std_up = GBO_local_search[metric][r_val][c_val][l_val]['std_up'][:NUM_RESULTS_LEFT]
                GBO_local_search_data_std_down = GBO_local_search[metric][r_val][c_val][l_val]['std_down'][:NUM_RESULTS_LEFT]
                GBO_local_search_label = f'GBO & LS {lin_labels}{l_val}'

            index = np.arange(NUM_RESULTS)[slices]

            if NUM_ROWS == 1 and NUM_COLS == 1:
                # Plot the sds.
                if metric in BBO:
                    axs.fill_between(
                        index, BBO_std_down, BBO_std_up, color = l_clr, alpha = 0.1)
                if metric in GBO:
                    axs.fill_between(
                        index, GBO_std_down, GBO_std_up, color = 'orange', alpha = 0.1)
                if metric in local_search:
                    axs.fill_between(
                        index, local_search_data_std_down, local_search_data_std_up, color = 'blue', alpha = 0.1)
                if metric in GBO_local_search:
                    axs.fill_between(
                        index, GBO_local_search_data_std_down, GBO_local_search_data_std_up, color = 'cyan', alpha = 0.1)

                # Plot the means.
                if metric in BBO:
                    sns.lineplot(x = index, y = BBO_mean, ax = axs, color = l_clr, alpha = 0.6, label=BBO_label)
                if metric in GBO:
                    sns.lineplot(x = index, y = GBO_mean, ax = axs, color = 'orange', alpha = 0.6, label=GBO_label)
                if metric in local_search:
                    sns.lineplot(x = index, y = local_search_mean, ax = axs, color = 'blue', alpha = 0.6, label=local_search_label)
                if metric in GBO_local_search:
                    sns.lineplot(x = index, y = GBO_local_search_mean, ax = axs, color = 'cyan', alpha = 0.6, label=GBO_local_search_label)
                subplot_title = f'({row_labels}{r_val}, {col_labels}{c_val})'
                axs.set_title(subplot_title, fontsize = font_size)
                axs.get_legend().remove()
                axs.set(yscale='log')

                handles, labels = axs.get_legend_handles_labels()


            elif NUM_ROWS == 1 or NUM_COLS == 1:
                x = max(r,c)
                # Plot the sds.
                if metric in BBO:
                    axs[x].fill_between(
                        index, BBO_std_down, BBO_std_up, color = l_clr, alpha = 0.1)
                if metric in GBO:
                    axs[xlabel].fill_between(
                        index, GBO_std_down, GBO_std_up, color = 'orange', alpha = 0.1)
                if metric in local_search:
                    axs[x].fill_between(
                        index, local_search_data_std_down, local_search_data_std_up, color = 'blue', alpha = 0.1)
                if metric in GBO_local_search:
                    axs[x].fill_between(
                        index, GBO_local_search_data_std_down, GBO_local_search_data_std_up, color = 'cyan', alpha = 0.1)

                # Plot the means.
                if metric in BBO:
                    sns.lineplot(x = index, y = BBO_mean, ax = axs[x], color = l_clr, alpha = 0.6, label=BBO_label)
                if metric in GBO:
                    sns.lineplot(x = index, y = GBO_mean, ax = axs[x], color = 'orange', alpha = 0.6, label=GBO_label)
                if metric in local_search:
                    sns.lineplot(x = index, y = local_search_mean, ax = axs[x], color = 'blue', alpha = 0.6, label=local_search_label)
                if metric in GBO_local_search:
                    sns.lineplot(x = index, y = GBO_local_search_mean, ax = axs[x], color = 'cyan', alpha = 0.6, label=GBO_local_search_label)
                subplot_title = f'({row_labels}{r_val}, {col_labels}{c_val})'
                axs[x].set_title(subplot_title, fontsize = font_size)
                axs[x].get_legend().remove()
                axs[x].set(yscale='log')

                handles, labels = axs[x].get_legend_handles_labels()

            else:
                # Plot the sds.
                if metric in BBO:
                    axs[r, c].fill_between(
                        index, BBO_std_down, BBO_std_up, color = l_clr, alpha = 0.1)
                if metric in GBO:
                    axs[r, c].fill_between(
                        index, GBO_std_down, GBO_std_up, color = 'orange', alpha = 0.1)
                if metric in local_search:
                    axs[r, c].fill_between(
                        index, local_search_data_std_down, local_search_data_std_up, color = 'blue', alpha = 0.1)
                if metric in GBO_local_search:
                    axs[r, c].fill_between(
                        index, GBO_local_search_data_std_down, GBO_local_search_data_std_up, color = 'cyan', alpha = 0.1)

                # Plot the means.
                if metric in BBO:
                    sns.lineplot(x = index, y = BBO_mean, ax = axs[r, c], color = l_clr, alpha = 0.6, label=BBO_label)
                if metric in GBO:
                    sns.lineplot(x = index, y = GBO_mean, ax = axs[r, c], color = 'orange', alpha = 0.6, label=GBO_label)
                if metric in local_search:
                    sns.lineplot(x = index, y = local_search_mean, ax = axs[r, c], color = 'blue', alpha = 0.6, label=local_search_label)
                if metric in GBO_local_search:
                    sns.lineplot(x = index, y = GBO_local_search_mean, ax = axs[r, c], color = 'cyan', alpha = 0.6, label=GBO_local_search_label)
                subplot_title = f'({row_labels}{r_val}, {col_labels}{c_val})'
                axs[r, c].set_title(subplot_title, fontsize = font_size)
                axs[r, c].get_legend().remove()
                axs[r, c].set(yscale='log')

                handles, labels = axs[r, c].get_legend_handles_labels()


fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1), ncol=4, fontsize=f'{font_size}')

plt.subplots_adjust(top=0.92, right=0.98)

# plt.ticklabel_format(style='scientific', axis='y', scilimits=(0, 3))
fig.tight_layout(pad=0.003)
# fig.suptitle(, size=font_size, y=0.95)
fig.text(0.5, 0.005, xlabel, ha='center', fontsize = font_size)
fig.text(0.003, 0.5, ylabel, va='center', rotation='vertical', fontsize = font_size)
plt.show()
