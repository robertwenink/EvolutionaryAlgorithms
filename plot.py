import glob
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

DATA_ROOT = 'river-swim/'

NUM_RESULTS = 250

###################### MBIE & EB #####################
baseline_files = glob.glob(DATA_ROOT + '/*.dat')
mbie = {}
mbie_eb = {}
for file in baseline_files:

    # File is mbie-eb.
    if file.startswith('river-swim/mbie-eb'):
        metric = file.replace('river-swim/mbie-eb_', '').replace('.dat', '')
        if metric not in (
            'sample_complexity','coverage_error_squared_R', 'reward_timeline', 'state_action_count'):
            data = pd.read_csv(file, sep = r'\t+', engine = 'python')
            columns = np.array(data.columns)
            mbie_eb[metric] = {}
            mbie_eb[metric]['mean'] = np.array(data[columns[1]])
            mbie_eb[metric]['std_up'] = np.array(data[columns[2]])
            mbie_eb[metric]['std_down'] = np.array(data[columns[3]])

    # File is mbie.
    else:
        metric = file.replace('river-swim/mbie_', '').replace('.dat', '')
        if metric not in (
            'sample_complexity','coverage_error_squared_R', 'reward_timeline', 'state_action_count'):
            data = pd.read_csv(file, sep = r'\t+', engine = 'python')
            columns = np.array(data.columns)
            mbie[metric] = {}
            mbie[metric]['mean'] = np.array(data[columns[1]], dtype=float)
            mbie[metric]['std_up'] = np.array(data[columns[2]], dtype=float)
            mbie[metric]['std_down'] = np.array(data[columns[3]], dtype=float)

######################################################

###################### Max- opt ######################
mediator_max_opt_files = glob.glob(DATA_ROOT + 'mediator-max-opt/*.dat')
mediator_max_opt = {}

for file in mediator_max_opt_files:
    filename = file.replace(
        'river-swim/mediator-max-opt/mediator-max-opt-', '')
    info = filename.split('-')
    offset = float(info[0])
    rho = float(info[1].split('_')[0])
    metric = info[1].replace(f'{str(rho)}_', '').replace('.dat', '')
    if metric not in (
        'sample_complexity','coverage_error_squared_R', 'reward_timeline', 'state_action_count'):
        data = pd.read_csv(file, sep = r'\t+', engine = 'python')
        columns = np.array(data.columns)

        # Traverse to find right dict.
        if metric not in mediator_max_opt:
            mediator_max_opt[metric] = {}
        if offset not in mediator_max_opt[metric]:
            mediator_max_opt[metric][offset] = {}
        if rho not in mediator_max_opt[metric][offset]:
            mediator_max_opt[metric][offset][rho] = {}

        mediator_max_opt[metric][offset][rho]['mean'] = \
            np.array(data[columns[1]], dtype=float)
        mediator_max_opt[metric][offset][rho]['std_up'] \
            = np.array(data[columns[2]], dtype=float)
        mediator_max_opt[metric][offset][rho]['std_down'] \
            = np.array(data[columns[3]], dtype=float)

######################################################

####################### Random #######################
mediator_random_files = glob.glob(DATA_ROOT + 'mediator-random/*.dat')
mediator_random = {}

for file in mediator_random_files:
    filename = file.replace(
        'river-swim/mediator-random/mediator-random-', '')
    info = filename.split('-')
    offset = float(info[0])
    rho = float(info[1].split('_')[0])
    metric = info[1].replace(f'{str(rho)}_', '').replace('.dat', '')
    if metric not in (
        'sample_complexity','coverage_error_squared_R', 'reward_timeline', 'state_action_count'):
        data = pd.read_csv(file, sep = r'\t+', engine = 'python')
        columns = np.array(data.columns)

        # Traverse to find right dict.
        if metric not in mediator_random:
            mediator_random[metric] = {}
        if offset not in mediator_random[metric]:
            mediator_random[metric][offset] = {}
        if rho not in mediator_random[metric][offset]:
            mediator_random[metric][offset][rho] = {}

        mediator_random[metric][offset][rho]['mean'] = \
            np.array(data[columns[1]])
        mediator_random[metric][offset][rho]['std_up'] \
            = np.array(data[columns[2]])
        mediator_random[metric][offset][rho]['std_down'] \
            = np.array(data[columns[3]])

######################################################

metric = 'cumulative_instantaneous_loss' # CHANGE FOR PLOT.
xlabel = 'Cumulative instantaneous loss'

# metric = 'cumulative_rewards' # CHANGE FOR PLOT.
# xlabel = 'Cumulative reward'

# metric = 'coverage_error_squared_T' # CHANGE FOR PLOT.
# xlabel = 'Coverage error transition probability'


# offsets = [1.0]
offsets = [0, 0.1, 0.2, 0.3]
rhos = [0.02, 0.08, 0.16, 0.32]

NUM_ROWS = len(offsets) # offset
NUM_COLS = len(rhos) # rho

font_size = 14

#Options
params = {'font.size' : font_size,
          'font.family' : 'serif',
          }
plt.rcParams.update(params) 


fig, axs = plt.subplots(
    NUM_ROWS, NUM_COLS, sharex=True, sharey=True)
for r in range(NUM_ROWS):
    for c in range(NUM_COLS):
        rho = rhos[c]
        offset = offsets[r]
        max_opt = mediator_max_opt[metric][offset][rho]['mean'][:NUM_RESULTS]
        random = mediator_random[metric][offset][rho]['mean'][:NUM_RESULTS]
        mbie_data = mbie[metric]['mean'][:NUM_RESULTS]
        mbie_eb_data = mbie_eb[metric]['mean'][:NUM_RESULTS]
        index = range(NUM_RESULTS)

        max_opt_std_up = mediator_max_opt[metric][offset][rho]['std_up'][:NUM_RESULTS]
        max_opt_std_down = mediator_max_opt[metric][offset][rho]['std_down'][:NUM_RESULTS]
        random_std_up = mediator_random[metric][offset][rho]['std_up'][:NUM_RESULTS]
        random_std_down = mediator_random[metric][offset][rho]['std_down'][:NUM_RESULTS]
        mbie_data_std_up = mbie[metric]['std_up'][:NUM_RESULTS]
        mbie_data_std_down = mbie[metric]['std_down'][:NUM_RESULTS]
        mbie_eb_data_std_up = mbie_eb[metric]['std_up'][:NUM_RESULTS]
        mbie_eb_data_std_down = mbie_eb[metric]['std_down'][:NUM_RESULTS]

        if NUM_ROWS == 1 and NUM_COLS == 1:
            # Plot the sds.
            axs.fill_between(
                index, max_opt_std_down, max_opt_std_up, color = 'red', alpha = 0.1)
            axs.fill_between(
                index, random_std_down, random_std_up, color = 'orange', alpha = 0.1)
            axs.fill_between(
                index, mbie_data_std_down, mbie_data_std_up, color = 'blue', alpha = 0.1)
            axs.fill_between(
                index, mbie_eb_data_std_down, mbie_eb_data_std_up, color = 'cyan', alpha = 0.1)

            # Plot the means.
            sns.lineplot(x = index, y = max_opt, ax = axs[c], color = 'red', alpha = 0.6, label='Max-Opt')
            sns.lineplot(x = index, y = random, ax = axs[c], color = 'orange', alpha = 0.6, label='Random')
            sns.lineplot(x = index, y = mbie_data, ax = axs[c], color = 'blue', alpha = 0.6, label='MBIE')
            sns.lineplot(x = index, y = mbie_eb_data, ax = axs[c], color = 'cyan', alpha = 0.6, label='MBIE-EB')
            subplot_title = r'($\kappa=$' + f'{offset}' + r', $\rho=$' + f'{rho}' + ')'
            axs.set_title(subplot_title, fontsize = font_size)
            axs.get_legend().remove()

            handles, labels = axs.get_legend_handles_labels()


        if NUM_ROWS == 1:
            # Plot the sds.
            axs[c].fill_between(
                index, max_opt_std_down, max_opt_std_up, color = 'red', alpha = 0.1)
            axs[c].fill_between(
                index, random_std_down, random_std_up, color = 'orange', alpha = 0.1)
            axs[c].fill_between(
                index, mbie_data_std_down, mbie_data_std_up, color = 'blue', alpha = 0.1)
            axs[c].fill_between(
                index, mbie_eb_data_std_down, mbie_eb_data_std_up, color = 'cyan', alpha = 0.1)

            # Plot the means.
            sns.lineplot(x = index, y = max_opt, ax = axs[c], color = 'red', alpha = 0.6, label='Max-Opt')
            sns.lineplot(x = index, y = random, ax = axs[c], color = 'orange', alpha = 0.6, label='Random')
            sns.lineplot(x = index, y = mbie_data, ax = axs[c], color = 'blue', alpha = 0.6, label='MBIE')
            sns.lineplot(x = index, y = mbie_eb_data, ax = axs[c], color = 'cyan', alpha = 0.6, label='MBIE-EB')
            subplot_title = r'($\kappa=$' + f'{offset}' + r', $\rho=$' + f'{rho}' + ')'
            axs[c].set_title(subplot_title, fontsize = font_size)
            axs[c].get_legend().remove()

            handles, labels = axs[c].get_legend_handles_labels()

        else:
            # Plot the sds.
            axs[r, c].fill_between(
                index, max_opt_std_down, max_opt_std_up, color = 'red', alpha = 0.1)
            axs[r, c].fill_between(
                index, random_std_down, random_std_up, color = 'orange', alpha = 0.1)
            axs[r, c].fill_between(
                index, mbie_data_std_down, mbie_data_std_up, color = 'blue', alpha = 0.1)
            axs[r, c].fill_between(
                index, mbie_eb_data_std_down, mbie_eb_data_std_up, color = 'cyan', alpha = 0.1)

            # Plot the means.
            sns.lineplot(x = index, y = max_opt, ax = axs[r, c], color = 'red', alpha = 0.6, label='Max-Opt')
            sns.lineplot(x = index, y = random, ax = axs[r, c], color = 'orange', alpha = 0.6, label='Random')
            sns.lineplot(x = index, y = mbie_data, ax = axs[r, c], color = 'blue', alpha = 0.6, label='MBIE')
            sns.lineplot(x = index, y = mbie_eb_data, ax = axs[r, c], color = 'cyan', alpha = 0.6, label='MBIE-EB')
            subplot_title = r'($\kappa=$' + f'{offset}' + r', $\rho=$' + f'{rho}' + ')'
            axs[r, c].set_title(subplot_title, fontsize = font_size)
            axs[r, c].get_legend().remove()

            handles, labels = axs[r, c].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1), ncol=4, fontsize=f'{font_size}')

plt.subplots_adjust(top=0.92, right=0.98)

plt.ticklabel_format(style='scientific', axis='y', scilimits=(0, 3))
fig.tight_layout(pad=0.003)
# fig.suptitle(, size=font_size, y=0.95)
fig.text(0.5, 0.005, 'Number of steps', ha='center', fontsize = font_size)
fig.text(0.003, 0.5, xlabel, va='center', rotation='vertical', fontsize = font_size)
plt.show()
