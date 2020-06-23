import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
import scipy.stats




sizes = [1, 2, 4, 6, 8, 10]

def stats_from_run(df, sizes):
    mean_evals = list()
    std_evals = list()
    for i in range(len(sizes)):
        all_runs = df.loc[data['dims'] == sizes[i]]
        mean_evals.append(all_runs['evals'].mean())
        std_evals.append(all_runs['evals'].std())
    return mean_evals, std_evals

mean_evals, std_evals = stats_from_run(data, sizes)
mean_evals_AM, std_evals_AM = stats_from_run(AMal_full, sizes)
mean_evals_AMu, std_evals_AMu = stats_from_run(AMal_u, sizes)

print(std_evals)
print(std_evals_AM)

plt.figure()
plt.plot(sizes, mean_evals, label='GOMEA', color='b')
plt.plot(sizes, mean_evals_AM, label='AMaLGaM-full', color='r')
plt.plot(sizes, mean_evals_AMu, label='AMaLGaM-univariate', color='g')

plt.xscale('log')
plt.yscale('log')
plt.xticks(sizes, sizes)
plt.errorbar(sizes, mean_evals, yerr=std_evals, capsize=2, capthick=2)
plt.errorbar(sizes, mean_evals_AM, yerr=std_evals_AM, capsize=2, capthick=2)
plt.errorbar(sizes, mean_evals_AMu, yerr=std_evals_AMu, capsize=2, capthick=2)

plt.yticks([100, 1000, 10000, 100000, 1000000], [100, 1000, 10000, 100000, 1000000])
plt.grid()
plt.xlabel('d')
plt.ylabel('Number of evaluations')
plt.legend()
plt.show()
