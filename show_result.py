from utility import dse, parse_pickle, plot
import pickle
import matplotlib.pyplot as plt
import numpy as np

curves_highest_mean_single_fidelity = []
curves_highest_mean_random = []
curves_highest_mean_multi_fidelity = []

for cm in range(1):
    with open('./result/pickle/random_mean_cm{}_rt10_mr200.pickle'.format(cm), 'rb') as f:
        histories_random = pickle.load(f)

    with open('./result/pickle/multi_fidelity_mean_final_cm{}_rt10_mr300.pickle'.format(cm), 'rb') as f:
        histories_multi_fidelity = pickle.load(f)

    with open('./result/pickle/single_fidelity_mean_final_cm{}_rt10_mr200.pickle'.format(cm), 'rb') as f:
        histories_single_fidelity = pickle.load(f)

    points_multi_fidelity, curve_highest_mean_multi_fidelity, pareto_front_multi_fidelity = plot.get_highest_mean_curve(histories_multi_fidelity, strategy='multi_fidelity', iterations=200)
    points_single_fidelity, curve_highest_mean_single_fidelity, pareto_front_single_fidelity = plot.get_highest_mean_curve(histories_single_fidelity, strategy='single_fidelity', iterations=200)
    points_random, curve_highest_mean_random, pareto_front_random = plot.get_highest_mean_curve(histories_random, strategy='random', iterations=200)

    curves_highest_mean_single_fidelity.append(curve_highest_mean_single_fidelity)
    curves_highest_mean_random.append(curve_highest_mean_random)
    curves_highest_mean_multi_fidelity.append(curve_highest_mean_multi_fidelity)

import seaborn as sns

sns.set_style('darkgrid') # darkgrid, white grid, dark, white and ticks\n",
plt.rc('axes', titlesize=20)     # fontsize of the axes title\n",
plt.rc('axes', labelsize=14)    # fontsize of the x and y labels\n",
plt.rc('xtick', labelsize=13)    # fontsize of the tick labels\n",
plt.rc('ytick', labelsize=13)    # fontsize of the tick labels\n",
plt.rc('legend', fontsize=13)    # legend fontsize\n",
plt.rc('font', size=13)          # controls default text sizes\n",
sns.color_palette('deep')
plt.figure(figsize=(10,6), tight_layout=True)

for cm in range(1):
    plt.plot(np.arange(len(curves_highest_mean_multi_fidelity[cm][6:])), curves_highest_mean_multi_fidelity[cm][6:], label='multi_fidelity')
    plt.plot(np.arange(len(curves_highest_mean_single_fidelity[cm][6:])), curves_highest_mean_single_fidelity[cm][6:], label='single_fidelity')
    plt.plot(np.arange(len(curves_highest_mean_random[cm][6:])), curves_highest_mean_random[cm][6:], label='random')

    plt.xlabel('Iteration')
    plt.ylabel('Hypervolume')
    plt.title('16 models'.format(cm))

    plt.legend()
    plt.savefig('16_models_mean_of_gpt3')
    plt.show()