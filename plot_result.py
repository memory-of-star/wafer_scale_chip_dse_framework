import matplotlib.pyplot as plt


values1_gp, values1_random, values2_gp, values2_random = [], [], [], []
# for i in range(10):
#     with open('result/log_fidelity2_model{}_gp'.format(i), 'r') as f:
#         lines = f.readlines()
#         values2_gp.append(list(map(lambda x: -float(x), lines[15::16])))

# for i in range(10):
#     with open('result/log_fidelity2_model{}_random'.format(i), 'r') as f:
#         lines = f.readlines()
#         values2_random.append(list(map(lambda x: -float(x), lines[15::16])))


for i in range(1):
    for k in range(10):
        with open('result/log_fidelity1_model{}_gp_time{}'.format(i, k), 'r') as f:
            lines = f.readlines()
            values1_gp[k].append(list(map(lambda x: -float(x), lines[15::16])))

for i in range(1):
    with open('result/log_fidelity1_model{}_random'.format(i), 'r') as f:
        lines = f.readlines()
        values1_random.append(list(map(lambda x: -float(x), lines[15::16])))

for i in range(len(values2_gp)):
    for j in range(len(values2_gp[i])):
        values2_gp[i][j] = max(values2_gp[i][:j+1])

for i in range(len(values2_random)):
    for j in range(len(values2_random[i])):
        values2_random[i][j] = max(values2_random[i][:j+1])

for i in range(len(values1_gp)):
    for j in range(len(values1_gp[i])):
        values1_gp[i][j] = max(values1_gp[i][:j+1])

for i in range(len(values1_random)):
    for j in range(len(values1_random[i])):
        values1_random[i][j] = max(values1_random[i][:j+1])

import numpy as np

values2_gp = np.array(values2_gp)
values2_random = np.array(values2_random)
values1_gp = np.array(values1_gp)
values1_random = np.array(values1_random)

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set_style('darkgrid') # darkgrid, white grid, dark, white and ticks
plt.rc('axes', titlesize=20)     # fontsize of the axes title
plt.rc('axes', labelsize=14)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=13)    # fontsize of the tick labels
plt.rc('ytick', labelsize=13)    # fontsize of the tick labels
plt.rc('legend', fontsize=13)    # legend fontsize
plt.rc('font', size=13)          # controls default text sizes

x = np.arange(0,len(values1_random)*2,2)
width=0.5
x1 = x-width/2
x2 = x+width/2

sns.color_palette('deep')

plt.figure(figsize=(10,6), tight_layout=True)
# plt.bar(x1, values1_random[:, -240], width=0.5, label='Baseline', color=sns.color_palette('Set2')[-1])
# plt.bar(x2, values1_gp[:, -240], width=0.5, label='Single Fidelity', color=sns.color_palette('Set2')[-2])

# plt.plot(np.arange(1,len(values1_random[0])+1), values1_random[0], linewidth=0.5, label='Baseline', color=sns.color_palette('Set2')[-1])
# plt.plot(np.arange(1,len(values1_gp[0])+1), values1_gp[0], linewidth=0.5, label='Single Fidelity', color=sns.color_palette('Set2')[-2])

plt.plot(np.arange(1,21), values1_random[0][:20], linewidth=0.5, label='Baseline', color=sns.color_palette('Set2')[-1])
plt.plot(np.arange(1,21), values1_gp[0][:20], linewidth=0.5, label='Single Fidelity', color=sns.color_palette('Set2')[-2])

plt.xticks(x,np.arange(1,len(values1_random)+1),fontsize=10)
plt.tick_params(axis='x',length=0)

plt.yscale("log")
plt.xlabel('Model Parameter Combination')
plt.ylabel('Throughput')

plt.title('Wafer Scale Chip DSE')
plt.legend(title='Different Strategy', title_fontsize = 13)

plt.savefig('Wafer Scale Chip DSE fidelity1.png')