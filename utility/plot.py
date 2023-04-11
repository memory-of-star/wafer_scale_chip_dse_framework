import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List

def plot_curve(data_1:np.ndarray, data_2:np.ndarray, label1 = 'Baseline', label2 = 'Single Fidelity', y_log = True, x_label = 'Model Parameter Combination', y_label = 'Throughput', title = 'Wafer Scale Chip DSE Curve', path='result.png'):
    sns.set_style('darkgrid') # darkgrid, white grid, dark, white and ticks
    plt.rc('axes', titlesize=20)     # fontsize of the axes title
    plt.rc('axes', labelsize=14)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=13)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=13)    # fontsize of the tick labels
    plt.rc('legend', fontsize=13)    # legend fontsize
    plt.rc('font', size=13)          # controls default text sizes
    sns.color_palette('deep')
    plt.figure(figsize=(10,6), tight_layout=True)

    plt.plot(np.arange(1,len(data_1) + 1), data_1, linewidth=0.5, label=label1, color=sns.color_palette('Set2')[-1])
    plt.plot(np.arange(1,len(data_2) + 1), data_2, linewidth=0.5, label=label2, color=sns.color_palette('Set2')[-2])
    
    if y_log:
        plt.yscale("log")

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend(title='Different Strategy', title_fontsize = 13)

    plt.savefig(path)

def plot_hist(data_random:List[np.ndarray], data_gp:List[np.ndarray], y_log = True, x_label = 'Model Parameter Combination', y_label = 'Throughput', title = 'Wafer Scale Chip DSE Hist', path='result.png'):
    sns.set_style('darkgrid') # darkgrid, white grid, dark, white and ticks
    plt.rc('axes', titlesize=20)     # fontsize of the axes title
    plt.rc('axes', labelsize=14)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=13)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=13)    # fontsize of the tick labels
    plt.rc('legend', fontsize=13)    # legend fontsize
    plt.rc('font', size=13)          # controls default text sizes
    sns.color_palette('deep')
    plt.figure(figsize=(10,6), tight_layout=True)

    x = np.arange(0,len(data_random)*2,2)
    width=0.5
    x1 = x-width/2
    x2 = x+width/2

    plt.bar(x1, data_random[:, -1], width=0.5, label='Baseline', color=sns.color_palette('Set2')[-1])
    plt.bar(x2, data_gp[:, -1], width=0.5, label='Single Fidelity', color=sns.color_palette('Set2')[-2])

    plt.xticks(x, np.arange(1, len(data_random)+1), fontsize=10)
    plt.tick_params(axis='x', length=0)
    
    if y_log:
        plt.yscale("log")

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend(title='Different Strategy', title_fontsize = 13)

    plt.savefig(path)

def get_curve(histories, strategy='multi_fidelity'):
    if strategy == 'multi_fidelity':
        _sum = [None, None]
        for i in range(len(histories)):
            for j in range(2):
                tmp = np.array([k[-1] for k in histories[i][j]])
                if not isinstance(_sum[j], np.ndarray):
                    _sum[j] = tmp
                else:
                    if len(_sum[j]) > len(tmp):
                        tmp = np.pad(tmp, (0, len(_sum[j]) - len(tmp)), mode='edge')
                    else:
                        _sum[j] = np.pad(_sum[j], (0, len(tmp) - len(_sum[j])), mode='edge')
                    _sum[j] += tmp
        for j in range(2):
            _sum[j] /= len(histories)
    elif strategy == 'random' or strategy == 'single_fidelity':
        _sum = None
        for i in range(len(histories)):
            tmp = np.array([k[-1] for k in histories[i]])
            if not isinstance(_sum, np.ndarray):
                _sum = tmp
            else:
                if len(_sum) > len(tmp):
                    tmp = np.pad(tmp, (0, len(_sum) - len(tmp)), mode='edge')
                else:
                    _sum = np.pad(_sum, (0, len(tmp) - len(_sum)), mode='edge')
                _sum += tmp
        _sum /= len(histories)
    return _sum