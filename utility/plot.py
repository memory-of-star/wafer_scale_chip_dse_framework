import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List
import copy

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

def get_highest_mean_curve_backup(histories, strategy='multi_fidelity'):
    if strategy == 'multi_fidelity':
        _sum = [None, None]
        for i in range(len(histories)):
            for j in range(2):
                tmp = np.array([k[-1] for k in histories[i][j]])
                for t in range(len(tmp)):
                    tmp[t] = min(min(tmp[:t+1]), 0)
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
            for t in range(len(tmp)):
                tmp[t] = min(min(tmp[:t+1]), 0)
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

def get_highest_mean_curve(histories, strategy='multi_fidelity', iterations=50):
    
    _sum = [] # (run_times, fidelity, max_runs, (design, model, objectives)) -> (fidelity, max_runs, objectives), average on run_times
    _sum2 = []
    if strategy == 'multi_fidelity':
        for run_time in range(len(histories)):
            for point in range(min(len(histories[0][0]), iterations)):
                for i in range(len(histories[run_time][0][point][-1])):
                    if i == 1:
                        histories[run_time][0][point][-1][i] = min(300, histories[run_time][0][point][-1][i])
                    elif i == 0:
                        histories[run_time][0][point][-1][i] = min(0, histories[run_time][0][point][-1][i])
                _sum.append(histories[run_time][0][point][-1])

        for run_time in range(len(histories)):
            for point in range(min(len(histories[0][1]), iterations)):
                for i in range(len(histories[run_time][1][point][-1])):
                    # histories[run_time][1][point][-1][i] = min(300, histories[run_time][1][point][-1][i])
                    if i == 1:
                        histories[run_time][1][point][-1][i] = min(300, histories[run_time][1][point][-1][i])
                    elif i == 0:
                        histories[run_time][1][point][-1][i] = min(0, histories[run_time][1][point][-1][i])
                _sum2.append(histories[run_time][1][point][-1])

        _sum = np.array(_sum)
        _sum = _sum.reshape((len(histories), min(len(histories[0][0]), iterations), len(histories[0][0][0][-1])))

        _sum2 = np.array(_sum2)
        _sum2 = _sum2.reshape((len(histories), min(len(histories[0][1]), iterations), len(histories[0][1][0][-1])))
        
    else:
        for run_time in range(len(histories)):
            for point in range(min(len(histories[0]), iterations)):
                for i in range(len(histories[run_time][point][-1])):
                    # histories[run_time][point][-1][i] = min(300, histories[run_time][point][-1][i])
                    if i == 1:
                        histories[run_time][point][-1][i] = min(300, histories[run_time][point][-1][i])
                    elif i == 0:
                        histories[run_time][point][-1][i] = min(0, histories[run_time][point][-1][i])
                _sum.append(histories[run_time][point][-1])
        _sum = np.array(_sum)
        _sum = _sum.reshape((len(histories), min(len(histories[0]), iterations), len(histories[0][0][-1])))
    
    # here, the shape of _sum is (run_times, max_runs, objectives)
    print(_sum.shape[2])
    if _sum.shape[2] > 1:
        from openbox.utils.multi_objective import NondominatedPartitioning
        hv_mean = []
        pareto_fronts = []
        for i in range(_sum.shape[0]):
            hv = []
            for j in range(_sum.shape[1]):
                partition = NondominatedPartitioning(_sum.shape[2], _sum[i, :j+1, :])
                hv.append(partition.compute_hypervolume([0, 300]))
            hv_mean.append(hv)
            partition = NondominatedPartitioning(_sum.shape[2], _sum[i, :, :])
            pareto_fronts.append(partition.pareto_Y)

        hv_mean = np.array(hv_mean)
        hv = copy.deepcopy(hv_mean)
        hv_max = np.max(hv, axis=0)
        hv_min = np.min(hv, axis=0)
        hv_mean = hv_mean.mean(axis=0)

        return _sum, hv_mean, pareto_fronts, hv_max, hv_min
    else:
        _sum = _sum.reshape((_sum.shape[0], _sum.shape[1]))
        for j in range(_sum.shape[1]):
            _sum[:, j] = np.amin(_sum[:, :j+1], axis=1)
        _sum = _sum.mean(axis=0)
        return _sum