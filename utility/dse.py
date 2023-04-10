from openbox import Optimizer, sp, History
from ConfigSpace import Configuration
from multiprocessing import Process, Queue
import multiprocessing
from typing import List
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from evaluator import func_single_fidelity1, func_single_fidelity2, func_multi_fidelity_with_inner_search
import evaluator
import plot
import copy
import random
from tqdm import tqdm

# build search space
def build_search_space():
    design_points = []
    root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    with open(os.path.join(root_path, 'data/design_points.list'), 'r') as f:
        lines = list(map(lambda x:x.strip()[1:-1].split(','), f.readlines()))
        for i in range(len(lines)):
            lines[i] = list(map(lambda x:float(x.strip()), lines[i]))
            lines[i] = list(map(lambda x:int(x), lines[i]))
            design_points.append(lines[i])

    design_space = [[] for i in range(12)]
    for i in design_points:
        for j in range(12):
            if i[j] not in design_space[j]:
                design_space[j].append(i[j])

    for i in range(12):
        design_space[i].sort()

    design_points.sort()

    return design_points, design_space
 

def process_single_fidelity(points_dict, threshold = 0, queue : Queue = None, design_space=None, design_points=None, **optimizer_kwargs):
    space = sp.ComplexConditionedSpace()
    
    variable_lst = []
    for i in range(len(design_space)):
        variable_lst.append(sp.Int('var{:02}'.format(i), design_space[i][0], max(design_space[i][-1], design_space[i][0] + 1), default_value=design_points[0][i]))
    
    space.add_variables(variable_lst)
    internal_points = list(map(lambda x:Configuration(space, x), points_dict))
    space.internal_points = internal_points
    
    optimizer_kwargs['config_space'] = space
    
    opt = Optimizer(**optimizer_kwargs)
    opt.early_stop_threshold = threshold # early stop mechanism
    history = opt.run()
    
    queue.put(history)

def process_multi_fidelity(points_dict, threshold = 0, queue : Queue = None, design_space=None, design_points=None, strategy='multi_fidelity', **optimizer_kwargs):
    space = sp.MultiFidelityComplexConditionedSpace()
    
    variable_lst = []
    for i in range(len(design_space)):
        variable_lst.append(sp.Int('var{:02}'.format(i), design_space[i][0], max(design_space[i][-1], design_space[i][0] + 1), default_value=design_points[0][i]))
    
    variable_lst.append(sp.Int('fidelity', 1, 2, default_value=1))
    
    space.set_fidelity_dimension()

    space.add_variables(variable_lst)
    internal_points = list(map(lambda x:Configuration(space, x), points_dict))
    space.internal_points = internal_points
    
    optimizer_kwargs['config_space'] = space
    
    opt = Optimizer(**optimizer_kwargs)
    opt.early_stop_threshold = threshold # early stop mechanism

    if strategy == 'multi_fidelity':
        history_lst = opt.mf_run()
        queue.put(history_lst)
    else:
        history_lst = opt.mf_run(fidelity=1)
        queue.put(history_lst)

# get an average curve from a set of repeated experiment histories
# input : List[History]
# return : np.array (average curve)
def get_average_curve(histories: List[History]):
    _sum = None
    for history in histories:
        objectives = np.array([-i[0] for i in history.objectives])
        for i in range(len(objectives)):
            objectives[i] = max(objectives[:i+1])

        if not isinstance(_sum, np.ndarray):
            _sum = objectives
        else:
            if len(_sum) > len(objectives):
                objectives = np.pad(objectives, (0, len(_sum) - len(objectives)), mode='edge')
            else:
                _sum = np.pad(_sum, (0, len(objectives) - len(_sum)), mode='edge')
            _sum += objectives
    _sum /= len(histories)
    return _sum

def get_average_curve_from_queue(queue: Queue):
    histories = []
    while not queue.empty():
        history = queue.get()
        histories.append(history)

    return get_average_curve(histories)

    
def single_fidelity_search(model_num = 1, run_times = 10, max_runs = 20, initial_runs = 6, run_name = 'result', **kwargs):
    design_points, design_space = build_search_space()

    var_names = ['var{:02}'.format(i) for i in range(len(design_space))]
    points_dic = [{k:v for k,v in zip(var_names, design_points[i])} for i in range(len(design_points))]

    manager = multiprocessing.Manager()
    queue_gp = manager.Queue()
    queue_random = manager.Queue()
    pool = []

    for k in range(run_times):
        for i in range(model_num):
            optimizer_kwargs = {
                            'objective_function':func_single_fidelity1,
                            'num_objs':1,
                            'num_constraints':0,
                            'max_runs':max_runs,
                            'surrogate_type':'gp',
                            'acq_optimizer_type':'true_random',
                            'initial_runs':initial_runs,
                            'init_strategy':'random',
                            'time_limit_per_trial':1000,
                            'task_id':'moc',
                            'acq_type':'ei'
                            }
            evaluator.choose_model = i
            p = Process(target=process_single_fidelity, args=(points_dic, -1e11, queue_gp, design_space, design_points), kwargs=optimizer_kwargs)
            p.start()
            pool.append(p)
            

        for i in range(model_num):
            optimizer_kwargs = {
                            'objective_function':func_single_fidelity1,
                            'num_objs':1,
                            'num_constraints':0,
                            'max_runs':max_runs,
                            'surrogate_type':'gp',
                            'acq_optimizer_type':'true_random',
                            'initial_runs':initial_runs,
                            'init_strategy':'random',
                            'time_limit_per_trial':1000,
                            'task_id':'moc',
                            "advisor_type": 'random',
                            'acq_type':'ei'
                            }
            evaluator.choose_model = i
            p = Process(target=process_single_fidelity, args=(points_dic, -1e11, queue_random, design_space, design_points), kwargs=optimizer_kwargs)
            p.start()
            pool.append(p)


    for p in pool:
        p.join()

    root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    histories_random, histories_gp = [], []
    while not queue_random.empty():
        history = queue_random.get()
        histories_random.append(history)

    while not queue_gp.empty():
        history = queue_gp.get()
        histories_gp.append(history)

    with open(os.path.join(root_path, 'result/pickle', run_name+'.pickle'), "wb") as f:
        pickle.dump([histories_random, histories_gp], f)

    curve_random = get_average_curve(histories_random)
    curve_gp = get_average_curve(histories_gp)

    plot.plot_curve(curve_random, curve_gp, path=os.path.join(root_path, 'result/picture', run_name+'_curve.png'))
    plot.plot_hist(np.array([curve_random]), np.array([curve_gp]), path=os.path.join(root_path, 'result/picture', run_name+'_hist.png'))


def multi_fidelity_double_circulation_search(model_num = 1, run_times = 20, max_runs = 100, strategy='multi_fidelity', initial_runs = 6, run_name = 'result', **kwargs):
    design_points, design_space = build_search_space()

    var_names = ['var{:02}'.format(i) for i in range(len(design_space))]
    points_dic = [{k:v for k,v in zip(var_names, design_points[i])} for i in range(len(design_points))]
    points_dic2 = []
    for i in range(len(points_dic)):
        dic = copy.deepcopy(points_dic[i])
        dic['fidelity'] = 1
        points_dic[i]['fidelity'] = 2
        points_dic2.append(dic)
        points_dic2.append(points_dic[i])

    manager = multiprocessing.Manager()
    queue = manager.Queue()
    pool = []

    for k in range(run_times):
        for i in range(model_num):
            if strategy == 'multi_fidelity':
                optimizer_kwargs = {
                                'objective_function':func_multi_fidelity_with_inner_search,
                                'num_objs':1,
                                'num_constraints':0,
                                'max_runs':max_runs,
                                'surrogate_type':'gp',
                                'acq_optimizer_type':'true_random',
                                'initial_runs':initial_runs,
                                'init_strategy':'random',
                                'time_limit_per_trial':1000,
                                'task_id':'moc',
                                'acq_type':'mfei',
                                'advisor_type':'mf_advisor'
                                }
            elif strategy == 'random':
                optimizer_kwargs = {
                                'objective_function':func_multi_fidelity_with_inner_search,
                                'num_objs':1,
                                'num_constraints':0,
                                'max_runs':max_runs,
                                'surrogate_type':'gp',
                                'acq_optimizer_type':'true_random',
                                'initial_runs':initial_runs,
                                'init_strategy':'random',
                                'time_limit_per_trial':1000,
                                'task_id':'moc',
                                'acq_type':'ei',
                                'advisor_type':'mf_random'
                                }
            evaluator.choose_model = i
            p = Process(target=process_multi_fidelity, args=(points_dic2, -1e11, queue, design_space, design_points, strategy), kwargs=optimizer_kwargs)
            p.start()
            pool.append(p)

    for p in pool:
        p.join()

    root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    histories = []
    while not queue.empty():
        history = queue.get()
        histories.append(history)

    with open(os.path.join(root_path, 'result/pickle', run_name+'.pickle'), "wb") as f:
        pickle.dump(histories, f)

    if strategy == "multi_fidelity":
        history1, history2 = [], []
        for history in histories:
            history1.append(history[1])
            history2.append(history[2])

        curve_1 = get_average_curve(history1)
        curve_2 = get_average_curve(history2)

        plot.plot_curve(curve_1, curve_2, label1='fidelity1', label2='fidelity2', x_label = 'iteration', path=os.path.join(root_path, 'result/picture', run_name+'_curve.png'))

def generate_legal_points(num = 100):
    design_points, design_space = build_search_space()

    var_names = ['var{:02}'.format(i) for i in range(len(design_space))]
    points_dic = [{k:v for k,v in zip(var_names, design_points[i])} for i in range(len(design_points))]
    points_dic2 = []
    for i in range(len(points_dic)):
        dic = copy.deepcopy(points_dic[i])
        dic['fidelity'] = 1
        points_dic[i]['fidelity'] = 2
        points_dic2.append(dic)
        points_dic2.append(points_dic[i])

    variable_lst = []
    for i in range(len(design_space)):
        variable_lst.append(sp.Int('var{:02}'.format(i), design_space[i][0], max(design_space[i][-1], design_space[i][0] + 1), default_value=design_points[0][i]))
    
    variable_lst.append(sp.Int('fidelity', 1, 2, default_value=1))
    
    space = sp.MultiFidelityComplexConditionedSpace()
    space.set_fidelity_dimension()
    space.add_variables(variable_lst)
    internal_points = list(map(lambda x:Configuration(space, x), points_dic2))
    space.internal_points = internal_points

    ret = []
    for i in tqdm(range(num)):
        evaluator.choose_model = random.randint(0, 15)
        config = space.sample_configuration()
        design_point, model_para = evaluator.generate(config)
        ret.append((design_point, model_para))
    
    return ret

if __name__ == "__main__":
    multi_fidelity_double_circulation_search(run_times=1, max_runs=20, run_name='try')