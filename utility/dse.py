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
from evaluator import func_single_fidelity1, func_single_fidelity2
import evaluator
import plot

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
 

def process(points_dict, threshold = 0, queue : Queue = None, design_space=None, design_points=None, **optimizer_kwargs):
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
            p = Process(target=process, args=(points_dic, -1e11, queue_gp, design_space, design_points), kwargs=optimizer_kwargs)
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
            p = Process(target=process, args=(points_dic, -1e11, queue_random, design_space, design_points), kwargs=optimizer_kwargs)
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


if __name__ == "__main__":
    single_fidelity_search(max_runs=20, run_name='test')