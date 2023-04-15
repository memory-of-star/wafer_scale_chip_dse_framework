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

from evaluator import func_single_fidelity1, func_single_fidelity2, func_multi_fidelity_with_inner_search, func_single_fidelity_with_inner_search, func_mo_single_fidelity_with_inner_search, func_mo_multi_fidelity_with_inner_search
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

def process_mfes(points_dict, threshold = 0, queue : Queue = None, design_space=None, design_points=None, **optimizer_kwargs):
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
    histories = opt.mfes_run()
    
    queue.put(histories)
    

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


def dse_(choose_model = 0, run_times = 20, max_runs = 100, multi_objective=False, strategy='multi_fidelity', initial_runs = 6, run_name = 'result', **kwargs):
    design_points, design_space = build_search_space()

    var_names = ['var{:02}'.format(i) for i in range(len(design_space))]
    points_dic = [{k:v for k,v in zip(var_names, design_points[i])} for i in range(len(design_points))]
    points_dic2 = []
    for i in range(len(points_dic)):
        dic = copy.deepcopy(points_dic[i])
        dic['fidelity'] = 1
        dic_ = copy.deepcopy(points_dic[i])
        dic_['fidelity'] = 2
        points_dic2.append(dic)
        points_dic2.append(dic_)

    manager = multiprocessing.Manager()
    queue = manager.Queue()
    pool = []

    for k in range(run_times):
        if strategy == 'multi_fidelity':
            optimizer_kwargs = {
                            'objective_function':None,
                            'fidelity_objective_functions':[evaluator.func_single_fidelity_with_inner_search, evaluator.func_single_fidelity_with_inner_search2],
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
                            'advisor_type':'mfes_advisor'
                            }
            p = Process(target=process_mfes, args=(points_dic, -1e11, queue, design_space, design_points), kwargs=optimizer_kwargs)
        elif strategy == 'random':
            optimizer_kwargs = {
                            'objective_function':func_single_fidelity_with_inner_search,
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
                            'advisor_type':'random'
                            }
            if multi_objective:
                optimizer_kwargs['num_objs'] = 2
                optimizer_kwargs['acq_type'] = 'ehvi'
                optimizer_kwargs['objective_function'] = func_mo_single_fidelity_with_inner_search
            p = Process(target=process_single_fidelity, args=(points_dic, -1e11, queue, design_space, design_points), kwargs=optimizer_kwargs)
        elif strategy == 'single_fidelity':
            optimizer_kwargs = {
                            'objective_function':func_single_fidelity_with_inner_search,
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
                            'advisor_type':'default'
                            }
            if multi_objective:
                optimizer_kwargs['num_objs'] = 2
                optimizer_kwargs['acq_type'] = 'ehvi'
                optimizer_kwargs['objective_function'] = func_mo_single_fidelity_with_inner_search
            p = Process(target=process_single_fidelity, args=(points_dic, -1e11, queue, design_space, design_points), kwargs=optimizer_kwargs)

        evaluator.choose_model = choose_model
        
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

def generate_legal_points(num = 100, choose_model=-1):
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
        if choose_model == -1:
            evaluator.choose_model = random.randint(0, 15)
        else:
            evaluator.choose_model = choose_model
        config = space.sample_configuration()
        design_point, model_para = evaluator.generate(config)
        ret.append((design_point, model_para))
    
    return ret

def KT_evaluator(size=100, choose_model = 0, multi_process = True, threads = 20):
    points_lst = generate_legal_points(size, choose_model)
    if multi_process:
        evaluation_list = evaluator.get_evaluation_list_multi_process(points_lst, threads=threads)
    else:
        evaluation_list = evaluator.get_evaluation_list(points_lst)
    kt, pairs = evaluator.test_KT(evaluation_list)
    return kt, points_lst, evaluation_list, pairs



import math
import random
from test_model_parameters import test_model_parameters
choose_model = 0

def factors(n):
    factors = [] 
    factors.append(1)
    for i in range(2, n + 1):
        if n % i == 0:
            factors.append(i)
    return factors 

def design_vec2design_dic(vec):
    ret = {
        "core_buffer_size": vec[0],
        "core_buffer_bw": vec[1],
        "core_mac_num": vec[2],
        "core_noc_bw": vec[3],
        "core_noc_vc": vec[4],
        "core_noc_buffer_size": vec[5],
        "reticle_bw": vec[6],
        "core_array_h": vec[7],
        "core_array_w": vec[8],
        "wafer_mem_bw": vec[9],
        "reticle_array_h": vec[10],
        "reticle_array_w": vec[11],
    }
    return ret

dimension_name = ["core_buffer_size",
        "core_buffer_bw",
        "core_mac_num",
        "core_noc_bw",
        "core_noc_vc",
        "core_noc_buffer_size",
        "reticle_bw",
        "core_array_h",
        "core_array_w",
        "wafer_mem_bw",
        "reticle_array_h",
        "reticle_array_w"]

def legal_model_parallel(design_point):
    model_parallel = {}
    model_parallel['micro_batch_size'] = random.randint(1, max(test_model_parameters[choose_model]['mini_batch_size'], 1))
    num_of_gpus = [32, 64, 128, 256, 512, 1024, 1536, 1920, 2520, 3072, 6000, 12000, 30000, 60000, 100000, 200000]
    wafer_num = math.ceil(num_of_gpus[choose_model] * 312 * 1000 / (design_point['core_mac_num'] * design_point['core_array_h'] * design_point['core_array_w'] * design_point['reticle_array_h'] * design_point['reticle_array_w'] * 2))
    model_parallel['data_parallel_size'] = random.randint(1, max(wafer_num, 1))

    factors_of_number_of_layers = factors(test_model_parameters[choose_model]['number_of_layers'])
    model_parallel['model_parallel_size'] = random.choice(factors_of_number_of_layers)

    factors_of_attention_heads = factors(test_model_parameters[choose_model]['attention_heads'])
    model_parallel['tensor_parallel_size'] = random.choice(factors_of_attention_heads)

    num_reticle_per_pipeline_stage_upper_bound = wafer_num * design_point['reticle_array_h'] * design_point['reticle_array_w']
    model_parallel['num_reticle_per_pipeline_stage'] = random.randint(1, max(num_reticle_per_pipeline_stage_upper_bound, 1))

    return model_parallel

def process_mfes_full_space(points_dict, threshold = 0, queue : Queue = None, design_space=None, design_points=None, strategy='multi_fidelity', **optimizer_kwargs):
    space = sp.SelfDefinedConditionedSpace()
    
    variable_lst = []
    for i in range(len(design_space)):
        variable_lst.append(sp.Int(dimension_name[i], design_space[i][0], max(design_space[i][-1], design_space[i][0] + 1), default_value=design_points[0][i]))

    v1 = sp.Int("micro_batch_size", 1, max(test_model_parameters[choose_model]['mini_batch_size'], 2), default_value=1)
    num_of_gpus = [32, 64, 128, 256, 512, 1024, 1536, 1920, 2520, 3072, 6000, 12000, 30000, 60000, 100000, 200000]
    wafer_num = math.ceil(num_of_gpus[choose_model] * 312 * 1000 / (4 * 8 * 8 * 6 * 8 * 2))
    v2 = sp.Int("data_parallel_size", 1, max(wafer_num, 2), default_value=1)

    factors_of_number_of_layers = factors(test_model_parameters[choose_model]['number_of_layers'])
    v3 = sp.Ordinal("model_parallel_size", factors_of_number_of_layers, default_value=1)

    factors_of_attention_heads = factors(test_model_parameters[choose_model]['attention_heads'])
    v4 = sp.Ordinal("tensor_parallel_size", factors_of_attention_heads, default_value=1)

    num_reticle_per_pipeline_stage_upper_bound = wafer_num * 55 * 73
    v5 = sp.Int("num_reticle_per_pipeline_stage", 1, max(num_reticle_per_pipeline_stage_upper_bound, 2), default_value=1)
    
    space.add_variables(variable_lst)
    space.add_hyperparameter(v1)
    space.add_hyperparameter(v2)
    space.add_hyperparameter(v3)
    space.add_hyperparameter(v4)
    space.add_hyperparameter(v5)

    def inner_sample_configuration(size):
        if size == 1:
            design_point = random.choice(points_dict)
            model_parallel = legal_model_parallel(design_point)

            wafer_num = math.ceil(num_of_gpus[choose_model] * 312 * 1000 / (design_point['core_mac_num'] * design_point['core_array_h'] * design_point['core_array_w'] * design_point['reticle_array_h'] * design_point['reticle_array_w'] * 2))

            if model_parallel['micro_batch_size'] * model_parallel['data_parallel_size'] > test_model_parameters[choose_model]['mini_batch_size']:
                model_parallel['micro_batch_size'] = random.randint(1, max(test_model_parameters[choose_model]['mini_batch_size'] // model_parallel['data_parallel_size'], 1))
            if math.ceil(design_point['reticle_array_h'] * design_point['reticle_array_w'] / (model_parallel['num_reticle_per_pipeline_stage'] * model_parallel['tensor_parallel_size'])) * wafer_num < model_parallel['model_parallel_size']:
                model_parallel['num_reticle_per_pipeline_stage'] = random.randint(1, max(design_point['reticle_array_h'] * design_point['reticle_array_w'] // (model_parallel['tensor_parallel_size'] * model_parallel['model_parallel_size']), 1))
            if model_parallel['tensor_parallel_size'] * model_parallel['num_reticle_per_pipeline_stage'] > design_point['reticle_array_h'] * design_point['reticle_array_w']:
                model_parallel['num_reticle_per_pipeline_stage'] = random.randint(1, max(design_point['reticle_array_h'] * design_point['reticle_array_w'] // model_parallel['tensor_parallel_size'], 1))
            
            design_point.update(model_parallel)

            return Configuration(space, design_point)
        else:
            ret = []
            for i in range(size):
                design_point = random.choice(points_dict)
                model_parallel = legal_model_parallel(design_point)

                wafer_num = math.ceil(num_of_gpus[choose_model] * 312 * 1000 / (design_point['core_mac_num'] * design_point['core_array_h'] * design_point['core_array_w'] * design_point['reticle_array_h'] * design_point['reticle_array_w'] * 2))

                if model_parallel['micro_batch_size'] * model_parallel['data_parallel_size'] > test_model_parameters[choose_model]['mini_batch_size']:
                    model_parallel['micro_batch_size'] = random.randint(1, max(test_model_parameters[choose_model]['mini_batch_size'] // model_parallel['data_parallel_size'], 1))
                if math.ceil(design_point['reticle_array_h'] * design_point['reticle_array_w'] / (model_parallel['num_reticle_per_pipeline_stage'] * model_parallel['tensor_parallel_size'])) * wafer_num < model_parallel['model_parallel_size']:
                    model_parallel['num_reticle_per_pipeline_stage'] = random.randint(1, max(design_point['reticle_array_h'] * design_point['reticle_array_w'] // (model_parallel['tensor_parallel_size'] * model_parallel['model_parallel_size']), 1))
                if model_parallel['tensor_parallel_size'] * model_parallel['num_reticle_per_pipeline_stage'] > design_point['reticle_array_h'] * design_point['reticle_array_w']:
                    model_parallel['num_reticle_per_pipeline_stage'] = random.randint(1, max(design_point['reticle_array_h'] * design_point['reticle_array_w'] // model_parallel['tensor_parallel_size'], 1))
                
                design_point.update(model_parallel)
                ret.append(Configuration(space, design_point))

            return ret
        
    space.set_sample_func(inner_sample_configuration)
    
    optimizer_kwargs['config_space'] = space
    
    opt = Optimizer(**optimizer_kwargs)
    opt.early_stop_threshold = threshold # early stop mechanism

    if strategy == 'multi_fidelity':
        histories = opt.mfes_run()
    else:
        histories = opt.run()
    
    import parse_pickle
    data = parse_pickle.parse_histories_full_space(histories=[histories], choose_model=choose_model, strategy=strategy)
    queue.put(data[0])



def dse(choose_model_ = 0, run_times = 20, max_runs = 100, multi_objective=False, strategy='multi_fidelity', initial_runs = 6, run_name = 'result', **kwargs):
    design_points, design_space = build_search_space()

    global choose_model 
    choose_model = choose_model_

    points_dic = [{k:v for k,v in zip(dimension_name, design_points[i])} for i in range(len(design_points))]

    manager = multiprocessing.Manager()
    queue = manager.Queue()
    pool = []

    for k in range(run_times):
        if strategy == 'multi_fidelity':
            optimizer_kwargs = {
                            'objective_function':None,
                            'fidelity_objective_functions':[evaluator.func_full_space1, evaluator.func_full_space2],
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
                            'advisor_type':'mfes_advisor'
                            }
            p = Process(target=process_mfes_full_space, args=(points_dic, -1e11, queue, design_space, design_points, strategy), kwargs=optimizer_kwargs)
        elif strategy == 'random':
            optimizer_kwargs = {
                            'objective_function':evaluator.func_full_space1,
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
                            'advisor_type':'random'
                            }
            if multi_objective:
                optimizer_kwargs['num_objs'] = 2
                optimizer_kwargs['acq_type'] = 'ehvi'
                optimizer_kwargs['objective_function'] = func_mo_single_fidelity_with_inner_search
            p = Process(target=process_mfes_full_space, args=(points_dic, -1e11, queue, design_space, design_points, strategy), kwargs=optimizer_kwargs)
        elif strategy == 'single_fidelity':
            optimizer_kwargs = {
                            'objective_function':evaluator.func_full_space1,
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
                            'advisor_type':'default'
                            }
            if multi_objective:
                optimizer_kwargs['num_objs'] = 2
                optimizer_kwargs['acq_type'] = 'ehvi'
                optimizer_kwargs['objective_function'] = func_mo_single_fidelity_with_inner_search
            p = Process(target=process_mfes_full_space, args=(points_dic, -1e11, queue, design_space, design_points, strategy), kwargs=optimizer_kwargs)

        evaluator.choose_model = choose_model
        
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

