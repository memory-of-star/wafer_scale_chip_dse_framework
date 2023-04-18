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
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dse4wse/test/dse'))

from evaluator import func_single_fidelity1, func_single_fidelity2, func_multi_fidelity_with_inner_search, func_single_fidelity_with_inner_search, func_mo_single_fidelity_with_inner_search, func_mo_multi_fidelity_with_inner_search
import evaluator
import plot
import copy
import random
from tqdm import tqdm
import api

# random.seed(1)

# build search space
num_of_gpus = [32, 64, 128, 256, 512, 1024, 1536, 1920, 2520, 3072, 6000, 12000, 30000, 60000, 100000, 200000]
def build_search_space():
    root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    design_points = np.load(os.path.join(root_path, 'data/design_points.list'), allow_pickle=True)

    with open(os.path.join(root_path, 'data/design_space.pickle'), 'rb') as f:
        design_space = pickle.load(f)

    for i in range(len(design_space)):
        design_space[i] = list(design_space[i])
        design_space[i].sort()

    return design_points, design_space
 
def fine_tune_model_para(design_point, model_parallel):
    factors_of_number_of_layers = factors(test_model_parameters[choose_model]['number_of_layers'])
    factors_of_attention_heads = factors(test_model_parameters[choose_model]['attention_heads'])
    wafer_num = math.ceil(num_of_gpus[choose_model] * 312 * 1000 / (design_point['core_mac_num'] * design_point['core_array_h'] * design_point['core_array_w'] * design_point['reticle_array_h'] * design_point['reticle_array_w'] * 2))
    model_parallel['num_reticle_per_model_chunk'] = design_point['reticle_array_h'] * design_point['reticle_array_w']

    if model_parallel['micro_batch_size'] * model_parallel['data_parallel_size'] > test_model_parameters[choose_model]['mini_batch_size']:
        model_parallel['micro_batch_size'] = random.randint(1, max(test_model_parameters[choose_model]['mini_batch_size'] // model_parallel['data_parallel_size'], 1))
    if model_parallel['data_parallel_size'] * model_parallel['model_parallel_size'] > wafer_num:
        for i in range(len(factors_of_number_of_layers)):
            if factors_of_number_of_layers[i] > max(wafer_num // model_parallel['data_parallel_size'], 1):
                break 
        model_parallel['model_parallel_size'] = random.choice(factors_of_number_of_layers[:i])
    if model_parallel['tensor_parallel_size'] * model_parallel['num_reticle_per_model_chunk'] > design_point['reticle_array_h'] * design_point['reticle_array_w']:
        # model_parallel['num_reticle_per_model_chunk'] = random.randint(1, max(design_point['reticle_array_h'] * design_point['reticle_array_w'] // model_parallel['tensor_parallel_size'], 1))
        for i in range(len(factors_of_attention_heads)):
            if factors_of_attention_heads[i] > max(design_point['reticle_array_h'] * design_point['reticle_array_w'] // model_parallel['num_reticle_per_model_chunk'], 1):
                break 
        model_parallel['tensor_parallel_size'] = random.choice(factors_of_attention_heads[:i])

    return design_point, model_parallel

def generate_legal_points(num = 100, _choose_model=-1):
    design_points, design_space = build_search_space()
    points_dict = [{k:v for k,v in zip(dimension_name, design_points[i])} for i in range(len(design_points))]
    space = sp.SelfDefinedConditionedSpace()

    global choose_model

    if _choose_model == -1:
        choose_model = random.randint(0, 15)
    else:
        choose_model = _choose_model
    
    variable_lst = []
    for i in range(len(design_space)):
        variable_lst.append(sp.Int(dimension_name[i], design_space[i][0], max(design_space[i][-1], design_space[i][0] + 1), default_value=design_points[0][i]))

    v1 = sp.Int("micro_batch_size", 1, max(test_model_parameters[choose_model]['mini_batch_size'], 2), default_value=1)
    
    wafer_num = math.ceil(num_of_gpus[choose_model] * 312 * 1000 / (4 * 8 * 8 * 6 * 8 * 2))
    v2 = sp.Int("data_parallel_size", 1, max(wafer_num, 2), default_value=1)

    factors_of_number_of_layers = factors(test_model_parameters[choose_model]['number_of_layers'])
    v3 = sp.Ordinal("model_parallel_size", factors_of_number_of_layers, default_value=1)

    factors_of_attention_heads = factors(test_model_parameters[choose_model]['attention_heads'])
    v4 = sp.Ordinal("tensor_parallel_size", factors_of_attention_heads, default_value=1)

    num_reticle_per_pipeline_stage_upper_bound = 55 * 73
    v5 = sp.Int("num_reticle_per_model_chunk", 1, max(num_reticle_per_pipeline_stage_upper_bound, 2), default_value=1)
    
    v6 = sp.Int('weight_streaming', 0, 1, default_value=0)

    space.add_variables(variable_lst)
    space.add_hyperparameter(v1)
    space.add_hyperparameter(v2)
    space.add_hyperparameter(v3)
    space.add_hyperparameter(v4)
    space.add_hyperparameter(v5)
    space.add_hyperparameter(v6)

    def inner_sample_configuration(size):
        if size == 1:
            while True:
                design_point = copy.deepcopy(random.choice(points_dict))
                model_parallel = legal_model_parallel(design_point)
                wafer_num = math.ceil(num_of_gpus[choose_model] * 312 * 1000 / (design_point['core_mac_num'] * design_point['core_array_h'] * design_point['core_array_w'] * design_point['reticle_array_h'] * design_point['reticle_array_w'] * 2))
                model_parallel['num_reticle_per_model_chunk'] = design_point['reticle_array_h'] * design_point['reticle_array_w']

                if model_parallel['micro_batch_size'] * model_parallel['data_parallel_size'] > test_model_parameters[choose_model]['mini_batch_size']:
                    model_parallel['micro_batch_size'] = random.randint(1, max(test_model_parameters[choose_model]['mini_batch_size'] // model_parallel['data_parallel_size'], 1))
                if model_parallel['data_parallel_size'] * model_parallel['model_parallel_size'] > wafer_num:
                    for i in range(len(factors_of_number_of_layers)):
                        if factors_of_number_of_layers[i] > max(wafer_num // model_parallel['data_parallel_size'], 1):
                            break 
                    model_parallel['model_parallel_size'] = random.choice(factors_of_number_of_layers[:i])
                if model_parallel['tensor_parallel_size'] * model_parallel['num_reticle_per_model_chunk'] > design_point['reticle_array_h'] * design_point['reticle_array_w']:
                    # model_parallel['num_reticle_per_model_chunk'] = random.randint(1, max(design_point['reticle_array_h'] * design_point['reticle_array_w'] // model_parallel['tensor_parallel_size'], 1))
                    for i in range(len(factors_of_attention_heads)):
                        if factors_of_attention_heads[i] > max(design_point['reticle_array_h'] * design_point['reticle_array_w'] // model_parallel['num_reticle_per_model_chunk'], 1):
                            break 
                    model_parallel['tensor_parallel_size'] = random.choice(factors_of_attention_heads[:i])
                
                model = copy.deepcopy(test_model_parameters[choose_model])
                model.update(model_parallel)

                wafer_scale_engine = api.create_wafer_scale_engine(**design_point)
                evaluator = api.create_evaluator(True, wafer_scale_engine, **model)
                try:
                    evaluator._find_best_intra_model_chunk_exec_params(inference=False)
                    break
                except:
                    pass


            design_point.update(model_parallel)

            return Configuration(space, design_point)
        else:
            ret = []
            for i in range(size):
                
                while True:
                    design_point = copy.deepcopy(random.choice(points_dict))
                    model_parallel = legal_model_parallel(design_point)
                    wafer_num = math.ceil(num_of_gpus[choose_model] * 312 * 1000 / (design_point['core_mac_num'] * design_point['core_array_h'] * design_point['core_array_w'] * design_point['reticle_array_h'] * design_point['reticle_array_w'] * 2))
                    model_parallel['num_reticle_per_model_chunk'] = design_point['reticle_array_h'] * design_point['reticle_array_w']

                    if model_parallel['micro_batch_size'] * model_parallel['data_parallel_size'] > test_model_parameters[choose_model]['mini_batch_size']:
                        model_parallel['micro_batch_size'] = random.randint(1, max(test_model_parameters[choose_model]['mini_batch_size'] // model_parallel['data_parallel_size'], 1))
                    if model_parallel['data_parallel_size'] * model_parallel['model_parallel_size'] > wafer_num:
                        for i in range(len(factors_of_number_of_layers)):
                            if factors_of_number_of_layers[i] > max(wafer_num // model_parallel['data_parallel_size'], 1):
                                break 
                        model_parallel['model_parallel_size'] = random.choice(factors_of_number_of_layers[:i])
                    if model_parallel['tensor_parallel_size'] * model_parallel['num_reticle_per_model_chunk'] > design_point['reticle_array_h'] * design_point['reticle_array_w']:
                        # model_parallel['num_reticle_per_model_chunk'] = random.randint(1, max(design_point['reticle_array_h'] * design_point['reticle_array_w'] // model_parallel['tensor_parallel_size'], 1))
                        for i in range(len(factors_of_attention_heads)):
                            if factors_of_attention_heads[i] > max(design_point['reticle_array_h'] * design_point['reticle_array_w'] // model_parallel['num_reticle_per_model_chunk'], 1):
                                break 
                        model_parallel['tensor_parallel_size'] = random.choice(factors_of_attention_heads[:i])
                    
                    model = copy.deepcopy(test_model_parameters[choose_model])
                    model.update(model_parallel)

                    wafer_scale_engine = api.create_wafer_scale_engine(**design_point)
                    evaluator = api.create_evaluator(True, wafer_scale_engine, **model)
                    try:
                        evaluator._find_best_intra_model_chunk_exec_params(inference=False)
                        break
                    except:
                        pass
                
                design_point.update(model_parallel)
                ret.append(Configuration(space, design_point))

            return ret
        
    space.set_sample_func(inner_sample_configuration)
    
    ret = []
    for i in tqdm(range(num)):
        design_point = dict(space.sample_configuration())
        model_parameters = copy.deepcopy(test_model_parameters[choose_model])
        model_parameters["micro_batch_size"] = design_point.pop("micro_batch_size")
        model_parameters["data_parallel_size"] = design_point.pop("data_parallel_size")
        model_parameters["model_parallel_size"] = design_point.pop("model_parallel_size")
        model_parameters["tensor_parallel_size"] = design_point.pop("tensor_parallel_size")
        model_parameters["num_reticle_per_model_chunk"] = design_point.pop("num_reticle_per_model_chunk")
        model_parameters['weight_streaming'] = design_point.pop("weight_streaming")
        ret.append((design_point, model_parameters))

    return ret


def KT_evaluator(size=100, choose_model = 0, multi_process = True, threads = 20):
    evaluator.choose_model = choose_model
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
    model_parallel['num_reticle_per_model_chunk'] = random.randint(1, max(num_reticle_per_pipeline_stage_upper_bound, 1))

    model_parallel['weight_streaming'] = random.randint(0, 1)

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
    v5 = sp.Int("num_reticle_per_model_chunk", 1, max(num_reticle_per_pipeline_stage_upper_bound, 2), default_value=1)
    
    v6 = sp.Int('weight_streaming', 0, 1, default_value=0)

    space.add_variables(variable_lst)
    space.add_hyperparameter(v1)
    space.add_hyperparameter(v2)
    space.add_hyperparameter(v3)
    space.add_hyperparameter(v4)
    space.add_hyperparameter(v5)
    space.add_hyperparameter(v6)

    def inner_sample_configuration(size):
        if size == 1:
            while True:
                design_point = copy.deepcopy(random.choice(points_dict))
                model_parallel = legal_model_parallel(design_point)
                wafer_num = math.ceil(num_of_gpus[choose_model] * 312 * 1000 / (design_point['core_mac_num'] * design_point['core_array_h'] * design_point['core_array_w'] * design_point['reticle_array_h'] * design_point['reticle_array_w'] * 2))
                model_parallel['num_reticle_per_model_chunk'] = design_point['reticle_array_h'] * design_point['reticle_array_w']

                if model_parallel['micro_batch_size'] * model_parallel['data_parallel_size'] > test_model_parameters[choose_model]['mini_batch_size']:
                    model_parallel['micro_batch_size'] = random.randint(1, max(test_model_parameters[choose_model]['mini_batch_size'] // model_parallel['data_parallel_size'], 1))
                if model_parallel['data_parallel_size'] * model_parallel['model_parallel_size'] > wafer_num:
                    for i in range(len(factors_of_number_of_layers)):
                        if factors_of_number_of_layers[i] > max(wafer_num // model_parallel['data_parallel_size'], 1):
                            break 
                    model_parallel['model_parallel_size'] = random.choice(factors_of_number_of_layers[:i])
                if model_parallel['tensor_parallel_size'] * model_parallel['num_reticle_per_model_chunk'] > design_point['reticle_array_h'] * design_point['reticle_array_w']:
                    # model_parallel['num_reticle_per_model_chunk'] = random.randint(1, max(design_point['reticle_array_h'] * design_point['reticle_array_w'] // model_parallel['tensor_parallel_size'], 1))
                    for i in range(len(factors_of_attention_heads)):
                        if factors_of_attention_heads[i] > max(design_point['reticle_array_h'] * design_point['reticle_array_w'] // model_parallel['num_reticle_per_model_chunk'], 1):
                            break 
                    model_parallel['tensor_parallel_size'] = random.choice(factors_of_attention_heads[:i])
                
                model = copy.deepcopy(test_model_parameters[choose_model])
                model.update(model_parallel)

                wafer_scale_engine = api.create_wafer_scale_engine(**design_point)
                evaluator = api.create_evaluator(True, wafer_scale_engine, **model)
                try:
                    evaluator._find_best_intra_model_chunk_exec_params(inference=False)
                    break
                except:
                    pass


            design_point.update(model_parallel)

            return Configuration(space, design_point)
        else:
            ret = []
            for i in range(size):
                
                while True:
                    design_point = copy.deepcopy(random.choice(points_dict))
                    model_parallel = legal_model_parallel(design_point)
                    wafer_num = math.ceil(num_of_gpus[choose_model] * 312 * 1000 / (design_point['core_mac_num'] * design_point['core_array_h'] * design_point['core_array_w'] * design_point['reticle_array_h'] * design_point['reticle_array_w'] * 2))
                    model_parallel['num_reticle_per_model_chunk'] = design_point['reticle_array_h'] * design_point['reticle_array_w']

                    if model_parallel['micro_batch_size'] * model_parallel['data_parallel_size'] > test_model_parameters[choose_model]['mini_batch_size']:
                        model_parallel['micro_batch_size'] = random.randint(1, max(test_model_parameters[choose_model]['mini_batch_size'] // model_parallel['data_parallel_size'], 1))
                    if model_parallel['data_parallel_size'] * model_parallel['model_parallel_size'] > wafer_num:
                        for i in range(len(factors_of_number_of_layers)):
                            if factors_of_number_of_layers[i] > max(wafer_num // model_parallel['data_parallel_size'], 1):
                                break 
                        model_parallel['model_parallel_size'] = random.choice(factors_of_number_of_layers[:i])
                    if model_parallel['tensor_parallel_size'] * model_parallel['num_reticle_per_model_chunk'] > design_point['reticle_array_h'] * design_point['reticle_array_w']:
                        # model_parallel['num_reticle_per_model_chunk'] = random.randint(1, max(design_point['reticle_array_h'] * design_point['reticle_array_w'] // model_parallel['tensor_parallel_size'], 1))
                        for i in range(len(factors_of_attention_heads)):
                            if factors_of_attention_heads[i] > max(design_point['reticle_array_h'] * design_point['reticle_array_w'] // model_parallel['num_reticle_per_model_chunk'], 1):
                                break 
                        model_parallel['tensor_parallel_size'] = random.choice(factors_of_attention_heads[:i])
                    
                    model = copy.deepcopy(test_model_parameters[choose_model])
                    model.update(model_parallel)

                    wafer_scale_engine = api.create_wafer_scale_engine(**design_point)
                    evaluator = api.create_evaluator(True, wafer_scale_engine, **model)
                    try:
                        evaluator._find_best_intra_model_chunk_exec_params(inference=False)
                        break
                    except:
                        pass
                
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

