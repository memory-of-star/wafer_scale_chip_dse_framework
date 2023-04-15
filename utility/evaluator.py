import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dse4wse/test/dse'))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from openbox import Optimizer, sp, History
from ConfigSpace import ConfigurationSpace, Configuration
import api2
import api1
from multiprocessing import Process, Queue
import multiprocessing
from typing import Iterable, List, Union, Tuple, Optional
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from test_model_parameters import test_model_parameters
import random
import math
import copy
from tqdm import tqdm

choose_model = 0
inner_max_runs = 20

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

def factors(n):
    factors = [] 
    factors.append(1)
    for i in range(2, n + 1):
        if n % i == 0:
            factors.append(i)
    return factors 

def func_single_fidelity1(config: sp.Configuration):
    try:
        design_point = design_vec2design_dic(tuple(dict(config).values()))
        prediction = api1.evaluate_design_point(design_point=design_point, model_parameters=test_model_parameters[choose_model], metric='throughput')
    except:
        prediction = -1e10
    result = dict()
    result['objs'] = [-prediction]
    return result

def func_single_fidelity2(config: sp.Configuration):
    try:
        design_point = design_vec2design_dic(tuple(dict(config).values()))
        prediction = api2.evaluate_design_point(design_point=design_point, model_parameters=test_model_parameters[choose_model], metric='throughput')
    except:
        prediction = -1e10
    result = dict()
    result['objs'] = [-prediction]
    return result

def func_full_space1(config: sp.Configuration):
    try:
        design_point = dict(config)
        model_parameters = copy.deepcopy(test_model_parameters[choose_model])
        model_parameters["micro_batch_size"] = design_point.pop("micro_batch_size")
        model_parameters["data_parallel_size"] = design_point.pop("data_parallel_size")
        model_parameters["model_parallel_size"] = design_point.pop("model_parallel_size")
        model_parameters["tensor_parallel_size"] = design_point.pop("tensor_parallel_size")
        model_parameters["num_reticle_per_pipeline_stage"] = design_point.pop("num_reticle_per_pipeline_stage")

        prediction = api1.evaluate_design_point(design_point=design_point, model_parameters=model_parameters, metric='throughput')
    except:
        prediction = -1e10
    result = dict()
    result['objs'] = [-prediction]
    return result

def func_full_space2(config: sp.Configuration):
    try:
        design_point = dict(config)
        model_parameters = copy.deepcopy(test_model_parameters[choose_model])
        model_parameters["micro_batch_size"] = design_point.pop("micro_batch_size")
        model_parameters["data_parallel_size"] = design_point.pop("data_parallel_size")
        model_parameters["model_parallel_size"] = design_point.pop("model_parallel_size")
        model_parameters["tensor_parallel_size"] = design_point.pop("tensor_parallel_size")
        model_parameters["num_reticle_per_pipeline_stage"] = design_point.pop("num_reticle_per_pipeline_stage")

        prediction = api2.evaluate_design_point(design_point=design_point, model_parameters=model_parameters, metric='throughput')
    except:
        prediction = -1e10
    result = dict()
    result['objs'] = [-prediction]
    return result



def func_multi_fidelity_with_inner_search(config: sp.Configuration):
    try:
        dic = dict(config)
        lst = tuple(dict(config).values())
        design_point = design_vec2design_dic(lst[1:])

        _space = sp.SelfDefinedConditionedSpace()
        v1 = sp.Int("micro_batch_size", 1, max(test_model_parameters[choose_model]['mini_batch_size'], 2), default_value=1)
        num_of_gpus = [32, 64, 128, 256, 512, 1024, 1536, 1920, 2520, 3072, 6000, 12000, 30000, 60000, 100000, 200000]
        wafer_num = math.ceil(num_of_gpus[choose_model] * 312 * 1000 / (design_point['core_mac_num'] * design_point['core_array_h'] * design_point['core_array_w'] * design_point['reticle_array_h'] * design_point['reticle_array_w'] * 2))
        v2 = sp.Int("data_parallel_size", 1, max(wafer_num, 2), default_value=1)

        factors_of_number_of_layers = factors(test_model_parameters[choose_model]['number_of_layers'])
        v3 = sp.Ordinal("model_parallel_size", factors_of_number_of_layers, default_value=1)

        factors_of_attention_heads = factors(test_model_parameters[choose_model]['attention_heads'])
        v4 = sp.Ordinal("tensor_parallel_size", factors_of_attention_heads, default_value=1)

        num_reticle_per_pipeline_stage_upper_bound = wafer_num * design_point['reticle_array_h'] * design_point['reticle_array_w']
        v5 = sp.Int("num_reticle_per_pipeline_stage", 1, max(num_reticle_per_pipeline_stage_upper_bound, 2), default_value=1)

        _space.add_hyperparameter(v1)
        _space.add_hyperparameter(v2)
        _space.add_hyperparameter(v3)
        _space.add_hyperparameter(v4)
        _space.add_hyperparameter(v5)
        
        def inner_sample_configuration(size):
            if size == 1:
                cfg = super(sp.SelfDefinedConditionedSpace, _space).sample_configuration(size)
                if cfg['data_parallel_size'] > wafer_num:
                    cfg['data_parallel_size'] = random.randint(1, max(wafer_num, 1))
                if cfg['micro_batch_size'] * cfg['data_parallel_size'] > test_model_parameters[choose_model]['mini_batch_size']:
                    cfg['micro_batch_size'] = random.randint(1, max(test_model_parameters[choose_model]['mini_batch_size'] // cfg['data_parallel_size'], 1))
                if math.ceil(design_point['reticle_array_h'] * design_point['reticle_array_w'] / (cfg['num_reticle_per_pipeline_stage'] * cfg['tensor_parallel_size'])) * wafer_num < cfg['model_parallel_size']:
                    cfg['num_reticle_per_pipeline_stage'] = random.randint(1, max(design_point['reticle_array_h'] * design_point['reticle_array_w'] // (cfg['tensor_parallel_size'] * cfg['model_parallel_size']), 1))
                if cfg['tensor_parallel_size'] * cfg['num_reticle_per_pipeline_stage'] > design_point['reticle_array_h'] * design_point['reticle_array_w']:
                    cfg['num_reticle_per_pipeline_stage'] = random.randint(1, max(design_point['reticle_array_h'] * design_point['reticle_array_w'] // cfg['tensor_parallel_size'], 1))
                return cfg
            else:
                cfgs = super(sp.SelfDefinedConditionedSpace, _space).sample_configuration(size)
                for i, cfg in enumerate(cfgs):
                    if cfg['data_parallel_size'] > wafer_num:
                        cfg['data_parallel_size'] = random.randint(1, max(wafer_num, 1))
                    if cfg['micro_batch_size'] * cfg['data_parallel_size'] > test_model_parameters[choose_model]['mini_batch_size']:
                        cfg['micro_batch_size'] = random.randint(1, max(test_model_parameters[choose_model]['mini_batch_size'] // cfg['data_parallel_size'], 1))
                    if math.ceil(design_point['reticle_array_h'] * design_point['reticle_array_w'] / (cfg['num_reticle_per_pipeline_stage'] * cfg['tensor_parallel_size'])) * wafer_num < cfg['model_parallel_size']:
                        cfg['num_reticle_per_pipeline_stage'] = random.randint(1, max(design_point['reticle_array_h'] * design_point['reticle_array_w'] // (cfg['tensor_parallel_size'] * cfg['model_parallel_size']), 1))
                    if cfg['tensor_parallel_size'] * cfg['num_reticle_per_pipeline_stage'] > design_point['reticle_array_h'] * design_point['reticle_array_w']:
                        cfg['num_reticle_per_pipeline_stage'] = random.randint(1, max(design_point['reticle_array_h'] * design_point['reticle_array_w'] // cfg['tensor_parallel_size'], 1))
                    
                    cfgs[i] = cfg
                return cfgs
        
        _space.set_sample_func(inner_sample_configuration)


        def inner_func(_config: sp.Configuration):
            try:
                _dic = dict(_config)
                temp_dic = copy.deepcopy(test_model_parameters[choose_model])
                temp_dic.update(_dic)
                if lst[0] == 1:
                    _prediction = api1.evaluate_design_point(design_point=design_point, model_parameters=temp_dic, metric='throughput')
                elif lst[0] == 2:
                    _prediction = api2.evaluate_design_point(design_point=design_point, model_parameters=temp_dic, metric='throughput')
                else:
                    raise ValueError
            except:
                print("inner func error!")
                _prediction = -1e10
            result = dict()
            result['objs'] = [-_prediction]
            return result

        _optimizer_kwargs = {
                    'config_space':_space,
                    'objective_function':inner_func,
                    'num_objs':1,
                    'num_constraints':0,
                    'max_runs':inner_max_runs,
                    'surrogate_type':'gp',
                    'acq_optimizer_type':'true_random',
                    'initial_runs':6,
                    'init_strategy':'random',
                    'time_limit_per_trial':1000,
                    'task_id':'moc',
                    'acq_type':'ei',
                    # 'advisor_type':'mf_advisor'
                    'advisor_type':'random'
                    }
        
        _opt = Optimizer(**_optimizer_kwargs)
        _opt.early_stop_threshold = -1e11
        _history = _opt.run()

        prediction = _history.get_incumbents()[0].objectives[0]

    except Exception as e:
        print("outer func error!: ", e)
        prediction = 1e10
        result = dict()
        result['objs'] = [prediction]
        return result
    
    result = dict()
    result['objs'] = [prediction]
    result['config'] = [dict(_history.get_incumbents()[0].config)]

    print('result_config: ', _history.get_incumbents()[0].config)
    return result

def func_single_fidelity_with_inner_search(config: sp.Configuration):
    try:
        dic = dict(config)
        lst = tuple(dict(config).values())
        design_point = design_vec2design_dic(lst)

        _space = sp.SelfDefinedConditionedSpace()
        v1 = sp.Int("micro_batch_size", 1, max(test_model_parameters[choose_model]['mini_batch_size'], 2), default_value=1)
        num_of_gpus = [32, 64, 128, 256, 512, 1024, 1536, 1920, 2520, 3072, 6000, 12000, 30000, 60000, 100000, 200000]
        wafer_num = math.ceil(num_of_gpus[choose_model] * 312 * 1000 / (design_point['core_mac_num'] * design_point['core_array_h'] * design_point['core_array_w'] * design_point['reticle_array_h'] * design_point['reticle_array_w'] * 2))
        v2 = sp.Int("data_parallel_size", 1, max(wafer_num, 2), default_value=1)

        factors_of_number_of_layers = factors(test_model_parameters[choose_model]['number_of_layers'])
        v3 = sp.Ordinal("model_parallel_size", factors_of_number_of_layers, default_value=1)

        factors_of_attention_heads = factors(test_model_parameters[choose_model]['attention_heads'])
        v4 = sp.Ordinal("tensor_parallel_size", factors_of_attention_heads, default_value=1)

        num_reticle_per_pipeline_stage_upper_bound = wafer_num * design_point['reticle_array_h'] * design_point['reticle_array_w']
        v5 = sp.Int("num_reticle_per_pipeline_stage", 1, max(num_reticle_per_pipeline_stage_upper_bound, 2), default_value=1)

        _space.add_hyperparameter(v1)
        _space.add_hyperparameter(v2)
        _space.add_hyperparameter(v3)
        _space.add_hyperparameter(v4)
        _space.add_hyperparameter(v5)
        
        def inner_sample_configuration(size):
            if size == 1:
                cfg = super(sp.SelfDefinedConditionedSpace, _space).sample_configuration(size)
                if cfg['data_parallel_size'] > wafer_num:
                    cfg['data_parallel_size'] = random.randint(1, max(wafer_num, 1))
                if cfg['micro_batch_size'] * cfg['data_parallel_size'] > test_model_parameters[choose_model]['mini_batch_size']:
                    cfg['micro_batch_size'] = random.randint(1, max(test_model_parameters[choose_model]['mini_batch_size'] // cfg['data_parallel_size'], 1))
                if math.ceil(design_point['reticle_array_h'] * design_point['reticle_array_w'] / (cfg['num_reticle_per_pipeline_stage'] * cfg['tensor_parallel_size'])) * wafer_num < cfg['model_parallel_size']:
                    cfg['num_reticle_per_pipeline_stage'] = random.randint(1, max(design_point['reticle_array_h'] * design_point['reticle_array_w'] // (cfg['tensor_parallel_size'] * cfg['model_parallel_size']), 1))
                if cfg['tensor_parallel_size'] * cfg['num_reticle_per_pipeline_stage'] > design_point['reticle_array_h'] * design_point['reticle_array_w']:
                    cfg['num_reticle_per_pipeline_stage'] = random.randint(1, max(design_point['reticle_array_h'] * design_point['reticle_array_w'] // cfg['tensor_parallel_size'], 1))
                return cfg
            else:
                cfgs = super(sp.SelfDefinedConditionedSpace, _space).sample_configuration(size)
                for i, cfg in enumerate(cfgs):
                    if cfg['data_parallel_size'] > wafer_num:
                        cfg['data_parallel_size'] = random.randint(1, max(wafer_num, 1))
                    if cfg['micro_batch_size'] * cfg['data_parallel_size'] > test_model_parameters[choose_model]['mini_batch_size']:
                        cfg['micro_batch_size'] = random.randint(1, max(test_model_parameters[choose_model]['mini_batch_size'] // cfg['data_parallel_size'], 1))
                    if math.ceil(design_point['reticle_array_h'] * design_point['reticle_array_w'] / (cfg['num_reticle_per_pipeline_stage'] * cfg['tensor_parallel_size'])) * wafer_num < cfg['model_parallel_size']:
                        cfg['num_reticle_per_pipeline_stage'] = random.randint(1, max(design_point['reticle_array_h'] * design_point['reticle_array_w'] // (cfg['tensor_parallel_size'] * cfg['model_parallel_size']), 1))
                    if cfg['tensor_parallel_size'] * cfg['num_reticle_per_pipeline_stage'] > design_point['reticle_array_h'] * design_point['reticle_array_w']:
                        cfg['num_reticle_per_pipeline_stage'] = random.randint(1, max(design_point['reticle_array_h'] * design_point['reticle_array_w'] // cfg['tensor_parallel_size'], 1))
                    
                    cfgs[i] = cfg
                return cfgs
        
        _space.set_sample_func(inner_sample_configuration)


        def inner_func(_config: sp.Configuration):
            try:
                _dic = dict(_config)
                temp_dic = copy.deepcopy(test_model_parameters[choose_model])
                temp_dic.update(_dic)
                _prediction = api1.evaluate_design_point(design_point=design_point, model_parameters=temp_dic, metric='throughput')
            except:
                print("inner func error!")
                _prediction = -1e10
            result = dict()
            result['objs'] = [-_prediction]
            return result

        _optimizer_kwargs = {
                    'config_space':_space,
                    'objective_function':inner_func,
                    'num_objs':1,
                    'num_constraints':0,
                    'max_runs':inner_max_runs,
                    'surrogate_type':'gp',
                    'acq_optimizer_type':'true_random',
                    'initial_runs':6,
                    'init_strategy':'random',
                    'time_limit_per_trial':1000,
                    'task_id':'moc',
                    'acq_type':'ei',
                    # 'advisor_type':'mf_advisor'
                    'advisor_type':'random'
                    }
        
        _opt = Optimizer(**_optimizer_kwargs)
        _opt.early_stop_threshold = -1e11
        _history = _opt.run()

        prediction = _history.get_incumbents()[0].objectives[0]

    except Exception as e:
        print("outer func error!: ", e)
        prediction = 1e10
        result = dict()
        result['objs'] = [prediction]
        return result
    
    result = dict()
    result['objs'] = [prediction]
    result['config'] = [dict(_history.get_incumbents()[0].config)]

    print('result_config: ', _history.get_incumbents()[0].config)
    return result

def func_single_fidelity_with_inner_search2(config: sp.Configuration):
    try:
        dic = dict(config)
        lst = tuple(dict(config).values())
        design_point = design_vec2design_dic(lst)

        _space = sp.SelfDefinedConditionedSpace()
        v1 = sp.Int("micro_batch_size", 1, max(test_model_parameters[choose_model]['mini_batch_size'], 2), default_value=1)
        num_of_gpus = [32, 64, 128, 256, 512, 1024, 1536, 1920, 2520, 3072, 6000, 12000, 30000, 60000, 100000, 200000]
        wafer_num = math.ceil(num_of_gpus[choose_model] * 312 * 1000 / (design_point['core_mac_num'] * design_point['core_array_h'] * design_point['core_array_w'] * design_point['reticle_array_h'] * design_point['reticle_array_w'] * 2))
        v2 = sp.Int("data_parallel_size", 1, max(wafer_num, 2), default_value=1)

        factors_of_number_of_layers = factors(test_model_parameters[choose_model]['number_of_layers'])
        v3 = sp.Ordinal("model_parallel_size", factors_of_number_of_layers, default_value=1)

        factors_of_attention_heads = factors(test_model_parameters[choose_model]['attention_heads'])
        v4 = sp.Ordinal("tensor_parallel_size", factors_of_attention_heads, default_value=1)

        num_reticle_per_pipeline_stage_upper_bound = wafer_num * design_point['reticle_array_h'] * design_point['reticle_array_w']
        v5 = sp.Int("num_reticle_per_pipeline_stage", 1, max(num_reticle_per_pipeline_stage_upper_bound, 2), default_value=1)

        _space.add_hyperparameter(v1)
        _space.add_hyperparameter(v2)
        _space.add_hyperparameter(v3)
        _space.add_hyperparameter(v4)
        _space.add_hyperparameter(v5)
        
        def inner_sample_configuration(size):
            if size == 1:
                cfg = super(sp.SelfDefinedConditionedSpace, _space).sample_configuration(size)
                if cfg['data_parallel_size'] > wafer_num:
                    cfg['data_parallel_size'] = random.randint(1, max(wafer_num, 1))
                if cfg['micro_batch_size'] * cfg['data_parallel_size'] > test_model_parameters[choose_model]['mini_batch_size']:
                    cfg['micro_batch_size'] = random.randint(1, max(test_model_parameters[choose_model]['mini_batch_size'] // cfg['data_parallel_size'], 1))
                if math.ceil(design_point['reticle_array_h'] * design_point['reticle_array_w'] / (cfg['num_reticle_per_pipeline_stage'] * cfg['tensor_parallel_size'])) * wafer_num < cfg['model_parallel_size']:
                    cfg['num_reticle_per_pipeline_stage'] = random.randint(1, max(design_point['reticle_array_h'] * design_point['reticle_array_w'] // (cfg['tensor_parallel_size'] * cfg['model_parallel_size']), 1))
                if cfg['tensor_parallel_size'] * cfg['num_reticle_per_pipeline_stage'] > design_point['reticle_array_h'] * design_point['reticle_array_w']:
                    cfg['num_reticle_per_pipeline_stage'] = random.randint(1, max(design_point['reticle_array_h'] * design_point['reticle_array_w'] // cfg['tensor_parallel_size'], 1))
                return cfg
            else:
                cfgs = super(sp.SelfDefinedConditionedSpace, _space).sample_configuration(size)
                for i, cfg in enumerate(cfgs):
                    if cfg['data_parallel_size'] > wafer_num:
                        cfg['data_parallel_size'] = random.randint(1, max(wafer_num, 1))
                    if cfg['micro_batch_size'] * cfg['data_parallel_size'] > test_model_parameters[choose_model]['mini_batch_size']:
                        cfg['micro_batch_size'] = random.randint(1, max(test_model_parameters[choose_model]['mini_batch_size'] // cfg['data_parallel_size'], 1))
                    if math.ceil(design_point['reticle_array_h'] * design_point['reticle_array_w'] / (cfg['num_reticle_per_pipeline_stage'] * cfg['tensor_parallel_size'])) * wafer_num < cfg['model_parallel_size']:
                        cfg['num_reticle_per_pipeline_stage'] = random.randint(1, max(design_point['reticle_array_h'] * design_point['reticle_array_w'] // (cfg['tensor_parallel_size'] * cfg['model_parallel_size']), 1))
                    if cfg['tensor_parallel_size'] * cfg['num_reticle_per_pipeline_stage'] > design_point['reticle_array_h'] * design_point['reticle_array_w']:
                        cfg['num_reticle_per_pipeline_stage'] = random.randint(1, max(design_point['reticle_array_h'] * design_point['reticle_array_w'] // cfg['tensor_parallel_size'], 1))
                    
                    cfgs[i] = cfg
                return cfgs
        
        _space.set_sample_func(inner_sample_configuration)


        def inner_func(_config: sp.Configuration):
            try:
                _dic = dict(_config)
                temp_dic = copy.deepcopy(test_model_parameters[choose_model])
                temp_dic.update(_dic)
                _prediction = api2.evaluate_design_point(design_point=design_point, model_parameters=temp_dic, metric='throughput')
            except:
                print("inner func error!")
                _prediction = -1e10
            result = dict()
            result['objs'] = [-_prediction]
            return result

        _optimizer_kwargs = {
                    'config_space':_space,
                    'objective_function':inner_func,
                    'num_objs':1,
                    'num_constraints':0,
                    'max_runs':inner_max_runs,
                    'surrogate_type':'gp',
                    'acq_optimizer_type':'true_random',
                    'initial_runs':6,
                    'init_strategy':'random',
                    'time_limit_per_trial':1000,
                    'task_id':'moc',
                    'acq_type':'ei',
                    # 'advisor_type':'mf_advisor'
                    'advisor_type':'random'
                    }
        
        _opt = Optimizer(**_optimizer_kwargs)
        _opt.early_stop_threshold = -1e11
        _history = _opt.run()

        prediction = _history.get_incumbents()[0].objectives[0]

    except Exception as e:
        print("outer func error!: ", e)
        prediction = 1e10
        result = dict()
        result['objs'] = [prediction]
        return result
    
    result = dict()
    result['objs'] = [prediction]
    result['config'] = [dict(_history.get_incumbents()[0].config)]

    print('result_config: ', _history.get_incumbents()[0].config)
    return result

def generate(config: sp.Configuration):
    try:
        dic = dict(config)
        lst = tuple(dict(config).values())
        design_point = design_vec2design_dic(lst[1:])

        _space = sp.SelfDefinedConditionedSpace()
        v1 = sp.Int("micro_batch_size", 1, max(test_model_parameters[choose_model]['mini_batch_size'], 2), default_value=1)
        num_of_gpus = [32, 64, 128, 256, 512, 1024, 1536, 1920, 2520, 3072, 6000, 12000, 30000, 60000, 100000, 200000]
        wafer_num = math.ceil(num_of_gpus[choose_model] * 312 * 1000 / (design_point['core_mac_num'] * design_point['core_array_h'] * design_point['core_array_w'] * design_point['reticle_array_h'] * design_point['reticle_array_w'] * 2))
        v2 = sp.Int("data_parallel_size", 1, max(wafer_num, 2), default_value=1)

        factors_of_number_of_layers = factors(test_model_parameters[choose_model]['number_of_layers'])
        v3 = sp.Ordinal("model_parallel_size", factors_of_number_of_layers, default_value=1)

        factors_of_attention_heads = factors(test_model_parameters[choose_model]['attention_heads'])
        v4 = sp.Ordinal("tensor_parallel_size", factors_of_attention_heads, default_value=1)

        num_reticle_per_pipeline_stage_upper_bound = wafer_num * design_point['reticle_array_h'] * design_point['reticle_array_w']
        v5 = sp.Int("num_reticle_per_pipeline_stage", 1, max(num_reticle_per_pipeline_stage_upper_bound, 2), default_value=1)

        _space.add_hyperparameter(v1)
        _space.add_hyperparameter(v2)
        _space.add_hyperparameter(v3)
        _space.add_hyperparameter(v4)
        _space.add_hyperparameter(v5)
        
        def inner_sample_configuration(size):
            if size == 1:
                cfg = super(sp.SelfDefinedConditionedSpace, _space).sample_configuration(size)
                if cfg['data_parallel_size'] > wafer_num:
                    cfg['data_parallel_size'] = random.randint(1, max(wafer_num, 1))
                if cfg['micro_batch_size'] * cfg['data_parallel_size'] > test_model_parameters[choose_model]['mini_batch_size']:
                    cfg['micro_batch_size'] = random.randint(1, max(test_model_parameters[choose_model]['mini_batch_size'] // cfg['data_parallel_size'], 1))
                if math.ceil(design_point['reticle_array_h'] * design_point['reticle_array_w'] / (cfg['num_reticle_per_pipeline_stage'] * cfg['tensor_parallel_size'])) * wafer_num < cfg['model_parallel_size']:
                    cfg['num_reticle_per_pipeline_stage'] = random.randint(1, max(design_point['reticle_array_h'] * design_point['reticle_array_w'] // (cfg['tensor_parallel_size'] * cfg['model_parallel_size']), 1))
                if cfg['tensor_parallel_size'] * cfg['num_reticle_per_pipeline_stage'] > design_point['reticle_array_h'] * design_point['reticle_array_w']:
                    cfg['num_reticle_per_pipeline_stage'] = random.randint(1, max(design_point['reticle_array_h'] * design_point['reticle_array_w'] // cfg['tensor_parallel_size'], 1))
                return cfg
            else:
                cfgs = super(sp.SelfDefinedConditionedSpace, _space).sample_configuration(size)
                for i, cfg in enumerate(cfgs):
                    if cfg['data_parallel_size'] > wafer_num:
                        cfg['data_parallel_size'] = random.randint(1, max(wafer_num, 1))
                    if cfg['micro_batch_size'] * cfg['data_parallel_size'] > test_model_parameters[choose_model]['mini_batch_size']:
                        cfg['micro_batch_size'] = random.randint(1, max(test_model_parameters[choose_model]['mini_batch_size'] // cfg['data_parallel_size'], 1))
                    if math.ceil(design_point['reticle_array_h'] * design_point['reticle_array_w'] / (cfg['num_reticle_per_pipeline_stage'] * cfg['tensor_parallel_size'])) * wafer_num < cfg['model_parallel_size']:
                        cfg['num_reticle_per_pipeline_stage'] = random.randint(1, max(design_point['reticle_array_h'] * design_point['reticle_array_w'] // (cfg['tensor_parallel_size'] * cfg['model_parallel_size']), 1))
                    if cfg['tensor_parallel_size'] * cfg['num_reticle_per_pipeline_stage'] > design_point['reticle_array_h'] * design_point['reticle_array_w']:
                        cfg['num_reticle_per_pipeline_stage'] = random.randint(1, max(design_point['reticle_array_h'] * design_point['reticle_array_w'] // cfg['tensor_parallel_size'], 1))
                    
                    cfgs[i] = cfg
                return cfgs
        
        _space.set_sample_func(inner_sample_configuration)

        _config = _space.sample_configuration()
        _dic = dict(_config)
        temp_dic = copy.deepcopy(test_model_parameters[choose_model])
        temp_dic.update(_dic)
                
        return design_point, temp_dic
        

    except Exception as e:
        print("outer func error!: ", e)
        return None


def func_mo_multi_fidelity_with_inner_search(config: sp.Configuration):
    try:
        dic = dict(config)
        lst = tuple(dict(config).values())
        design_point = design_vec2design_dic(lst[1:])

        _space = sp.SelfDefinedConditionedSpace()
        v1 = sp.Int("micro_batch_size", 1, max(test_model_parameters[choose_model]['mini_batch_size'], 2), default_value=1)
        num_of_gpus = [32, 64, 128, 256, 512, 1024, 1536, 1920, 2520, 3072, 6000, 12000, 30000, 60000, 100000, 200000]
        wafer_num = math.ceil(num_of_gpus[choose_model] * 312 * 1000 / (design_point['core_mac_num'] * design_point['core_array_h'] * design_point['core_array_w'] * design_point['reticle_array_h'] * design_point['reticle_array_w'] * 2))
        v2 = sp.Int("data_parallel_size", 1, max(wafer_num, 2), default_value=1)

        factors_of_number_of_layers = factors(test_model_parameters[choose_model]['number_of_layers'])
        v3 = sp.Ordinal("model_parallel_size", factors_of_number_of_layers, default_value=1)

        factors_of_attention_heads = factors(test_model_parameters[choose_model]['attention_heads'])
        v4 = sp.Ordinal("tensor_parallel_size", factors_of_attention_heads, default_value=1)

        num_reticle_per_pipeline_stage_upper_bound = wafer_num * design_point['reticle_array_h'] * design_point['reticle_array_w']
        v5 = sp.Int("num_reticle_per_pipeline_stage", 1, max(num_reticle_per_pipeline_stage_upper_bound, 2), default_value=1)

        _space.add_hyperparameter(v1)
        _space.add_hyperparameter(v2)
        _space.add_hyperparameter(v3)
        _space.add_hyperparameter(v4)
        _space.add_hyperparameter(v5)
        
        def inner_sample_configuration(size):
            if size == 1:
                cfg = super(sp.SelfDefinedConditionedSpace, _space).sample_configuration(size)
                if cfg['data_parallel_size'] > wafer_num:
                    cfg['data_parallel_size'] = random.randint(1, max(wafer_num, 1))
                if cfg['micro_batch_size'] * cfg['data_parallel_size'] > test_model_parameters[choose_model]['mini_batch_size']:
                    cfg['micro_batch_size'] = random.randint(1, max(test_model_parameters[choose_model]['mini_batch_size'] // cfg['data_parallel_size'], 1))
                if math.ceil(design_point['reticle_array_h'] * design_point['reticle_array_w'] / (cfg['num_reticle_per_pipeline_stage'] * cfg['tensor_parallel_size'])) * wafer_num < cfg['model_parallel_size']:
                    cfg['num_reticle_per_pipeline_stage'] = random.randint(1, max(design_point['reticle_array_h'] * design_point['reticle_array_w'] // (cfg['tensor_parallel_size'] * cfg['model_parallel_size']), 1))
                if cfg['tensor_parallel_size'] * cfg['num_reticle_per_pipeline_stage'] > design_point['reticle_array_h'] * design_point['reticle_array_w']:
                    cfg['num_reticle_per_pipeline_stage'] = random.randint(1, max(design_point['reticle_array_h'] * design_point['reticle_array_w'] // cfg['tensor_parallel_size'], 1))
                return cfg
            else:
                cfgs = super(sp.SelfDefinedConditionedSpace, _space).sample_configuration(size)
                for i, cfg in enumerate(cfgs):
                    if cfg['data_parallel_size'] > wafer_num:
                        cfg['data_parallel_size'] = random.randint(1, max(wafer_num, 1))
                    if cfg['micro_batch_size'] * cfg['data_parallel_size'] > test_model_parameters[choose_model]['mini_batch_size']:
                        cfg['micro_batch_size'] = random.randint(1, max(test_model_parameters[choose_model]['mini_batch_size'] // cfg['data_parallel_size'], 1))
                    if math.ceil(design_point['reticle_array_h'] * design_point['reticle_array_w'] / (cfg['num_reticle_per_pipeline_stage'] * cfg['tensor_parallel_size'])) * wafer_num < cfg['model_parallel_size']:
                        cfg['num_reticle_per_pipeline_stage'] = random.randint(1, max(design_point['reticle_array_h'] * design_point['reticle_array_w'] // (cfg['tensor_parallel_size'] * cfg['model_parallel_size']), 1))
                    if cfg['tensor_parallel_size'] * cfg['num_reticle_per_pipeline_stage'] > design_point['reticle_array_h'] * design_point['reticle_array_w']:
                        cfg['num_reticle_per_pipeline_stage'] = random.randint(1, max(design_point['reticle_array_h'] * design_point['reticle_array_w'] // cfg['tensor_parallel_size'], 1))
                    
                    cfgs[i] = cfg
                return cfgs
        
        _space.set_sample_func(inner_sample_configuration)


        def inner_func(_config: sp.Configuration):
            try:
                _dic = dict(_config)
                temp_dic = copy.deepcopy(test_model_parameters[choose_model])
                temp_dic.update(_dic)
                if lst[0] == 1:
                    _prediction = api1.evaluate_design_point(design_point=design_point, model_parameters=temp_dic, metric='throughput')
                elif lst[0] == 2:
                    _prediction = api2.evaluate_design_point(design_point=design_point, model_parameters=temp_dic, metric='throughput')
                else:
                    raise ValueError
            except:
                print("inner func error!")
                _prediction = -1e10
            result = dict()
            result['objs'] = [-_prediction]
            return result

        _optimizer_kwargs = {
                    'config_space':_space,
                    'objective_function':inner_func,
                    'num_objs':1,
                    'num_constraints':0,
                    'max_runs':inner_max_runs,
                    'surrogate_type':'gp',
                    'acq_optimizer_type':'true_random',
                    'initial_runs':6,
                    'init_strategy':'random',
                    'time_limit_per_trial':1000,
                    'task_id':'moc',
                    'acq_type':'ei',
                    # 'advisor_type':'mf_advisor'
                    'advisor_type':'random'
                    }
        
        _opt = Optimizer(**_optimizer_kwargs)
        _opt.early_stop_threshold = -1e11
        _history = _opt.run()

        prediction = _history.get_incumbents()[0].objectives[0]

    except Exception as e:
        print("outer func error!: ", e)
        prediction = 1e10
        result = dict()
        result['objs'] = [prediction]
        return result
    
    result = dict()
    result['objs'] = [prediction, 1]
    result['config'] = [dict(_history.get_incumbents()[0].config)]

    print('result_config: ', _history.get_incumbents()[0].config)
    return result


def func_mo_single_fidelity_with_inner_search(config: sp.Configuration):
    try:
        dic = dict(config)
        lst = tuple(dict(config).values())
        design_point = design_vec2design_dic(lst)

        _space = sp.SelfDefinedConditionedSpace()
        v1 = sp.Int("micro_batch_size", 1, max(test_model_parameters[choose_model]['mini_batch_size'], 2), default_value=1)
        num_of_gpus = [32, 64, 128, 256, 512, 1024, 1536, 1920, 2520, 3072, 6000, 12000, 30000, 60000, 100000, 200000]
        wafer_num = math.ceil(num_of_gpus[choose_model] * 312 * 1000 / (design_point['core_mac_num'] * design_point['core_array_h'] * design_point['core_array_w'] * design_point['reticle_array_h'] * design_point['reticle_array_w'] * 2))
        v2 = sp.Int("data_parallel_size", 1, max(wafer_num, 2), default_value=1)

        factors_of_number_of_layers = factors(test_model_parameters[choose_model]['number_of_layers'])
        v3 = sp.Ordinal("model_parallel_size", factors_of_number_of_layers, default_value=1)

        factors_of_attention_heads = factors(test_model_parameters[choose_model]['attention_heads'])
        v4 = sp.Ordinal("tensor_parallel_size", factors_of_attention_heads, default_value=1)

        num_reticle_per_pipeline_stage_upper_bound = wafer_num * design_point['reticle_array_h'] * design_point['reticle_array_w']
        v5 = sp.Int("num_reticle_per_pipeline_stage", 1, max(num_reticle_per_pipeline_stage_upper_bound, 2), default_value=1)

        _space.add_hyperparameter(v1)
        _space.add_hyperparameter(v2)
        _space.add_hyperparameter(v3)
        _space.add_hyperparameter(v4)
        _space.add_hyperparameter(v5)
        
        def inner_sample_configuration(size):
            if size == 1:
                cfg = super(sp.SelfDefinedConditionedSpace, _space).sample_configuration(size)
                if cfg['data_parallel_size'] > wafer_num:
                    cfg['data_parallel_size'] = random.randint(1, max(wafer_num, 1))
                if cfg['micro_batch_size'] * cfg['data_parallel_size'] > test_model_parameters[choose_model]['mini_batch_size']:
                    cfg['micro_batch_size'] = random.randint(1, max(test_model_parameters[choose_model]['mini_batch_size'] // cfg['data_parallel_size'], 1))
                if math.ceil(design_point['reticle_array_h'] * design_point['reticle_array_w'] / (cfg['num_reticle_per_pipeline_stage'] * cfg['tensor_parallel_size'])) * wafer_num < cfg['model_parallel_size']:
                    cfg['num_reticle_per_pipeline_stage'] = random.randint(1, max(design_point['reticle_array_h'] * design_point['reticle_array_w'] // (cfg['tensor_parallel_size'] * cfg['model_parallel_size']), 1))
                if cfg['tensor_parallel_size'] * cfg['num_reticle_per_pipeline_stage'] > design_point['reticle_array_h'] * design_point['reticle_array_w']:
                    cfg['num_reticle_per_pipeline_stage'] = random.randint(1, max(design_point['reticle_array_h'] * design_point['reticle_array_w'] // cfg['tensor_parallel_size'], 1))
                return cfg
            else:
                cfgs = super(sp.SelfDefinedConditionedSpace, _space).sample_configuration(size)
                for i, cfg in enumerate(cfgs):
                    if cfg['data_parallel_size'] > wafer_num:
                        cfg['data_parallel_size'] = random.randint(1, max(wafer_num, 1))
                    if cfg['micro_batch_size'] * cfg['data_parallel_size'] > test_model_parameters[choose_model]['mini_batch_size']:
                        cfg['micro_batch_size'] = random.randint(1, max(test_model_parameters[choose_model]['mini_batch_size'] // cfg['data_parallel_size'], 1))
                    if math.ceil(design_point['reticle_array_h'] * design_point['reticle_array_w'] / (cfg['num_reticle_per_pipeline_stage'] * cfg['tensor_parallel_size'])) * wafer_num < cfg['model_parallel_size']:
                        cfg['num_reticle_per_pipeline_stage'] = random.randint(1, max(design_point['reticle_array_h'] * design_point['reticle_array_w'] // (cfg['tensor_parallel_size'] * cfg['model_parallel_size']), 1))
                    if cfg['tensor_parallel_size'] * cfg['num_reticle_per_pipeline_stage'] > design_point['reticle_array_h'] * design_point['reticle_array_w']:
                        cfg['num_reticle_per_pipeline_stage'] = random.randint(1, max(design_point['reticle_array_h'] * design_point['reticle_array_w'] // cfg['tensor_parallel_size'], 1))
                    
                    cfgs[i] = cfg
                return cfgs
        
        _space.set_sample_func(inner_sample_configuration)


        def inner_func(_config: sp.Configuration):
            try:
                _dic = dict(_config)
                temp_dic = copy.deepcopy(test_model_parameters[choose_model])
                temp_dic.update(_dic)
                _prediction = api1.evaluate_design_point(design_point=design_point, model_parameters=temp_dic, metric='throughput')
            except:
                print("inner func error!")
                _prediction = -1e10
            result = dict()
            result['objs'] = [-_prediction]
            return result

        _optimizer_kwargs = {
                    'config_space':_space,
                    'objective_function':inner_func,
                    'num_objs':1,
                    'num_constraints':0,
                    'max_runs':inner_max_runs,
                    'surrogate_type':'gp',
                    'acq_optimizer_type':'true_random',
                    'initial_runs':6,
                    'init_strategy':'random',
                    'time_limit_per_trial':1000,
                    'task_id':'moc',
                    'acq_type':'ei',
                    # 'advisor_type':'mf_advisor'
                    'advisor_type':'random'
                    }
        
        _opt = Optimizer(**_optimizer_kwargs)
        _opt.early_stop_threshold = -1e11
        _history = _opt.run()

        prediction = _history.get_incumbents()[0].objectives[0]

    except Exception as e:
        print("outer func error!: ", e)
        prediction = 1e10
        result = dict()
        result['objs'] = [prediction]
        return result
    
    result = dict()
    result['objs'] = [prediction, 1]
    result['config'] = [dict(_history.get_incumbents()[0].config)]

    print('result_config: ', _history.get_incumbents()[0].config)
    return result

def test_KT(data_lst): # data_list : List[(x_i, y_i), ]
    pairs = []
    length = len(data_lst)
    c_2n = length * (length - 1) // 2
    num_discordant = 0
    for i in tqdm(range(length)):
        for j in range(length):
            if i != j:
                # x1 = api1.evaluate_design_point(design_point=data_lst[i][0], model_parameters=data_lst[i][1], metric='throughput')
                # y1 = api2.evaluate_design_point(design_point=data_lst[i][0], model_parameters=data_lst[i][1], metric='throughput')

                # x2 = api1.evaluate_design_point(design_point=data_lst[j][0], model_parameters=data_lst[j][1], metric='throughput')
                # y2 = api2.evaluate_design_point(design_point=data_lst[j][0], model_parameters=data_lst[j][1], metric='throughput')
                x1 = data_lst[i][0]
                y1 = data_lst[i][1]

                x2 = data_lst[j][0]
                y2 = data_lst[j][1]

                if (x1 - x2) * (y1 - y2) < 0:
                    num_discordant += 1
                    pairs.append((i, j))
    
    return 1 - 2 * num_discordant / c_2n, pairs

def get_evaluation_list(points_list):
    evaluation_lst = []

    for point in points_list:
        x = api1.evaluate_design_point(design_point=point[0], model_parameters=point[1], metric='throughput')
        y = api2.evaluate_design_point(design_point=point[0], model_parameters=point[1], metric='throughput')

        evaluation_lst.append((x, y))

    return evaluation_lst


def get_evaluation_list_multi_process(points_list, threads = 1):
    evaluation_lst = []
    manager = multiprocessing.Manager()
    queue = manager.Queue()
    pool = []

    points_per_thread = len(points_list) // threads

    def get_evaluation_list_single_thread(points_list, queue):
        evaluation_lst = []
        for point in points_list:
            x = api1.evaluate_design_point(design_point=point[0], model_parameters=point[1], metric='throughput')
            y = api2.evaluate_design_point(design_point=point[0], model_parameters=point[1], metric='throughput')
            evaluation_lst.append((x, y))
            queue.put((x, y))
        return evaluation_lst

    for i in range(threads):
        if i == threads - 1:
            _points_list = points_list[i * points_per_thread:]
        else:
            _points_list = points_list[i * points_per_thread:(i + 1) * points_per_thread]
        
        p = Process(target=get_evaluation_list_single_thread, args=(_points_list, queue))
        pool.append(p)
        p.start()
    
    for p in pool:
        p.join()

    while not queue.empty():
        evaluation = queue.get()
        evaluation_lst.append(evaluation)

    return evaluation_lst

