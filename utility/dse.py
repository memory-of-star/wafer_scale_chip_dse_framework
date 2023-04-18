import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dse4wse/test/dse'))
from openbox import Optimizer, sp, History
from ConfigSpace import Configuration
from multiprocessing import Process, Queue
import multiprocessing
from typing import List
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import evaluator
import plot
import copy
import random
from tqdm import tqdm
import api
import test_model_parameters
import math
import utility

class DSE():
    def __init__(self, choose_model=0, strategy='multi_fidelity', run_name='default', run_times=10, max_runs=100):
        
        # file kwargs
        self.root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # multi processes kwargs
        self.manager = multiprocessing.Manager()
        self.queue = self.manager.Queue()
        self.pool = []

        # space kwargs
        self.choose_model = choose_model
        self.num_of_gpus = [32, 64, 128, 256, 512, 1024, 1536, 1920, 2520, 3072, 6000, 12000, 30000, 60000, 100000, 200000]
        self.fixed_model_parameters = test_model_parameters.test_model_parameters

        print('constructing design space...')
        self.design_points, self.design_space = self.build_design_space()
        print('constructing design space completed!')

        self.dimension_name = ["core_buffer_size", "core_buffer_bw", "core_mac_num", "core_noc_bw", "core_noc_vc",
            "core_noc_buffer_size", "reticle_bw", "core_array_h", "core_array_w", "wafer_mem_bw", "reticle_array_h",
            "reticle_array_w"]
        self.points_dic = [{k:v for k,v in zip(self.dimension_name, self.design_points[i])} for i in range(len(self.design_points))]

        # optimization kwargs
        self.strategy = strategy
        self.run_name = run_name
        self.run_times = run_times
        self.max_runs = max_runs

        self.metric = 'throughput'

        self.num_objs = 1
        self.initial_runs = 6
        self.init_strategy = 'random'
        self.fidelity_functions = [self.evaluator_factory(use_high_fidelity=True, metric='throughput'), self.evaluator_factory(use_high_fidelity=False, metric='throughput')] # a list, from high fidelity to low fidelity
        
        print('constructing optimization space...')
        self.space = self.build_optimization_space()
        print('constructing optimization space completed!')

        self.optimizer_kwargs = {
            'config_space':self.space,
            'num_objs':self.num_objs,
            'num_constraints':0,
            'max_runs':self.max_runs,
            'surrogate_type':'gp',
            'acq_optimizer_type':'true_random',
            'initial_runs':self.initial_runs,
            'init_strategy':self.init_strateg,
            'time_limit_per_trial':1000,
            'task_id':'moc',
            'acq_type':'ei',
        }

        if strategy == 'multi_fidelity':
            self.optimizer_kwargs.update({
                            'objective_function':None,
                            'fidelity_objective_functions':self.fidelity_functions,
                            'advisor_type':'mfes_advisor'
                            })
        elif strategy == 'random':
            self.optimizer_kwargs.update({
                            'objective_function':self.fidelity_functions[0],
                            'advisor_type':'random'
                            })
            
        elif strategy == 'single_fidelity':
            self.optimizer_kwargs.update({
                            'objective_function':self.fidelity_functions[0],
                            'advisor_type':'default'
                            })

        # model parameter
        self.factors_of_number_of_layers = self.factors(self.fixed_model_parameters[self.choose_model]['number_of_layers'])
        self.factors_of_attention_heads = self.factors(self.fixed_model_parameters[self.choose_model]['attention_heads'])

        print('initialization completed!')

    def run(self):
        for _ in range(self.run_times):
            p = Process(target=self.process)
            p.start()
            self.pool.append(p)

        for p in self.pool:
            p.join()

        histories = []
        while not self.queue.empty():
            history = self.queue.get()
            histories.append(history)

        with open(os.path.join(self.root_path, 'result/pickle', self.run_name+'.pickle'), "wb") as f:
            pickle.dump(histories, f)


    def build_design_space(self):
        design_points = np.load(os.path.join(self.root_path, 'data/design_points.npy'), allow_pickle=True)

        with open(os.path.join(self.root_path, 'data/design_space.pickle'), 'rb') as f:
            design_space = pickle.load(f)

        for i in range(len(design_space)):
            design_space[i] = list(design_space[i])
            design_space[i].sort()

        return design_points, design_space

    def build_optimization_space(self):
        space = sp.SelfDefinedConditionedSpace()
        
        variable_lst = []
        for i in range(len(self.design_space)):
            if self.dimension_name[i] == 'reticle_bw': # 'reticle_bw' is float, other variables are int
                variable_lst.append(sp.Real(self.dimension_name[i], self.design_space[i][0], max(self.design_space[i][-1], self.design_space[i][0] + 0.01), default_value=self.design_points[0][i]))
            else:
                variable_lst.append(sp.Int(self.dimension_name[i], self.design_space[i][0], max(self.design_space[i][-1], self.design_space[i][0] + 1), default_value=int(self.design_points[0][i])))

        # self.dimension_name = ["core_buffer_size", "core_buffer_bw", "core_mac_num", "core_noc_bw", "core_noc_vc",
        #     "core_noc_buffer_size", "reticle_bw", "core_array_h", "core_array_w", "wafer_mem_bw", "reticle_array_h",
        #     "reticle_array_w"]

        max_wafer_num = math.ceil(self.num_of_gpus[self.choose_model] * 312 * 1000 / (4 * 8 * 8 * 1 * 1 * 2)) # here we need to update the numbers when we change the dataset
        factors_of_number_of_layers = self.factors(self.fixed_model_parameters[self.choose_model]['number_of_layers'])
        factors_of_attention_heads = self.factors(self.fixed_model_parameters[self.choose_model]['attention_heads'])
        num_reticle_per_model_chunk_upper_bound = max_wafer_num * 55 * 73

        v_micro_batch_size = sp.Int("micro_batch_size", 1, max(self.fixed_model_parameters[self.choose_model]['mini_batch_size'], 2), default_value=1)
        v_data_parallel_size = sp.Int("data_parallel_size", 1, max(max_wafer_num, 2), default_value=1)
        v_model_parallel_size = sp.Ordinal("model_parallel_size", factors_of_number_of_layers, default_value=1)
        v_tensor_parallel_size = sp.Ordinal("tensor_parallel_size", factors_of_attention_heads, default_value=1)
        v_num_reticle_per_model_chunk = sp.Int("num_reticle_per_model_chunk", 1, max(num_reticle_per_model_chunk_upper_bound, 2), default_value=1)
        v_weight_streaming = sp.Int('weight_streaming', 0, 1, default_value=0)

        space.add_variables(variable_lst)
        space.add_variables([v_micro_batch_size, v_data_parallel_size, v_model_parallel_size, v_tensor_parallel_size, v_num_reticle_per_model_chunk, v_weight_streaming])

        sampling_func = self.sampling_configuration_factory()
        space.set_sample_func(sampling_func)
        return space


    def process(self):
        
        opt = Optimizer(**self.optimizer_kwargs)

        if self.strategy == 'multi_fidelity':
            histories = opt.mfes_run()
        else:
            histories = opt.run()
        
        data = self.parse_histories(histories=[histories])
        self.queue.put(data[0])

    def config_2_design_model(self, config):
        design_point = dict(config)
        model_para = copy.deepcopy(self.fixed_model_parameters[self.choose_model])
        model_para["micro_batch_size"] = design_point.pop("micro_batch_size")
        model_para["data_parallel_size"] = design_point.pop("data_parallel_size")
        model_para["model_parallel_size"] = design_point.pop("model_parallel_size")
        model_para["tensor_parallel_size"] = design_point.pop("tensor_parallel_size")
        model_para["num_reticle_per_model_chunk"] = design_point.pop("num_reticle_per_model_chunk")
        model_para['weight_streaming'] = design_point.pop("weight_streaming")

        return design_point, model_para

    def parse_histories(self, histories):
        if self.strategy == 'multi_fidelity':
            ret = []  # max_runs, 3(e.g. None, fidelity1, fidelity2), length of fidelity 1 history (or 2)
            for run_num in range(len(histories)):
                single_run = []  # 2(e.g. None, fidelity1, fidelity2), length of fidelity 1 history (or 2)
                for fidelity in range(2):
                    single_fidelity = []  # length of fidelity 1 history (or 2)
                    for i in range(len(histories[run_num][fidelity].observations)):
                        design_point, model_para = self.config_2_design_model(histories[run_num][fidelity].observations[i].config)
                        obj = histories[run_num][fidelity].observations[i].objectives[0]

                        single_fidelity.append((design_point, model_para, obj))
                    single_run.append(single_fidelity)
                ret.append(single_run)
        elif self.strategy == 'single_fidelity' or self.strategy == 'random':
            ret = []  # max_runs, length of fidelity 1 history (or 2)
            for run_num in range(len(histories)):
                single_run = []  # length of fidelity 1 history (or 2)

                for i in range(len(histories[run_num].observations)):
                    design_point, model_para = self.config_2_design_model(histories[run_num].observations[i].config)
                    obj = histories[run_num].observations[i].objectives[0]

                    single_run.append((design_point, model_para, obj))
                ret.append(single_run)

        return ret

    def get_legal_configuration(self):
        cnt = 0
        while True:
            design_point = copy.deepcopy(random.choice(self.points_dic))
            model_parallel = self.get_legal_model_parallel(design_point)
            design_point, model_parallel = self.fine_tune_model_para(design_point, model_parallel)
            model = copy.deepcopy(self.fixed_model_parameters[self.choose_model])
            model.update(model_parallel)

            wafer_scale_engine = api.create_wafer_scale_engine(**design_point)
            evaluator = api.create_evaluator(True, wafer_scale_engine, **model)

            cnt += 1

            try:
                evaluator._find_best_intra_model_chunk_exec_params(inference=False)
                break
            except:
                pass

            if cnt > 10000:
                raise ValueError('Can\'t get invalid configuration!')

        design_point.update(model_parallel)
        return Configuration(self.space, design_point)

    def sampling_configuration_factory(self):
        def sample_configuration(size):
            if size == 1:
                return self.get_legal_configuration()
            else:
                ret = []
                for _ in range(size):
                    ret.append(self.get_legal_configuration())
                return ret
        return sample_configuration

    def get_legal_model_parallel(self, design_point):
        model_parallel = {}
        model_parallel['micro_batch_size'] = random.randint(1, max(self.fixed_model_parameters[self.choose_model]['mini_batch_size'], 1))

        wafer_num = math.ceil(self.num_of_gpus[self.choose_model] * 312 * 1000 / (design_point['core_mac_num'] * design_point['core_array_h'] * design_point['core_array_w'] * design_point['reticle_array_h'] * design_point['reticle_array_w'] * 2))
        model_parallel['data_parallel_size'] = random.randint(1, max(wafer_num, 1))

        factors_of_number_of_layers = self.factors(self.fixed_model_parameters[self.choose_model]['number_of_layers'])
        model_parallel['model_parallel_size'] = random.choice(factors_of_number_of_layers)

        factors_of_attention_heads = self.factors(self.fixed_model_parameters[self.choose_model]['attention_heads'])
        model_parallel['tensor_parallel_size'] = random.choice(factors_of_attention_heads)

        num_reticle_per_pipeline_stage_upper_bound = wafer_num * design_point['reticle_array_h'] * design_point['reticle_array_w']
        model_parallel['num_reticle_per_model_chunk'] = random.randint(1, max(num_reticle_per_pipeline_stage_upper_bound, 1))

        model_parallel['weight_streaming'] = random.randint(0, 1)

        return model_parallel

    def fine_tune_model_para(self, design_point, model_parallel):
        
        wafer_num = math.ceil(self.num_of_gpus[self.choose_model] * 312 * 1000 / (design_point['core_mac_num'] * design_point['core_array_h'] * design_point['core_array_w'] * design_point['reticle_array_h'] * design_point['reticle_array_w'] * 2))
        model_parallel['num_reticle_per_model_chunk'] = design_point['reticle_array_h'] * design_point['reticle_array_w']

        if model_parallel['micro_batch_size'] * model_parallel['data_parallel_size'] > self.fixed_model_parameters[self.choose_model]['mini_batch_size']:
            model_parallel['micro_batch_size'] = random.randint(1, max(self.fixed_model_parameters[self.choose_model]['mini_batch_size'] // model_parallel['data_parallel_size'], 1))

        if model_parallel['data_parallel_size'] * model_parallel['model_parallel_size'] > wafer_num:
            for i in range(len(self.factors_of_number_of_layers)):
                if self.factors_of_number_of_layers[i] > max(wafer_num // model_parallel['data_parallel_size'], 1):
                    break 
            model_parallel['model_parallel_size'] = random.choice(self.factors_of_number_of_layers[:i])

        # if model_parallel['tensor_parallel_size'] * model_parallel['num_reticle_per_model_chunk'] > design_point['reticle_array_h'] * design_point['reticle_array_w']:
        #     for i in range(len(self.factors_of_attention_heads)):
        #         if self.factors_of_attention_heads[i] > max(design_point['reticle_array_h'] * design_point['reticle_array_w'] // model_parallel['num_reticle_per_model_chunk'], 1):
        #             break 
        #     model_parallel['tensor_parallel_size'] = random.choice(self.factors_of_attention_heads[:i])
        model_parallel['tensor_parallel_size'] = 1

        return design_point, model_parallel


    def generate_legal_points(self, num = 100):
        ret = []
        for i in tqdm(range(num)):
            design_point, model_parameters = self.config_2_design_model(self.space.sample_configuration())
            ret.append((design_point, model_parameters))

        return ret

    def KT_evaluator(self, size=100, multi_process = True, threads = 20):
        points_lst = self.generate_legal_points(size)
        if multi_process:
            evaluation_list = evaluator.get_evaluation_list_multi_process(points_lst, threads=threads)
        else:
            evaluation_list = evaluator.get_evaluation_list(points_lst)
        kt, pairs = evaluator.test_KT(evaluation_list)
        return kt, points_lst, evaluation_list, pairs
        

    def evaluator_factory(self, use_high_fidelity=True, metric='throughput'):
        _use_high_fidelity = use_high_fidelity
        _metric = metric
        def evaluation_func(config: sp.Configuration):
            nonlocal _use_high_fidelity
            nonlocal _metric
            try:
                design_point, model_parameters = self.config_2_design_model(config)
                prediction = api.evaluate_design_point(design_point=design_point, model_parameters=model_parameters, metric=_metric, use_high_fidelity=_use_high_fidelity)
            except:
                prediction = -1e10
            result = dict()
            result['objs'] = [-prediction]
            return result

        return evaluation_func

    def factors(self, n):
        factors = [] 
        factors.append(1)
        for i in range(2, n + 1):
            if n % i == 0:
                factors.append(i)
        return factors 