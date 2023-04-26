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
    def __init__(self, choose_model=0, strategy='multi_fidelity', run_name='default', run_times=10, max_runs=100, metrics=['throughput'], ref_point=[0, 300], mean_of_all = False, use_low_fidelity = False, add_noise=False):

        self.throughput_weight = []

        self.mean_of_all = mean_of_all
        self.add_noise = add_noise

        # random.seed(1)
        self.factors_ = {}
        
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
        self.design_points, self.design_space, self.points_dic = self.build_design_space()
        print('constructing design space completed!')

        self.dimension_name = ["core_buffer_size", "core_buffer_bw", "core_mac_num", "core_noc_bw", "core_noc_vc",
            "core_noc_buffer_size", "reticle_bw", "core_array_h", "core_array_w", "wafer_mem_bw", "reticle_array_h",
            "reticle_array_w"]
        # self.points_dic = [{k:v for k,v in zip(self.dimension_name, self.design_points[i])} for i in range(len(self.design_points))]


        # optimization kwargs
        self.strategy = strategy
        self.run_name = run_name
        self.run_times = run_times
        self.max_runs = max_runs
        self.ref_point = ref_point

        self.use_low_fidelity = use_low_fidelity

        self.num_objs = len(metrics)
        self.metrics = metrics

        if self.use_low_fidelity:
            self.fidelity_functions = [self.evaluator_factory(use_high_fidelity=False, metrics=self.metrics, use_mean=True)]
        elif not mean_of_all:
            self.fidelity_functions = [self.evaluator_factory(use_high_fidelity=True, metrics=self.metrics), self.evaluator_factory(use_high_fidelity=False, metrics=self.metrics)] # a list, from high fidelity to low fidelity
        else:
            self.fidelity_functions = [self.evaluator_factory(use_high_fidelity=True, metrics=self.metrics, use_mean=True, add_noise=self.add_noise), self.evaluator_factory(use_high_fidelity=False, metrics=self.metrics, use_mean=True)]

        self.initial_runs = 6
        self.init_strategy = 'random'
        
        
        print('constructing optimization space...')
        if not mean_of_all:
            self.space = self.build_optimization_space()
        else:
            self.space = self.build_optimization_space_fixed_model_parallel()
        print('constructing optimization space completed!')

        self.optimizer_kwargs = {
            'config_space':self.space,
            'num_objs':self.num_objs,
            'num_constraints':0,
            'max_runs':self.max_runs,
            'surrogate_type':'gp',
            'acq_optimizer_type':'true_random',
            'initial_runs':self.initial_runs,
            'init_strategy':self.init_strategy,
            'time_limit_per_trial':1000,
            'random_state':None,
            'task_id':'moc',
            'acq_type':'ei',
            'num_acq_optimizer_points':20000,
        }
        if self.num_objs > 1:
            self.optimizer_kwargs.update({'ref_point':self.ref_point,
                                          'acq_type':'ehvi',})

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
        self.factors_of_batch_size = self.factors(self.fixed_model_parameters[self.choose_model]['mini_batch_size'])

        self.factors_of_batch_size_all_models = []
        for cm in range(len(self.fixed_model_parameters)):
            self.factors_of_batch_size_all_models.append(self.factors(self.fixed_model_parameters[cm]['mini_batch_size']))

        print('start initializing throughput weight...')
        self.initialize_throughput_weight()

        print('initialization completed!')

    def run(self):
        for _ in range(self.run_times):
            p = Process(target=self.process)
            p.start()
            self.pool.append(p)

        for p in self.pool:
            p.join()

        del self.points_dic # 这里之后就用不到self.points_dic了

        histories = []
        while not self.queue.empty():
            history = self.queue.get()
            histories.append(history)

        with open(os.path.join(self.root_path, 'result/pickle', self.run_name+'.pickle'), "wb") as f:
            pickle.dump(histories, f)


    def build_design_space(self):
        # design_points = np.load(os.path.join(self.root_path, 'data/design_points3.npy'), allow_pickle=True)

        # with open(os.path.join(self.root_path, 'data/design_space3.pickle'), 'rb') as f:
        #     design_space = pickle.load(f)

        # with open(os.path.join(self.root_path, 'data/points_dic3.pickle'), 'rb') as f:
        #     points_dic = pickle.load(f)

        design_points = np.load(os.path.join(self.root_path, 'data/GPU_design_points.npy'), allow_pickle=True)

        with open(os.path.join(self.root_path, 'data/GPU_design_space.pickle'), 'rb') as f:
            design_space = pickle.load(f)

        with open(os.path.join(self.root_path, 'data/GPU_points_dic.pickle'), 'rb') as f:
            points_dic = pickle.load(f)

        for i in range(len(design_space)):
            design_space[i] = list(design_space[i])
            design_space[i].sort()

        return design_points, design_space, points_dic

    def build_optimization_space(self):
        space = sp.SelfDefinedConditionedSpace()
        
        variable_lst = []
        for i in range(len(self.design_space)):
            if self.dimension_name[i] == 'reticle_bw': # 'reticle_bw' is float, other variables are int
                variable_lst.append(sp.Real(self.dimension_name[i], self.design_space[i][0], max(self.design_space[i][-1], self.design_space[i][0] + 0.01), default_value=self.design_points[0][i]))
            else:
                variable_lst.append(sp.Int(self.dimension_name[i], self.design_space[i][0], max(self.design_space[i][-1], self.design_space[i][0] + 1), default_value=int(self.design_points[0][i])))

        del self.design_points # 这之后self.design_points就用不到了

        # self.dimension_name = ["core_buffer_size", "core_buffer_bw", "core_mac_num", "core_noc_bw", "core_noc_vc",
        #     "core_noc_buffer_size", "reticle_bw", "core_array_h", "core_array_w", "wafer_mem_bw", "reticle_array_h",
        #     "reticle_array_w"]

        max_wafer_num = math.ceil(self.num_of_gpus[self.choose_model] * 312 * 1000 / (4 * 8 * 8 * 1 * 1 * 2)) # here we need to update the numbers when we change the dataset
        # factors_of_number_of_layers = self.factors(self.fixed_model_parameters[self.choose_model]['number_of_layers'])
        # factors_of_attention_heads = self.factors(self.fixed_model_parameters[self.choose_model]['attention_heads'])
        num_reticle_per_model_chunk_upper_bound = max_wafer_num * 55 * 73

        v_micro_batch_size = sp.Int("micro_batch_size", 1, max(self.fixed_model_parameters[self.choose_model]['mini_batch_size'], 2), default_value=1)
        v_data_parallel_size = sp.Int("data_parallel_size", 1, max(max_wafer_num, 2), default_value=1)
        v_model_parallel_size = sp.Int("model_parallel_size", 1, max(self.fixed_model_parameters[self.choose_model]['number_of_layers'], 2), default_value=1)
        v_tensor_parallel_size = sp.Int("tensor_parallel_size", 1, max(self.fixed_model_parameters[self.choose_model]['attention_heads'], 2), default_value=1)
        v_num_reticle_per_model_chunk = sp.Int("num_reticle_per_model_chunk", 1, max(num_reticle_per_model_chunk_upper_bound, 2), default_value=1)
        v_weight_streaming = sp.Int('weight_streaming', 0, 1, default_value=0)

        space.add_variables(variable_lst)
        space.add_variables([v_micro_batch_size, v_data_parallel_size, v_model_parallel_size, v_tensor_parallel_size, v_num_reticle_per_model_chunk, v_weight_streaming])

        sampling_func = self.sampling_configuration_factory()
        space.set_sample_func(sampling_func)
        return space
    

    def build_optimization_space_fixed_model_parallel(self):
        space = sp.SelfDefinedConditionedSpace()
        
        variable_lst = []
        for i in range(len(self.design_space)):
            if self.dimension_name[i] == 'reticle_bw': # 'reticle_bw' is float, other variables are int
                variable_lst.append(sp.Real(self.dimension_name[i], self.design_space[i][0], max(self.design_space[i][-1], self.design_space[i][0] + 0.01), default_value=self.design_points[0][i]))
            else:
                variable_lst.append(sp.Int(self.dimension_name[i], self.design_space[i][0], max(self.design_space[i][-1], self.design_space[i][0] + 1), default_value=int(self.design_points[0][i])))

        del self.design_points # 这之后self.design_points就用不到了

        space.add_variables(variable_lst)

        # sampling_func = self.sampling_configuration_factory()
        def func(size=1):
            if size == 1:
                design_point = random.choice(self.points_dic)
                return Configuration(space, design_point)
            else:
                design_points = random.choices(self.points_dic, k=size)
                ret = []
                for design_point in design_points:
                    ret.append(Configuration(space, design_point))
                return ret
            
        space.set_sample_func(func)
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
                        if not self.mean_of_all:
                            design_point, model_para = self.config_2_design_model(histories[run_num][fidelity].observations[i].config)
                            obj = histories[run_num][fidelity].observations[i].objectives
                            single_fidelity.append((design_point, model_para, obj))
                        else:
                            design_point = dict(histories[run_num][fidelity].observations[i].config)
                            obj = histories[run_num][fidelity].observations[i].objectives
                            single_fidelity.append((design_point, obj))
                    single_run.append(single_fidelity)
                ret.append(single_run)
        elif self.strategy == 'single_fidelity' or self.strategy == 'random':
            ret = []  # max_runs, length of fidelity 1 history (or 2)
            for run_num in range(len(histories)):
                single_run = []  # length of fidelity 1 history (or 2)

                for i in range(len(histories[run_num].observations)):
                    if not self.mean_of_all:
                        design_point, model_para = self.config_2_design_model(histories[run_num].observations[i].config)
                        obj = histories[run_num].observations[i].objectives # a list of objectives, length equal to self.num_objs
                        single_run.append((design_point, model_para, obj))
                    else:
                        design_point = dict(histories[run_num].observations[i].config)
                        obj = histories[run_num].observations[i].objectives
                        single_run.append((design_point, obj))
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

            break

            # wafer_scale_engine = api.create_wafer_scale_engine(**design_point)
            # evaluator = api.create_evaluator(True, wafer_scale_engine, **model)

            # cnt += 1

            # try:
            #     evaluator._find_best_intra_model_chunk_exec_params(inference=False)
            #     break
            # except:
            #     pass

            # if cnt > 10000:
            #     raise ValueError('Can\'t get invalid configuration!')

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

        # factors_of_batch_size = self.factors(self.fixed_model_parameters[self.choose_model]['mini_batch_size'])
        model_parallel['micro_batch_size'] = random.choice(self.factors_of_batch_size)

        wafer_num = math.ceil(self.num_of_gpus[self.choose_model] * 312 * 1000 / (design_point['core_mac_num'] * design_point['core_array_h'] * design_point['core_array_w'] * design_point['reticle_array_h'] * design_point['reticle_array_w'] * 2))
        
        factors_of_wafer_num = self.factors(max(wafer_num, 1))
        # model_parallel['data_parallel_size'] = random.randint(1, max(wafer_num, 1))
        model_parallel['data_parallel_size'] = random.choice(factors_of_wafer_num)

        # model_parallel['model_parallel_size'] = random.randint(1, max(self.fixed_model_parameters[self.choose_model]['number_of_layers'], 1))

        model_parallel['model_parallel_size'] = random.choice(self.factors_of_number_of_layers)

        # factors_of_attention_heads = self.factors(self.fixed_model_parameters[self.choose_model]['attention_heads'])
        model_parallel['tensor_parallel_size'] = random.choice(self.factors_of_attention_heads)

        num_reticle_per_pipeline_stage_upper_bound = wafer_num * design_point['reticle_array_h'] * design_point['reticle_array_w']
        model_parallel['num_reticle_per_model_chunk'] = random.randint(1, max(num_reticle_per_pipeline_stage_upper_bound, 1))

        model_parallel['weight_streaming'] = random.randint(0, 1)

        return model_parallel

    def fine_tune_model_para(self, design_point, model_parallel):
        
        wafer_num = math.ceil(self.num_of_gpus[self.choose_model] * 312 * 1000 / (design_point['core_mac_num'] * design_point['core_array_h'] * design_point['core_array_w'] * design_point['reticle_array_h'] * design_point['reticle_array_w'] * 2))
        model_parallel['num_reticle_per_model_chunk'] = design_point['reticle_array_h'] * design_point['reticle_array_w']
        factors_of_wafer_num = self.factors(max(wafer_num, 1))

        if model_parallel['micro_batch_size'] * model_parallel['data_parallel_size'] > self.fixed_model_parameters[self.choose_model]['mini_batch_size']:
            # model_parallel['micro_batch_size'] = random.randint(1, max(self.fixed_model_parameters[self.choose_model]['mini_batch_size'] // model_parallel['data_parallel_size'], 1))
            t = random.randint(0, 1)
            if t == 0:
                for i in range(len(self.factors_of_batch_size)):
                    if self.factors_of_batch_size[i] > max(self.fixed_model_parameters[self.choose_model]['mini_batch_size'] // model_parallel['data_parallel_size'], 1):
                        break 
                i = max(i, 1)
                model_parallel['micro_batch_size'] = random.choice(self.factors_of_batch_size[:i])
            else:
                for i in range(len(factors_of_wafer_num)):
                    if factors_of_wafer_num[i] > max(self.fixed_model_parameters[self.choose_model]['mini_batch_size'] // model_parallel['micro_batch_size'], 1):
                        break 
                i = max(i, 1)
                model_parallel['data_parallel_size'] = random.choice(factors_of_wafer_num[:i])

        if model_parallel['data_parallel_size'] * model_parallel['model_parallel_size'] > wafer_num:
            t = random.randint(0, 1)
            if t == 0:
                for i in range(len(factors_of_wafer_num)):
                    if factors_of_wafer_num[i] > max(wafer_num//model_parallel['model_parallel_size'], 1):
                        break 
                i = max(i, 1)
                model_parallel['data_parallel_size'] = random.choice(factors_of_wafer_num[:i])
                
            else:
                for i in range(len(self.factors_of_number_of_layers)):
                    if self.factors_of_number_of_layers[i] > max(wafer_num//model_parallel['data_parallel_size'], 1):
                        break 
                i = max(i, 1)
                model_parallel['model_parallel_size'] = random.choice(self.factors_of_number_of_layers[:i])

        if model_parallel['tensor_parallel_size'] > design_point['reticle_array_h'] * design_point['reticle_array_w']:
            for i in range(len(self.factors_of_attention_heads)):
                if self.factors_of_attention_heads[i] > max(design_point['reticle_array_h'] * design_point['reticle_array_w'], 1):
                    break 
            i = max(i, 1)
            model_parallel['tensor_parallel_size'] = random.choice(self.factors_of_attention_heads[:i])


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

    def KT_evaluator_16_models_mean(self, size=100, threads = 20):
        assert self.mean_of_all == True

        manager = multiprocessing.Manager()
        queues = [manager.Queue() for _ in range(threads)]
        pool = []

        configs = self.space.sample_configuration(size=size)

        points = [dict(config) for config in configs]

        def get_evaluations(configs, queue):
            for config in configs:
                result0 = self.fidelity_functions[0](config)
                result1 = self.fidelity_functions[1](config)
                obj0 = result0['objs'][0]
                obj1 = result1['objs'][0]
                queue.put((obj0, obj1))

        for i in range(threads):
            configs_num_per_thread = math.ceil(size / threads)
            thread_configs = configs[i*configs_num_per_thread : (i+1)*configs_num_per_thread]

            p = multiprocessing.Process(target=get_evaluations, args=(thread_configs, queues[i]))
            pool.append(p)
            p.start()

        for p in pool:
            p.join()

        evaluations = []

        for i in range(threads):
            while not queues[i].empty():
                evaluations.append(queues[i].get())
        
        kt, pairs = evaluator.test_KT(evaluations)
        return kt, points, evaluations, pairs

        

        
    def config_2_design_fixed_model_parallel(self, config, choose_model):
        design_point = dict(config)
        model_para = self.fixed_model_parameters[choose_model]
        factors_of_batch = self.factors_of_batch_size_all_models[choose_model]

        wafer_num = math.ceil(self.num_of_gpus[choose_model] * 312 * 1000 / (design_point['core_mac_num'] * design_point['core_array_h'] * design_point['core_array_w'] * design_point['reticle_array_h'] * design_point['reticle_array_w'] * 2))
        model_para['tensor_parallel_size'] = 1
        model_para['model_parallel_size'] = 1

        closest = min(factors_of_batch, key=lambda x: abs(x - model_para['mini_batch_size'] // wafer_num))
        model_para['micro_batch_size'] = closest
        model_para['data_parallel_size'] = wafer_num

        return design_point, model_para


    def evaluator_factory(self, use_high_fidelity=True, metrics=['throughput'], use_mean=False, add_noise=False):
        _use_high_fidelity = use_high_fidelity
        _metrics = metrics
        def evaluation_func(config: sp.Configuration):
            nonlocal _use_high_fidelity
            nonlocal _metrics
            try:
                design_point, model_parameters = self.config_2_design_model(config)
                objs = []
                for i in range(len(metrics)):
                    if _metrics[i] == 'throughput':
                        objs.append(-api.evaluate_design_point(design_point=design_point, model_parameters=model_parameters, metric=_metrics[i], use_high_fidelity=_use_high_fidelity))
                    elif _metrics[i] == 'power':
                        power = api.evaluate_design_point(design_point=design_point, model_parameters=model_parameters, metric=_metrics[i], use_high_fidelity=True) / 100
                        # if power > 300:
                        #     raise ValueError('Power > 30000!')
                        objs.append(power)
                    elif _metrics[i] == 'latency':
                        objs.append(api.evaluate_design_point(design_point=design_point, model_parameters=model_parameters, metric=_metrics[i], use_high_fidelity=_use_high_fidelity))
            except Exception as e:
                print('ERROR in evaluation_func! ', e)
                objs = []
                for i in range(len(metrics)):
                    objs.append(1e10)
            result = dict()
            result['objs'] = objs
            return result
        
        def evaluation_func_mean(config: sp.Configuration):
            nonlocal _use_high_fidelity
            nonlocal _metrics
            try:
                mean_objs = []
                for cm in range(14):
                    design_point, model_parameters = self.config_2_design_fixed_model_parallel(config, cm)
                    objs = []
                    for i in range(len(metrics)):
                        if _metrics[i] == 'throughput':
                            throughput = -api.evaluate_design_point(design_point=design_point, model_parameters=model_parameters, metric=_metrics[i], use_high_fidelity=_use_high_fidelity)*self.throughput_weight[cm]
                            if add_noise:
                                throughput *= random.normalvariate(1, 0.05)
                            objs.append(throughput)
                        elif _metrics[i] == 'power':
                            power = api.evaluate_design_point(design_point=design_point, model_parameters=model_parameters, metric=_metrics[i], use_high_fidelity=True) / 100
                            objs.append(power)
                        elif _metrics[i] == 'latency':
                            objs.append(api.evaluate_design_point(design_point=design_point, model_parameters=model_parameters, metric=_metrics[i], use_high_fidelity=_use_high_fidelity))
                    
                    if len(mean_objs) == 0:
                        mean_objs = objs
                    else:
                        for i in range(len(objs)):
                            mean_objs[i] += objs[i]
                for i in range(len(mean_objs)):
                    mean_objs[i] = mean_objs[i] / 16
            except Exception as e:
                print('ERROR in evaluation_func! ', e)
                mean_objs = []
                for i in range(len(metrics)):
                    mean_objs.append(1e10)
            result = dict()
            result['objs'] = mean_objs
            return result
        

        if use_mean == False:
            return evaluation_func
        else:
            return evaluation_func_mean

    def factors(self, n):
        if n in self.factors_.keys():
            return self.factors_[n]
        factors = []
        # factors.append(1)
        for i in range(1, int(math.sqrt(n)) + 1):
            if n % i == 0:
                factors.append(i)
                if i != n // i:
                    factors.append(n // i)
        factors = sorted(factors)
        self.factors_[n] = copy.deepcopy(factors)
        return factors
    

    def initialize_throughput_weight(self):
        self.throughput_weight = []

        for cm in range(16):
            num_heads = self.fixed_model_parameters[cm]['attention_heads']
            d_model = self.fixed_model_parameters[cm]['hidden_size']
            key_size = d_model // num_heads
            seq_len = self.fixed_model_parameters[cm]['sequence_length']
            num_layers = self.fixed_model_parameters[cm]['number_of_layers']

            position_embeddings = 2 * d_model * seq_len
            kqv_proj = num_layers * 2 * 3 * seq_len * d_model * (key_size * num_heads)
            kq_logits = num_layers * 2 * (seq_len**2) * (key_size * num_heads)
            softmax = num_layers * 3 * (key_size * num_heads) * (seq_len**2)
            softmax_q_red = num_layers * (seq_len**2) * (key_size * num_heads)
            final_linear = num_layers * 2 * seq_len * (key_size * num_heads) * d_model
            sm_v_dot = num_layers * 2 * (seq_len**2) * (key_size * num_heads)
            attention_blocks = kqv_proj + kq_logits + softmax + softmax_q_red + sm_v_dot + final_linear

            dense_blocks = num_layers * 16 * seq_len * (d_model**2)
            
            layer_norm_flops = num_layers * 2 * 7 * (seq_len * d_model)

            gelu_flops = num_layers * 20 * 4 * (seq_len * d_model)
            total_flops_per_step = position_embeddings + layer_norm_flops + attention_blocks + dense_blocks + gelu_flops

            total_flops_per_step = total_flops_per_step * 3
            total_flops_per_step -= position_embeddings

            self.throughput_weight.append((total_flops_per_step * 400) / (self.num_of_gpus[cm] * 312 * 1e12))


    def low_fidelity_mapping_2_high_fidelity(self, histories, strategy='multi_fidelity', iterations=50):
        _sum = [] # (run_times, fidelity, max_runs, (design, model, objectives)) -> (fidelity, max_runs, objectives), average on run_times

        _sum_high = [] # we save the corresponding evaluation points in high fidelity objective function in _sum_high

        high_func = self.evaluator_factory(use_high_fidelity=True, metrics=self.metrics, use_mean=True)

        for run_time in range(len(histories)):
            for point in range(min(len(histories[0]), iterations)):
                # here we need to get the evaluation in high fidelity function
                config = Configuration(self.space, histories[run_time][point][0])
                obj_high = high_func(config)
                for i in range(len(histories[run_time][point][-1])):
                    if i == 1:
                        obj_high[i] = min(300, obj_high[i])
                        histories[run_time][point][-1][i] = min(300, histories[run_time][point][-1][i])
                    elif i == 0:
                        obj_high[i] = min(0, obj_high[i])
                        histories[run_time][point][-1][i] = min(0, histories[run_time][point][-1][i])
                _sum.append(histories[run_time][point][-1])
                _sum_high.append(obj_high)
        _sum = np.array(_sum)
        _sum = _sum.reshape((len(histories), min(len(histories[0]), iterations), len(histories[0][0][-1])))

        _sum_high = np.array(_sum_high)
        _sum_high = _sum_high.reshape((len(histories), min(len(histories[0]), iterations), len(histories[0][0][-1])))
        
        # here, the shape of _sum and _sum_high is (run_times, max_runs, objectives)
        # we need to find the pareto points in _sum, and then get the corresponding points in _sum_high
        
        from openbox.utils.multi_objective import NondominatedPartitioning
        # hv_mean = []
        # pareto_fronts = []
        # for i in range(_sum.shape[0]):
        #     hv = []
        #     for j in range(_sum.shape[1]):
        #         partition = NondominatedPartitioning(_sum.shape[2], _sum[i, :j+1, :])
        #         hv.append(partition.compute_hypervolume([0, 300]))
        #     hv_mean.append(hv)
        #     partition = NondominatedPartitioning(_sum.shape[2], _sum[i, :, :])
        #     pareto_fronts.append(partition.pareto_Y)

        # hv_mean = np.array(hv_mean)
        # hv = copy.deepcopy(hv_mean)
        # hv_max = np.max(hv, axis=0)
        # hv_min = np.min(hv, axis=0)
        # hv_mean = hv_mean.mean(axis=0)

        # return _sum, hv_mean, pareto_fronts, hv_max, hv_min
        




if __name__ == '__main__':
    dse_runner = DSE()
    print(dse_runner.throughput_weight)

