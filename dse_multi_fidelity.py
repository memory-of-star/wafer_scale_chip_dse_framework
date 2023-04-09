from openbox import Optimizer, sp
from ConfigSpace import ConfigurationSpace, Configuration
import api2
import api1

# build search space
design_points = []
with open('design_points.list', 'r') as f:
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

choose_model = 0

test_model_parameters = [{
        "attention_heads": 24,
        "hidden_size": 2304,
        "sequence_length": 2048,
        "number_of_layers": 24,
        "mini_batch_size": 512,
        "micro_batch_size": 512,
        "data_parallel_size": 8,
        "model_parallel_size": 1,
        "tensor_parallel_size": 1,
    },
    {
        "attention_heads": 32,
        "hidden_size": 3072,
        "sequence_length": 2048,
        "number_of_layers": 30,
        "mini_batch_size": 512,
        "micro_batch_size": 512,
        "data_parallel_size": 8,
        "model_parallel_size": 1,
        "tensor_parallel_size": 2,
    },
    {
        "attention_heads": 32,
        "hidden_size": 4096,
        "sequence_length": 2048,
        "number_of_layers": 36,
        "mini_batch_size": 512,
        "micro_batch_size": 512,
        "data_parallel_size": 8,
        "model_parallel_size": 1,
        "tensor_parallel_size": 4,
    },
    {
        "attention_heads": 48,
        "hidden_size": 6144,
        "sequence_length": 2048,
        "number_of_layers": 40,
        "mini_batch_size": 1024,
        "micro_batch_size": 1024,
        "data_parallel_size": 8,
        "model_parallel_size": 1,
        "tensor_parallel_size": 8,
    },
    {
        "attention_heads": 64,
        "hidden_size": 8192,
        "sequence_length": 2048,
        "number_of_layers": 48,
        "mini_batch_size": 1536,
        "micro_batch_size": 1536,
        "data_parallel_size": 8,
        "model_parallel_size": 2,
        "tensor_parallel_size": 8,
    },
    {
        "attention_heads": 80,
        "hidden_size": 10240,
        "sequence_length": 2048,
        "number_of_layers": 60,
        "mini_batch_size": 1792,
        "micro_batch_size": 1792,
        "data_parallel_size": 8,
        "model_parallel_size": 4,
        "tensor_parallel_size": 8,
    },
    {
        "attention_heads": 96,
        "hidden_size": 12288,
        "sequence_length": 2048,
        "number_of_layers": 80,
        "mini_batch_size": 2304,
        "micro_batch_size": 2304,
        "data_parallel_size": 8,
        "model_parallel_size": 8,
        "tensor_parallel_size": 8,
    },
    {
        "attention_heads": 128,
        "hidden_size": 16384,
        "sequence_length": 2048,
        "number_of_layers": 96,
        "mini_batch_size": 2160,
        "micro_batch_size": 2160,
        "data_parallel_size": 8,
        "model_parallel_size": 16,
        "tensor_parallel_size": 8,
    },
    {
        "attention_heads": 128,
        "hidden_size": 20480,
        "sequence_length": 2048,
        "number_of_layers": 105,
        "mini_batch_size": 2520,
        "micro_batch_size": 2520,
        "data_parallel_size": 8,
        "model_parallel_size": 35,
        "tensor_parallel_size": 8,
    },
    {
        "attention_heads": 160,
        "hidden_size": 25600,
        "sequence_length": 2048,
        "number_of_layers": 128,
        "mini_batch_size": 3072,
        "micro_batch_size": 3072,
        "data_parallel_size": 8,
        "model_parallel_size": 64,
        "tensor_parallel_size": 8,
    },
    {
        "attention_heads": 256,
        "hidden_size": 32000,
        "sequence_length": 2048,
        "number_of_layers": 192,
        "mini_batch_size": 3072,
        "micro_batch_size": 3072,
        "data_parallel_size": 8,
        "model_parallel_size": 64,
        "tensor_parallel_size": 8,
    },
    {
        "attention_heads": 432,
        "hidden_size": 43200,
        "sequence_length": 2048,
        "number_of_layers": 192,
        "mini_batch_size": 5500,
        "micro_batch_size": 5500,
        "data_parallel_size": 8,
        "model_parallel_size": 64,
        "tensor_parallel_size": 8,
    },
    {
        "attention_heads": 512,
        "hidden_size": 66560,
        "sequence_length": 2048,
        "number_of_layers": 195,
        "mini_batch_size": 10000,
        "micro_batch_size": 10000,
        "data_parallel_size": 8,
        "model_parallel_size": 65,
        "tensor_parallel_size": 8,
    },
    {
        "attention_heads": 620,
        "hidden_size": 80600,
        "sequence_length": 2048,
        "number_of_layers": 240,
        "mini_batch_size": 15000,
        "micro_batch_size": 15000,
        "data_parallel_size": 8,
        "model_parallel_size": 80,
        "tensor_parallel_size": 8,
    },
    {
        "attention_heads": 850,
        "hidden_size": 102000,
        "sequence_length": 2048,
        "number_of_layers": 270,
        "mini_batch_size": 20000,
        "micro_batch_size": 20000,
        "data_parallel_size": 8,
        "model_parallel_size": 90,
        "tensor_parallel_size": 8,
    },
    {
        "attention_heads": 1024,
        "hidden_size": 158720,
        "sequence_length": 2048,
        "number_of_layers": 315,
        "mini_batch_size": 20000,
        "micro_batch_size": 20000,
        "data_parallel_size": 8,
        "model_parallel_size": 105,
        "tensor_parallel_size": 8,
    },
    ]

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

def func(config: sp.Configuration):
    try:
        dic = dict(config)
        lst = tuple(dict(config).values())
        design_point = design_vec2design_dic(lst[1:])
        if lst[0] == 1:
            prediction = api1.evaluate_design_point(design_point=design_point, model_parameters=test_model_parameters[choose_model])
        elif lst[0] == 2:
            prediction = api2.evaluate_design_point(design_point=design_point, model_parameters=test_model_parameters[choose_model])
        else:
            raise ValueError
    except:
        prediction = -1e10
    result = dict()
    result['objs'] = [-prediction]
    return result

    

def process(points_dict, output_path, threshold = 0, queue = None, **optimizer_kwargs):
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
    history_lst = opt.mf_run()
    
    for i, history in enumerate(history_lst):
        if i != 0:
            y_flatten = [i[0] for i in history.objectives]
            configs_lst = history.configurations

            with open(output_path + '_fidelity{}_history'.format(i), 'w+') as f:
                for i in range(len(y_flatten)):
                    f.write(str(configs_lst[i]))
                    f.write('\n')
                    f.write(str(y_flatten[i]))
                    f.write('\n')
    

import copy

var_names = ['var{:02}'.format(i) for i in range(len(design_space))]
points_dic = [{k:v for k,v in zip(var_names, design_points[i])} for i in range(len(design_points))]

points_dic2 = []
for i in range(len(points_dic)):
    dic = copy.deepcopy(points_dic[i])
    dic['fidelity'] = 1
    points_dic[i]['fidelity'] = 2
    points_dic2.append(dic)
    points_dic2.append(points_dic[i])


from multiprocessing import Process

for i in range(0, 1):
    optimizer_kwargs = {
                    'objective_function':func,
                    'num_objs':1,
                    'num_constraints':0,
                    'max_runs':20,
                    'surrogate_type':'gp',
                    'acq_optimizer_type':'true_random',
                    'initial_runs':6,
                    'init_strategy':'random',
                    'time_limit_per_trial':1000,
                    'task_id':'moc',
                    'acq_type':'mfei',
                    'advisor_type':'mf_advisor'
                    }
    choose_model = i
    p = Process(target=process, args=(points_dic2, 'result/log_multifidelity_model{}'.format(i), -1e11), kwargs=optimizer_kwargs)
    p.start()

