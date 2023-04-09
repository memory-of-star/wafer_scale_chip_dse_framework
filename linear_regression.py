from openbox import Optimizer, sp
from ConfigSpace import ConfigurationSpace, Configuration
import api2
import api1


from scipy import stats

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
        "micro_batch_size": 1, # 1 - mini batch
        "data_parallel_size": 1, # 1 - wafer number
        "model_parallel_size": 1, # 1 - number of layers, number of layers % x == 0
        "tensor_parallel_size": 1, # 1 - attention_heads, % == 0
        "num_reticle_per_pipeline_stage": 1, # 1 - wafer number * reticle per wafer / (model para * tensor para)
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


import random
from multiprocessing import Process, Queue
import os
import pandas as pd

# df = pd.read_excel(os.path.join(os.path.dirname(os.path.abspath(__file__)), "design_points.xlsx"))
# design_point = df.loc[10].to_dict()

# x = api1.evaluate_design_point(design_point=design_point, model_parameters=test_model_parameters[0])

# print(x)




# print(len(design_points))
# print(design_space)
# print(design_points)


queue = Queue()

def task(queue):
    points = random.sample(design_points, 20)
    for i in points:
        design_point = design_vec2design_dic(i)
        v1 = api1.evaluate_design_point(design_point=design_point, model_parameters=test_model_parameters[0], metric='throughput')
        v2 = api2.evaluate_design_point(design_point=design_point, model_parameters=test_model_parameters[0], metric='throughput')
        print("############ ", v1, "  ", v2, " ##################")
        queue.put((v1, v2))

pool = []

for i in range(3):
    p = Process(target=task, args=(queue, ))
    pool.append(p)
    p.start()

for p in pool:
    p.join()

x, y = [], []

while not queue.empty():
    point = queue.get()
    x.append(point[0])
    y.append(point[1])

result = stats.linregress(x, y)
print(result.slope) # 查看回归系数，即2.0
print(result.intercept) # 查看截距，即0.0
print(result.rvalue) # 查看相关系数，即1.0
print(result.pvalue) # 查看p值，即0.0
print(result.stderr) # 查看标准误差，即0.0
