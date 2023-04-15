import pickle
from openbox import Optimizer, sp, History
from ConfigSpace import Configuration
import test_model_parameters
import copy
import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dse4wse/test/dse'))

import api2
import api1

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

def parse_pickle(run_name='try', choose_model=0, strategy='multi_fidelity'):
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'result/pickle', run_name+'.pickle')
    with open(path, 'rb') as f: # historis shape: max_runs, 3(e.g. None, fidelity1, fidelity2), length of fidelity 1 history (or 2) 
        histories = pickle.load(f)

    # print(len(histories[0][1]))

    if strategy == 'multi_fidelity':
        ret = []  # max_runs, 3(e.g. None, fidelity1, fidelity2), length of fidelity 1 history (or 2)
        for run_num in range(len(histories)):
            single_run = []  # 2(e.g. None, fidelity1, fidelity2), length of fidelity 1 history (or 2)
            for fidelity in range(2):
                single_fidelity = []  # length of fidelity 1 history (or 2)
                for i in range(len(histories[run_num][fidelity].observations)):
                    model_para = copy.deepcopy(test_model_parameters.test_model_parameters[choose_model])
                    model_para.update(histories[run_num][fidelity].observations[i].inner_config[0])
                    lst = tuple(dict(histories[run_num][fidelity].observations[i].config).values())
                    design_point = design_vec2design_dic(lst)
                    obj = histories[run_num][fidelity].observations[i].objectives[0]

                    single_fidelity.append((design_point, model_para, obj))
                single_run.append(single_fidelity)
            ret.append(single_run)
    elif strategy == 'single_fidelity' or strategy == 'random':
        ret = []  # max_runs, length of fidelity 1 history (or 2)
        for run_num in range(len(histories)):
            single_run = []  # length of fidelity 1 history (or 2)

            for i in range(len(histories[run_num].observations)):
                model_para = copy.deepcopy(test_model_parameters.test_model_parameters[choose_model])
                model_para.update(histories[run_num].observations[i].inner_config[0])
                lst = tuple(dict(histories[run_num].observations[i].config).values())
                design_point = design_vec2design_dic(lst)
                obj = histories[run_num].observations[i].objectives[0]

                single_run.append((design_point, model_para, obj))
            ret.append(single_run)

    return ret

