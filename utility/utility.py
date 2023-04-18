def factors(n):
    factors = [] 
    factors.append(1)
    for i in range(2, n + 1):
        if n % i == 0:
            factors.append(i)
    return factors 

def parse_histories_full_space(histories, choose_model, strategy='multi_fidelity'):
    if strategy == 'multi_fidelity':
        ret = []  # max_runs, 3(e.g. None, fidelity1, fidelity2), length of fidelity 1 history (or 2)
        for run_num in range(len(histories)):
            single_run = []  # 2(e.g. None, fidelity1, fidelity2), length of fidelity 1 history (or 2)
            for fidelity in range(2):
                single_fidelity = []  # length of fidelity 1 history (or 2)
                for i in range(len(histories[run_num][fidelity].observations)):
                    design_point = dict(histories[run_num][fidelity].observations[i].config)
                    model_para = copy.deepcopy(test_model_parameters.test_model_parameters[choose_model])
                    model_para["micro_batch_size"] = design_point.pop("micro_batch_size")
                    model_para["data_parallel_size"] = design_point.pop("data_parallel_size")
                    model_para["model_parallel_size"] = design_point.pop("model_parallel_size")
                    model_para["tensor_parallel_size"] = design_point.pop("tensor_parallel_size")
                    model_para["num_reticle_per_model_chunk"] = design_point.pop("num_reticle_per_model_chunk")
                    model_para['weight_streaming'] = design_point.pop("weight_streaming")

                    obj = histories[run_num][fidelity].observations[i].objectives[0]

                    single_fidelity.append((design_point, model_para, obj))
                single_run.append(single_fidelity)
            ret.append(single_run)
    elif strategy == 'single_fidelity' or strategy == 'random':
        ret = []  # max_runs, length of fidelity 1 history (or 2)
        for run_num in range(len(histories)):
            single_run = []  # length of fidelity 1 history (or 2)

            for i in range(len(histories[run_num].observations)):
                design_point = dict(histories[run_num].observations[i].config)
                model_para = copy.deepcopy(test_model_parameters.test_model_parameters[choose_model])
                model_para["micro_batch_size"] = design_point.pop("micro_batch_size")
                model_para["data_parallel_size"] = design_point.pop("data_parallel_size")
                model_para["model_parallel_size"] = design_point.pop("model_parallel_size")
                model_para["tensor_parallel_size"] = design_point.pop("tensor_parallel_size")
                model_para["num_reticle_per_model_chunk"] = design_point.pop("num_reticle_per_model_chunk")
                model_para['weight_streaming'] = design_point.pop("weight_streaming")

                obj = histories[run_num].observations[i].objectives[0]

                single_run.append((design_point, model_para, obj))
            ret.append(single_run)

    return ret