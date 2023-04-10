


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