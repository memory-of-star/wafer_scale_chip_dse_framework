from utility import dse, parse_pickle
import pickle

# dse.multi_fidelity_double_circulation_search(run_times=1, model_num=1, max_runs=20, run_name='try2', advisor_type='random')
# x = parse_pickle.parse_pickle(run_name='try2')

x = dse.generate_legal_points(200000)
with open('legal_points.pickle', 'wb') as f:
    pickle.dump(x, f)
