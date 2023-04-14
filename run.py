from utility import dse, parse_pickle
import pickle
import argparse

parser = argparse.ArgumentParser(description="Process some integers.")
parser.add_argument("-rn", "--run_name", type=str, help="name of this run")
parser.add_argument("-st", "--strategy", type=str, help="strategy type, you can choose from multi_fidelity, multi_fidelity and random")

args = parser.parse_args()

# run_name = 'multi_fidelity_v0.2'
# strategy = 'multi_fidelity'
dse.multi_fidelity_double_circulation_search(run_times=10, model_num=1, max_runs=20, multi_objective=True, run_name=args.run_name, strategy=args.strategy)

# run_name = 'single_fidelity_v0.2'
# strategy = 'single_fidelity'
# dse.multi_fidelity_double_circulation_search(run_times=10, model_num=1, max_runs=50, run_name=run_name, strategy=strategy)

# run_name = 'random_v0.2'
# strategy = 'random'
# dse.multi_fidelity_double_circulation_search(run_times=10, model_num=1, max_runs=50, run_name=run_name, strategy=strategy)
