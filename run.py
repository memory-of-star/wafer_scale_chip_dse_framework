from utility import dse, parse_pickle
import pickle
import argparse

parser = argparse.ArgumentParser(description="Process some integers.")
parser.add_argument("-rn", "--run_name", type=str, default='default', help="name of this run")
parser.add_argument("-st", "--strategy", type=str, default='multi_fidelity', help="strategy type, you can choose from multi_fidelity, multi_fidelity and random")
parser.add_argument("-cm", "--choose_model", type=int, default=0, help="the model you choose")
parser.add_argument("-rt", "--run_times", type=int, default=10, help="times of repeated experiments")
parser.add_argument("-mr", "--max_runs", type=int, default=50, help="times of repeated experiments")

args = parser.parse_args()

# run_name = 'multi_fidelity_v0.2'
# strategy = 'multi_fidelity'
dse.multi_fidelity_double_circulation_search(choose_model=args.choose_model, run_times=args.run_times, max_runs=args.max_runs, multi_objective=False, run_name=args.run_name, strategy=args.strategy)

# run_name = 'single_fidelity_v0.2'
# strategy = 'single_fidelity'
# dse.multi_fidelity_double_circulation_search(run_times=10, model_num=1, max_runs=50, run_name=run_name, strategy=strategy)

# run_name = 'random_v0.2'
# strategy = 'random'
# dse.multi_fidelity_double_circulation_search(run_times=10, model_num=1, max_runs=50, run_name=run_name, strategy=strategy)
