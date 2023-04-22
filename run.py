from utility import dse, parse_pickle
import pickle
import argparse

# parser = argparse.ArgumentParser(description="Process some integers.")
# parser.add_argument("-rn", "--run_name", type=str, default='random_test_cm0_rt10_mr50', help="name of this run")
# parser.add_argument("-st", "--strategy", type=str, default='random', help="strategy type, you can choose from multi_fidelity, multi_fidelity and random")
# parser.add_argument("-cm", "--choose_model", type=int, default=0, help="the model you choose")
# parser.add_argument("-rt", "--run_times", type=int, default=10, help="times of repeated experiments")
# parser.add_argument("-mr", "--max_runs", type=int, default=50, help="times of repeated experiments")
# parser.add_argument("-ps", "--pre_set", type=int, default=-1, help="-1 means don't use pre-set values, otherwise we use a set of values which are pre set, we use it for quick test")

metrics = ['throughput', 'power']

# args = parser.parse_args()
# if args.pre_set == -1:
#     dse_runner = dse.DSE(choose_model=args.choose_model, strategy=args.strategy, run_name=args.run_name, run_times=args.run_times, max_runs=args.max_runs, metrics=metrics)
# elif args.pre_set == 0:  # random
#     dse_runner = dse.DSE(choose_model=args.choose_model, strategy='random', run_name='random_test_cm{}_rt{}_mr{}'.format(args.choose_model, args.run_times, args.max_runs), run_times=args.run_times, max_runs=args.max_runs, metrics=metrics)
# elif args.pre_set == 1:  # single fidelity
#     dse_runner = dse.DSE(choose_model=args.choose_model, strategy='single_fidelity', run_name='single_fidelity_test_cm{}_rt{}_mr{}'.format(args.choose_model, args.run_times, args.max_runs), run_times=args.run_times, max_runs=args.max_runs, metrics=metrics)
# elif args.pre_set == 2:  # multi-fidelity
#     dse_runner = dse.DSE(choose_model=args.choose_model, strategy='multi_fidelity', run_name='multi_fidelity_test_cm{}_rt{}_mr{}'.format(args.choose_model, args.run_times, args.max_runs), run_times=args.run_times, max_runs=args.max_runs, metrics=metrics)
# dse_runner.run()

rt = 30
mr = 50

for cm in range(1):
    dse_runner = dse.DSE(choose_model=cm, strategy='random', run_name='random_final_cm{}_rt{}_mr{}'.format(cm, rt, mr), run_times=rt, max_runs=mr, metrics=metrics)
    dse_runner.run()
    del dse_runner

    # dse_runner = dse.DSE(choose_model=cm, strategy='single_fidelity', run_name='single_fidelity_final_cm{}_rt{}_mr{}'.format(cm, rt, mr), run_times=rt, max_runs=mr, metrics=metrics)
    # dse_runner.run()
    # del dse_runner

    # dse_runner = dse.DSE(choose_model=cm, strategy='multi_fidelity', run_name='multi_fidelity_final_cm{}_rt{}_mr{}'.format(cm, rt, mr+100), run_times=rt, max_runs=mr+100, metrics=metrics)
    # dse_runner.run()
    # del dse_runner

###############################################

# run_name = 'multi_fidelity_v0.2'
# strategy = 'multi_fidelity'


# run_name = 'single_fidelity_v0.2'
# strategy = 'single_fidelity'
# dse.multi_fidelity_double_circulation_search(run_times=10, model_num=1, max_runs=50, run_name=run_name, strategy=strategy)

# run_name = 'random_v0.2'
# strategy = 'random'
# dse.multi_fidelity_double_circulation_search(run_times=10, model_num=1, max_runs=50, run_name=run_name, strategy=strategy)
