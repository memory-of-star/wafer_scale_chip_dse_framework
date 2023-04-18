from utility import dse, parse_pickle
import pickle
import argparse

models_num = 1
kt = []
for i in range(models_num):
    _kt, _points_lst, _evaluation_list, _pairs = dse.KT_evaluator(size=1000, choose_model=i, threads=20)
    kt.append(_kt)
    with open('discordant_pairs{}.pickle'.format(i), 'wb') as f:
        pickle.dump((_kt, _points_lst, _evaluation_list, _pairs), f)

print(kt)

###########

# for i in range(15):
#     with open('discordant_pairs{}.pickle'.format(i), 'rb') as f:
#         _kt, _points_lst, _evaluation_list, _pairs = pickle.load(f)
#         print(_kt)

# print(_kt)
# length = len(_pairs)
# sum1 = 0
# sum2 = 0

# for pair in _pairs:
#     sum1 += abs((_evaluation_list[pair[0]][0] - _evaluation_list[pair[1]][0]) / (_evaluation_list[pair[0]][0] + 1))
#     sum2 += abs((_evaluation_list[pair[0]][1] - _evaluation_list[pair[1]][1]) / (_evaluation_list[pair[0]][1] + 1))

# print(sum1)
# print(sum2)
# print(len(_pairs))


# import sys
# import os
# sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dse4wse/test/dse'))
# import api


# ret = dse.generate_legal_points(1)[0]
# high = api.evaluate_design_point(design_point=ret[0], model_parameters=ret[1], metric='throughput', use_high_fidelity=True)
# low = api.evaluate_design_point(design_point=ret[0], model_parameters=ret[1], metric='throughput', use_high_fidelity=False)

# print(high)
# print(low)