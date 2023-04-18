from utility import dse, parse_pickle
import pickle
import argparse

models_num = 1
kt = []
for i in range(models_num):
    _kt, _points_lst, _evaluation_list, _pairs = dse.KT_evaluator(size=1000, choose_model=0, threads=20)
    kt.append(_kt)
    with open('discordant_pairs{}.pickle'.format(i), 'wb') as f:
        pickle.dump((_kt, _points_lst, _evaluation_list, _pairs), f)

print(kt)

# with open('discordant_pairs.pickle', 'rb') as f:
#     _kt, _points_lst, _evaluation_list, _pairs = pickle.load(f)

# print(_kt)
# length = len(_pairs)
# sum1 = 0
# sum2 = 0

# for pair in _pairs:
#     sum1 += abs((_evaluation_list[pair[0]][0] - _evaluation_list[pair[1]][0]) / _evaluation_list[pair[0]][0])
