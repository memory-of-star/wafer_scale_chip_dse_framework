from utility import dse, parse_pickle
import pickle
import argparse

models_num = 16
kt = []
for i in range(models_num):
    kt.append(dse.KT_evaluator(size=1000, choose_model=i))

print(kt)