import pickle
from openbox import Optimizer, sp, History
from ConfigSpace import Configuration

with open('./result/pickle/test.pickle', 'rb') as f:
    histories_random, histories_gp = pickle.load(f)

print(len(histories_gp))

print(histories_gp[0].objectives)
print(histories_gp[1].objectives)
print(histories_gp[2].objectives)

print(histories_random[0].objectives)
print(histories_random[1].objectives)
print(histories_random[2].objectives)