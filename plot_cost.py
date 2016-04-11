
import matplotlib.pyplot as plt
import numpy as np
import ipdb

f = open('cae_svhn_cost.txt', 'rb')
lines = [line.strip() for line in f.readlines()]
cae_cost = np.empty((len(lines)), dtype=np.float32)
for i, l in enumerate(lines):
    cae_cost[i] = float(l[5:])


f = open('spcae_svhn_cost.txt', 'rb')
lines = [line.strip() for line in f.readlines()]
spcae_cost = np.empty((len(lines)), dtype=np.float32)
for i, l in enumerate(lines):
    spcae_cost[i] = float(l[5:])

# ipdb.set_trace()
plt.plot(np.arange(cae_cost.shape[0]),cae_cost, 'b', label='CAE-SVHN')
plt.plot(np.arange(spcae_cost.shape[0]),spcae_cost, 'r', label='SSCAE-SVHN')
plt.title('learning convergence')
plt.legend()
plt.show()
