import numpy as np

x = np.array([range(10), range(10), range(10)])

x = x.T
# print(x)
tmp =np.array(list(x)[:5])
print(tmp)