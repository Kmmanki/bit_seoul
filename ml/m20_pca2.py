import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets import load_diabetes

datasets = load_diabetes()

x = datasets.data
y = datasets.target

print(x.shape)
print(y.shape)

pca = PCA()
x2d = pca.fit(x)
comsum = np.cumsum(pca.explained_variance_ratio_) # -> 누적합
print(comsum)

d = np.argmax(comsum >= 0.95)+1
print(comsum >= 0.95)
print(d)

import matplotlib.pyplot as plt

plt.plot(comsum)
plt.grid()
plt.show()