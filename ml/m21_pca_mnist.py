import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

(x_train, _), (x_test, _) = mnist.load_data()

x = np.append(x_train, x_test, axis=0)
x = x.reshape(x.shape[0], x.shape[1] *  x.shape[2])

print(x.shape)

pca = PCA()
pca.fit(x)
cumsum = np.cumsum(pca.explained_variance_ratio_)

d = np.argmax(cumsum > 1) +1 

print(cumsum >= 0.95)
print(d)


plt.plot(cumsum)
plt.grid()
plt.show()

pca = PCA(n_components=d+30)
pca.fit_transform(x)
sum1 =sum( pca.explained_variance_ratio_)
print(sum1)