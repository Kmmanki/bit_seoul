import numpy as np
import matplotlib.pyplot as plt


def relu(x):
    return np.maximum(0,x)

x = np.arange(-5,5,0.1)


y = relu(x)

plt.plot(x,y)
plt.grid()
plt.show()

#relu 친구들 찾기
'''
ELU a(e**x -1) ~ x 
Leaky Relu 0.1x ~ x
selu lamda a(e**x -1) ~ x 

'''
