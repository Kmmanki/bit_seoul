#1. data
import numpy as np
 

x = np.array([range(1, 101), range(311, 411), range(100)])
y = np.array((range(101, 201), range(711, 811), range(100)))

print(x)
print(x.shape) # (3,100)

#(100,3)의 형태로 변환을 시켜라 

print(x.transpose())
print(x.transpose().shape)

