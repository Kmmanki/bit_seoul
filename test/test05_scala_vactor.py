import numpy as np

arr = np.array([
    [
        [1,2,3],
        [4,5,6]
    ],[
        [7,8,9],
        [10,11,12]
    ]
    ])

print(arr.shape)
print(arr.T.shape)

arr2= np.array([1,2,3,4,5,6])
print(arr2.T)
print(arr2.T.shape)