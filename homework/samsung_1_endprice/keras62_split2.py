import numpy as np

# data = []
# for i in range(100):
#     day = []
#     day.append(i)
#     day.append(i)
#     day.append(i)
#     day.append(i)
#     data.append(day)

# data = np.array(data)
# # print(data)

# # print(data)




def spli(data, size):
    result = [] 
    for i in range(data.shape[0]):
        if i+ 5 <  data.shape[0]:
            result.append(data[i: i+size])
        else:
            return np.array(result)
    return np.array(result)


# tmp = spli(data, 5)
# print(tmp)
# print(tmp.shape)
