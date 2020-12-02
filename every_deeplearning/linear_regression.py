import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = [[2,81],[4,93],[6,91],[8,97]]

x = [i[0] for i in data]
y = [i[1] for i in data]

# print(x)
# print(y)

# plt.figure(figsize=(8,5))
# plt.scatter(x,y)
# plt.show()

x_data = np.array(x)
y_data = np.array(y)

w = 0
b = 0

learing_rage = 0.00005 #학습률이 너무 크거나 작으면 발산 
ehpochs = 2001

for i in range(ehpochs):
    y_predict = w * x_data + b
    error = y_data - y_predict

    w_diff = -(2/len(x_data) * sum(x_data * (error))) #부호 바꾸면서 
    b_diff = -(2/len(x_data) * sum((error)))
    print(w_diff)
    print(b_diff)

    w = w - learing_rage * w_diff
    b = b - learing_rage * b_diff

    if i %100 == 0:
        print("epoch=%.f, 기울기= %.04f, 절편=%.04f" % (i , w, b))

print('=================================================')
y_predict = w * x_data + b
print(y_predict)
plt.scatter(x,y)
plt.plot(y_predict)
# plt.plot([min(x_data), max(x_data)], [min(y_predict), max(y_predict)])
plt.show()