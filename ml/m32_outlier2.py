import numpy as np

def outliers(data_out) :
    list=[]
    for i in range(len(data_out)) :
        quartile_1, quartile_3=np.percentile(data_out[i], [25, 75]) 
        print(i,"번째 인덱스의 1사분위 :", quartile_1) 
        print(i,"번째 인덱스의 3사분위 :", quartile_3) 
        iqr = quartile_3 - quartile_1 
        lower_bound = quartile_1 - (iqr * 1.5)
        upper_bound = quartile_3 + (iqr * 1.5)
        list.append(np.where(([data_out[i]]>upper_bound) | (data_out[i]<lower_bound)))

    return list

a = np.array([[1,2,3,4,10000], [6, 7,5000, 90, 100]])

b = outliers(a)
print("이상치의 위치 : ", b) 