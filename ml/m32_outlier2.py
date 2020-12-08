import numpy as np

a = np.array(
    [[1,2,3,4, 10000, 6,7,5000,90,100],
    [10000,20000,-80000,40000,50000,60000,70000,8,90000,100000],
    ])

def outliers(data_out) :
    data_out = data_out.T # 뒤집어서 빼기 편하게
    del_index = np.array([],dtype=np.int)
    for cnt in range(data_out.shape[1]):
        column_data = data_out[:,cnt]
        quartile_1, quartile_3 = np.percentile(column_data, [25,75]) #데이터의 25%, 75%지점
        iqr = quartile_3 - quartile_1
        lower_bound = quartile_1 - (iqr * 1.5)
        upper_bound = quartile_3 + (iqr * 1.5)
        outlier_index = np.where((column_data>upper_bound)|(column_data<lower_bound))
        del_index = np.append(del_index, outlier_index) # 인덱스를 모아서
    data_out = np.delete(data_out, del_index, axis=0) # 모은 인덱스를 빼라
    data_out = data_out.T # 다시 뒤집자
    return data_out


b = outliers(a)
print("이상치의 위치 : ", b) 