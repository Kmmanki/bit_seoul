import numpy as np

def outlier(data_out):
    quaritile_1, quaritile_3 = np.percentile(data_out, [25, 75])
    print('1사분위 : ', quaritile_1) # 1/4 끝나는 지점
    print('3사분위 : ', quaritile_3) # 3/4 끝나는지점
    iqr = quaritile_3 - quaritile_1 
    lower_bound = quaritile_1 - (iqr * 1.5)
    upper_bound = quaritile_3 + (iqr * 1.5)
    return np.where((data_out > upper_bound) |  (data_out < lower_bound))


a = np.array([1,2,3,4,10000,6,7,5000,90,100])
b = outlier(a)
print('이상치의 위치', b)

#과제 스칼라가 아닌 백터의 상황에서 이상치 제거 