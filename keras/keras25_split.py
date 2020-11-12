import numpy as np 

dataset = np.array(range(1, 31))
size= int(int(len(dataset)) /2)

def split_x(seq, size):
    aaa = []

    # 11 - 5 =6 
    for i in range(len(seq)-  size +1): # 행 만드는 반복문 
        
        subset =seq[i : (i+size)] #dataset 에서 한칸씩 밀어나면서 size 갯수 만큼 꺼내기
        # [0 : 5] 1 2 3 4 5 
        # [1 : 6] 2 3 4 5 6  
        # [2 : 7] 3 4 5 6 7 
        # [2 : 8] 4 5 6 7 8 
        

        aaa.append([item for item in subset]) 
        #append(1 2 3 4 5) i = 0
        #append(2 3 4 5 6) i = 1
        #append(3 4 5 6 7) i = 2 
        #append(4 5 6 7 8) i = 3
        #append(5 6 7 8 9) i = 4
        #append(6 7 8 9 10) i = 5
    print(type(aaa))
    return np.array(aaa)

datasets = split_x(dataset, size)
print("===========================================================")
print(datasets)