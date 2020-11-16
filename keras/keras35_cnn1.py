import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten

model = Sequential()
model.add(Conv2D(10, (2,2),input_shape=(10,10,1))) # -> 아웃풋이 9,9,10 (9 * 9가 10개(filter) )
model.add(Conv2D(5, (2,2), padding='same')) #아웃풋 (9,9,5) 
model.add(Conv2D(3 ,(3,3))) #아웃풋 (7,7,3) 인풋 - 커널 + 스트라이드 = 8 - 3 + 1 =  6
model.add(Conv2D(7 ,(2,2))) #아웃풋 (6,6,7) 

#차원이 줄어들지 않아서 Dense로 넘길 시 주의 필요

#filter: 출력 공간의 차원 = 다음 레이어에 넘길 노드의 갯수  = 노드 수 
#Kernel_size: 2D 컨볼 루션 창의 높이와 너비 Ex(2,2) 가로2 높이2의 16픽셀 -> 5,5 -> kernel(2,2) ->(4,4)됨
#strides: 컨볼 루션의 스트라이드를 지정하는 정수 또는 2 개의 정수로 구성된 튜플 / 보폭 픽셀간 넘어가는 단위  / 디폴트 =1
#pading: 경계 처리 방법 / same, vaild 2개 / 디폴트 vaild -> 가장자리의 데이터의 연산의 횟수가 중앙의 데이터의 연산 횟수와 차이나는 것 때문에 사용 (5,5) ->kernel(2,2) -> (5,5)
#   same: 출력 이미지 사이즈가 입력 이미지 사이즈와 동일합니다.
#   valid : 유효한 영역만 출력이 됩니다. 따라서 출력 이미지 사이즈는 입력 사이즈보다 작습니다
#input_shape: (rows, cols, channels) => 가로 세로 특성
# 입력모양: (배치사이즈, rows,cols,channels)  한번에 처리 할 이미지의 파일의 수 , 가로, 세로, 특성

#LSTM 파라미터
#unit:출력차원
#reurn_sequensces: 마지막 출력의 반환 여부 
#input_shape(timesteps=열, feature = 몇개 씩 연산할 지)
# timestes 1~ 100 일 때 1~10 2 ~ 11 즉 10일치 씩 

#Maxpooling2d
#스트라이드 없이 잘라서 가장 특성치가 높은 픽셀만 남기는 기법

model.add(MaxPooling2D()) # stride 없기 때문에 반으로 줄어듬 (3,3,7) 노드는 이전의 노드 가져옴
#Dense로 넘기려면 2차원으로 넘겨야 함 How! Flatten
model.add(Flatten()) # 3*3*7 = 63(노드 수 )
model.add(Dense(1))

model.summary()
# model.add(Conv2D(10, (2,2),input_shape=(10,10,1)))
#( 1 * 2 * 2 + 1 ) * 10 
# 파라미터의 수 = (채널 * 커널사이즈 * +1(바이어스) ) 나가는 노드