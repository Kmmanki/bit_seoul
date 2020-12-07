from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
#Flatten 확인

docs = [
    "너무 재밌어요", "참 최고에요", '참 잘 만든 영화에요',
    '추천하고 싶은 영화입니다.', ' 한 번 더 보고 싶네요', '글쎄요',
    '별로에요','생각보다 지루해요','연기가 어색해요', ' 재미없어요'
    ,'너무 재미없다' , '참 재밌네요'
]

#긍정 1 , 부정 0
labels =np.array([1,1,1,1,1,0,0,0,0,0,0,1])

token = Tokenizer()
token.fit_on_texts(docs)
print(token.word_index)

x = token.texts_to_sequences(docs)
print(x)
#[[2, 3], [1, 4], [1, 5, 6, 7], [8, 9, 10], [11, 12, 13, 14, 15], [16], [17], [18, 19], [20, 21], [22], [2, 23, 24], [1, 25]] 가장 긴것에 맞춰 0을 추가 어디에? 앞에 -> 자연어는 시계열이라 뒤가 영향이 큼

from tensorflow.keras.preprocessing.sequence import pad_sequences
pad_x = pad_sequences(x, padding='pre')

print(pad_x)
'''
[[ 0  0  0  2  3]
 [ 0  0  0  1  4]
 [ 0  1  5  6  7]
 [ 0  0  8  9 10]
 [11 12 13 14 15]
 [ 0  0  0  0 16]
 [ 0  0  0  0 17]
 [ 0  0  0 18 19]
 [ 0  0  0 20 21]
 [ 0  0  0  0 22]
 [ 0  0  2 23 24]
 [ 0  0  0  1 25]]
'''

word_size = len(token.word_index) + 1

print("전체 토큰 사이즈 : ", word_size)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM,  Flatten
# from tensorflow.keras. import 

model = Sequential()
model.add(Embedding(25, 10)) #단어 사전의 개수, 아웃풋 노드의 수,  12,5의 5 = 가장 큰 문장의 cols
#10의 벡터로 줄여준다. 즉 25,10
model.add(Flatten()) # 넘어오는값이 None, None 10 이기에 Flatten할 수 없다
model.add(Dense(1, activation='sigmoid'))

model.summary()
'''
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
embedding (Embedding)        (None, 5, 10)             250       #원핫 인코딩을 백터화, 배터화된 단어사전의 개수
_________________________________________________________________
lstm (LSTM)                  (None, 32)                5504      
_________________________________________________________________
dense (Dense)                (None, 1)                 33
=================================================================
Total params: 5,787
Trainable params: 5,787
Non-trainable params: 0
_________________________________________________________________

_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
embedding (Embedding)        (None, None, 10)          250      # 인풋 랭스 안할 시 
_________________________________________________________________
lstm (LSTM)                  (None, 32)                5504
_________________________________________________________________
dense (Dense)                (None, 1)                 33
=================================================================
Total params: 5,787
Trainable params: 5,787
Non-trainable params: 0
'''

model.compile(optimizer='adam',
                loss= 'binary_crossentropy', metrics=['acc'])

model.fit(pad_x, labels, epochs= 300)

acc = model.evaluate(pad_x, labels)[1]
print('acc', acc)