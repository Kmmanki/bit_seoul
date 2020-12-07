# 12-04

키워드 

Redue, 전이학습, embedding, tokenizer, pad_sequensce

<br>

------

<br>



## Reduce Learning Rate

keras 74

fit의 callback으로 개선이 없다면 러닝레이트를 감소시켜준다.





```
earlyStopping = EarlyStopping(monitor='loss', 
                            patience=5, mode='min')

#3번동안 개선이 없으면 lr을 50% 감축, 더 지나서 5번 없으면 Es에 의해 종료
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=3,
                             factor=0.5, verbose=1)

hist = model.fit(x_train, y_train, validation_split=0.2, epochs=30, batch_size=128, verbose=1, callbacks=[earlyStopping, ck])

```



<br>

------

<br>

## 전이학습

keras76



```
from tensorflow.keras.applications import VGG16
# model = VGG16()
vgg = VGG16(weights='imagenet', input_shape=(32,32,3), include_top=False)
vgg.trainable=False
vgg.summary()
print(len(vgg.trainable_weights)) #동결하기 전 가중치 32 = 바이어스 16 + 가중치 16, 동결 후 0 학습 하지 않는다 

model = Sequential()
model.add(vgg)
model.add(Flatten())
model.add(Dense(10, activation='softmax'))
model.summary()
```

- 존재하는 모델을 불러와 사용하는 방법
  - include_top input layer를 사용하지 않음, input_shape를 통해 재구성
  - trainable=False
    - 불러온 모델을 가중치를 가지고 있기 때문에 특징만 뽑아줌
    - 특징을 추출하여 분류하는 레이어만 만들어 주면됨

<br>

------

<br>

## Tokenizer

keras79

단어를 어절별로 나누어 주며 높은 빈도수를 앞으로 배치

```
from tensorflow.keras.preprocessing.text import Tokenizer

text = '나는 진짜 맛있는 밥을 진짜 먹었다.'
token = Tokenizer()
token.fit_on_texts([text])

print(token.word_index)
# text = '나는 울트라 맛있는 밥을 먹었다.'
# {'나는': 1, '울트라': 2, '맛있는': 3, '밥을': 4, '먹었다': 5}
# text = '나는 진짜 맛있는 밥을 진짜 먹었다.'
# {'진짜': 1, '나는': 2, '맛있는': 3, '밥을': 4, '먹었다': 5}

x = token.texts_to_sequences([text])
print(x) #[[2, 1, 3, 4, 1, 5]]

from tensorflow.keras.utils import to_categorical

word_size = len(token.word_index)
x = to_categorical(x, num_classes=word_size+1)
print(x)
'''
[[2, 1, 3, 4, 1, 5]]
[[[0. 0. 1. 0. 0. 0.]
  [0. 1. 0. 0. 0. 0.]
  [0. 0. 0. 1. 0. 0.]
  [0. 0. 0. 0. 1. 0.]
  [0. 1. 0. 0. 0. 0.]
  [0. 0. 0. 0. 0. 1.]]] 
  낭비가 심함 .
  차원을 낮추어 만들면 낭비를 줄임 = 임베딩
'''

```

token.texts_to_sequences 결과

```
#[[2, 3], [1, 4], [1, 5, 6, 7], [8, 9, 10], [11, 12, 13, 14, 15], [16], [17], [18, 19], [20, 21], [22], [2, 23, 24], [1, 25]] 가장 긴것에 맞춰 0을 추가 어디에? 앞에 -> 자연어는 시계열이라 뒤가 영향이 큼

```



<br>

-----

<br>

## pad_sequensce

keras79

```
from tensorflow.keras.preprocessing.sequence import pad_sequences

x_test = pad_sequences(x_test,maxlen=100, padding='pre')
x_train = pad_sequences(x_train, maxlen=100, padding='pre')

#pad 결과
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
```



<br>

-----

<br>

## Embeding

keras79

```
# model.add(Embedding(25, 10, input_length=5)) #단어 사전의 개수, 아웃풋 노드의 수,  12,5의 5 = x의 cols_size
model.add(Embedding(25, 10)) #단어 사전의 개수, 아웃풋 노드의 수,  12,5의 5 = 가장 큰 문장의 cols
```

