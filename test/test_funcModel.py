import numpy as np
from tensorflow.keras.layers import Input,Dense, concatenate
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

#1.데이터
x1 = np.array((range(0,10), range(20, 30), range(50, 60)))
x2 = np.array((range(5,15), range(25, 35), range(55, 65)))
y = np.array((range(7,17), range(27, 37), range(57, 67)))

x1 = x1.T
x2 = x2.T
y = y.T

x1_train, x1_test, y_train, y_test, x2_train, x2_test = train_test_split(x1,y,x2, train_size=0.8)

# print(x1)
# print(x2)
# print(y)

#2.모델
    #x1 데이터 입력
input1 = Input(shape=(3,))
dense1 = Dense(16, activation='relu')(input1)

    #x2 데이터 입력
input2 = Input(shape=(3,))
dense2 = Dense(8, activation='relu')(input2)

concat = concatenate([dense1, dense2])
dense3 = Dense(4, activation='relu')(concat)
dense3 = Dense(3, activation='linear')(dense3)

model = Model(inputs=[input1, input2], outputs=[dense3])

model.summary()
#3.훈련
model.compile(loss='mse', metrics=['acc'], optimizer='adam')
model.fit([x1_train, x2_train],y_train, batch_size=1, epochs=100, validation_split=0.2, verbose=1 )


#4. 평가
result = model.evaluate([x1_test, x2_test], y_test, batch_size=1)
print(result)

y_predict = model.predict([x1_test,x2_test])
def R2(y_test, y_precit):
    return r2_score(y_test, y_precit)



print(R2(y_test, y_predict))
