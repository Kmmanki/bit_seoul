import numpy as np
from tensorflow.keras.datasets import cifar100
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error, r2_score

dataset = load_boston()

x = dataset.data
y = dataset.target

pca = PCA()
pca.fit(x)
cumsum = np.cumsum(pca.explained_variance_ratio_)

d95 = np.argmax(cumsum > 0.95) +1 
d1 = np.argmax(cumsum >= 1.0) +1 

x_d95 = PCA(n_components=d95)
x_d1 = PCA(n_components=d1)

x_d95 = x_d95.fit_transform(x)
x_d1 = x_d1.fit_transform(x)

x_d95_train, x_d95_test, y_train, y_test = train_test_split(x, y, train_size=0.8)
x_d1_train, x_d1_test  = train_test_split(x,  train_size=0.8)



model95 = Sequential() #r2 =0.8
model95.add(Dense(30, activation='relu', input_shape=(x_d95_train.shape[1],)))
model95.add(Dense(100, activation='relu'))
model95.add(Dropout(0.2))
model95.add(Dense(400, activation='relu'))
model95.add(Dropout(0.5))
model95.add(Dense(300, activation='relu'))
model95.add(Dropout(0.5))
model95.add(Dense(100, activation='relu'))
model95.add(Dropout(0.2))
model95.add(Dense(1))

model1 = Sequential() #r2 =0.8
model1.add(Dense(30, activation='relu', input_shape=(x_d1_train.shape[1],)))
model1.add(Dense(100, activation='relu'))
model1.add(Dropout(0.2))
model1.add(Dense(400, activation='relu'))
model1.add(Dropout(0.5))
model1.add(Dense(300, activation='relu'))
model1.add(Dropout(0.5))
model1.add(Dense(100, activation='relu'))
model1.add(Dropout(0.2))
model1.add(Dense(1))

ealystopping = EarlyStopping(monitor='loss', patience=30, mode='min')

model1.compile(loss='mse', optimizer='adam', metrics=[])
model95.compile(loss='mse', optimizer='adam', metrics=[])

model1.fit(x_d1_train, y_train, epochs=500, callbacks=[ealystopping], verbose=1, validation_split=0.2, batch_size=4)
model95.fit(x_d95_train, y_train, epochs=500, callbacks=[ealystopping], verbose=1, validation_split=0.2, batch_size=4)

loss1 = model1.evaluate(x_d1_test, y_test, batch_size=512)
loss95 = model95.evaluate(x_d95_test, y_test, batch_size=512)

x_predict95 = x_d95_test[30:40]
x_predict1 = x_d1_test[30:40]

y95 = model95.predict(x_predict95)
y1 = model1.predict(x_predict1)
print("loss95",loss95)
print("r295",r2_score(y95, y_test[30:40] ))
print("loss1",loss1)
print("r21",r2_score(y1, y_test[30:40] ))

'''
noPCA
loss:  13.658388137817383
RMSE 2.6497556936268225
R2:  0.8747736192967522

loss95 29.475940704345703
r2 95 0.7025964632727291


loss1 116.25751495361328
r2 1 -35.345250735933334
'''

