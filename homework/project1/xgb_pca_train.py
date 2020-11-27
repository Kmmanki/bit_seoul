import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.decomposition import PCA
import datetime 
x=np.load('./homework/project1/npy/proejct1_x.npy')
y=np.load('./homework/project1/npy/proejct1_y.npy')

x= x[:1000]
y = y[:1000]
x = x.reshape(x.shape[0], x.shape[1]*x.shape[2]*x.shape[3])
print(x.shape)
start_time = datetime.datetime.now()
pca = PCA(n_components=0.99)
x = pca.fit_transform(x)
end_time = datetime.datetime.now()
print( end_time -start_time )

print(x.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state =66)

parameters = [
    {'n_estimators':[100,200,300], 'learning_rate':[0.1, 0.3, 0.001, 0.01], 'max_depth':[4,5,6] },
    {'n_estimators':[90,100,110], 'learning_rate':[0.1, 0.001 ,0.01], 'max_depth':[4,5,6], 'colsample_bytree':[0.6, 0.9, 1]},
    {'n_estimators':[90,110], 'learning_rate':[0.1,0.001, 0.5], 'max_depth':[4,5,6], 'colsample_bytree':[0.6, 0.9, 1 ], 'colsample_bylevel':[0.6,0.7,0.9]},
]


model = RandomizedSearchCV(XGBClassifier(), parameters ,cv=5)
model.fit(x_train, y_train)

acc= model.score(x_test, y_test)
print(acc)

