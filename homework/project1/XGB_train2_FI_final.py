

# xgb를 사용하기 위해서 할것
# 1. pca 하습

import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.decomposition import PCA
import datetime 
import pickle as pk
#pca 모델을 저장해야 항상 일정한 컬럼 개수가 나올 것 
start_time = datetime.datetime.now()

x=np.load('./homework/project1/npy/proejct1_x.npy')
y=np.load('./homework/project1/npy/proejct1_y.npy')


x = x.reshape(x.shape[0], x.shape[1]*x.shape[2]*x.shape[3])
print(x.shape)
pca = PCA(n_components=0.99)
x = pca.fit_transform(x)



print(x.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state =66)


model = XGBClassifier()
model.fit(x_train, y_train)

acc= model.score(x_test, y_test)
print(acc)


from tensorflow.keras.preprocessing.image import ImageDataGenerator

xy=ImageDataGenerator(rescale=1./255).flow_from_directory(
    './homework/project1/x_predict', #실제 이미지가 있는 폴더는 라벨이 됨. (ad/normal=0/1)
    target_size=(178,218),
    batch_size=20000,
    class_mode='binary'
)
x_predict = xy[0][0]
y_real = xy[0][1]
x_predict = x_predict.reshape(x_predict.shape[0], x_predict.shape[1]*x_predict.shape[2]*x_predict.shape[3])
x_predict = pca.transform(x_predict)



index7 =np.sort(model.feature_importances_)[::-1][int(0.7 *len(model.feature_importances_) )]

delete_list = []
for i in model.feature_importances_:
    if i <= index7:
        print(i,"제거 ")
        delete_list.append(model.feature_importances_.tolist().index(i))

x_train  = np.delete(x_train, delete_list, axis=1)
x_test  = np.delete(x_test, delete_list, axis=1)
x_predict = np.delete(x_predict, delete_list, axis=1)

parameters = [
    {'n_estimators':[100,200,300], 'learning_rate':[0.1, 0.3, 0.001, 0.01], 'max_depth':[4,5,6] },
    {'n_estimators':[90,100,110], 'learning_rate':[0.1, 0.001 ,0.01], 'max_depth':[4,5,6], 'colsample_bytree':[0.6, 0.9, 1]},
    {'n_estimators':[90,110], 'learning_rate':[0.1,0.001, 0.5], 'max_depth':[4,5,6], 'colsample_bytree':[0.6, 0.9, 1 ], 'colsample_bylevel':[0.6,0.7,0.9]},
]


model = RandomizedSearchCV(XGBClassifier(), parameters ,cv=5)
model.fit(x_train, y_train)

y_predict = model.predict(x_predict)

print(y_predict)
print("예상값", np.round(y_predict).reshape(y_predict.shape[0],))

print("acc:", accuracy_score(y_real,y_predict))

end_time = datetime.datetime.now()
print( end_time -start_time )