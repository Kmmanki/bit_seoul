import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# x=np.load('./homework/project1/npy/proejct1_x_pca.npy')
x=np.load('./homework/project1/npy/project1_x_pca.npy')
y=np.load('./homework/project1/npy/proejct1_y.npy')


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state =66)
#XGB를 사용하기 위해 2차원으로 reshape
# x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2]*x_train.shape[3])
# x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2]*x_test.shape[3])


model = XGBClassifier(n_jobs=-1,                        tree_method='gpu_hist', 
                            predictor='gpu_predictor')
score = model.fit(x_train, y_train)
score = model.score(x_test, y_test)
print('R2', score)

print(x_train.shape)

l = model.feature_importances_.tolist()

invaild_feature = []
for i in range(len(l)):
    if l[i] == 0:
        invaild_feature.append(i)
        

x_train  = np.delete(x_train, invaild_feature, axis=1)
x_test  = np.delete(x_test, invaild_feature, axis=1)

print(x_train.shape)

model.fit(x_train, y_train)
print("score", model.score(x_test,y_test))


index7 =np.sort(model.feature_importances_)[::-1][int(0.7 *len(model.feature_importances_) )]

delete_list = []
for i in model.feature_importances_:
    if i < index7:
        print(i,"제거 ")
        delete_list.append(model.feature_importances_.tolist().index(i))

x_train  = np.delete(x_train, delete_list, axis=1)
x_test  = np.delete(x_test, delete_list, axis=1)
# print(model.feature_importances_)

print(x_train.shape)
model.fit(x_train, y_train)
print("score", model.score(x_test,y_test))