import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator

x=np.load('./homework/project1/npy/proejct1_x.npy')
y=np.load('./homework/project1/npy/proejct1_y.npy')


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state =66)
#XGB를 사용하기 위해 2차원으로 reshape
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2]*x_train.shape[3])
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2]*x_test.shape[3])


model = XGBClassifier(n_jobs=-1)
model.fit(x_train[:3],y_train[:3])


print(x_train.shape)

#nan인 컬럼 거르기 위함
invail_index = []
fi_list = model.feature_importances_.tolist()
for  i in range(len(fi_list)):
    if str(fi_list[i]) == 'nan':
        invail_index.append(i)

x_train  = np.delete(x_train, invail_index, axis=1)
x_test  = np.delete(x_test, invail_index, axis=1)
#어라 컬럼이 0이다?
print(x_train.shape)


threshhold = np.sort(model.feature_importances_)
print(threshhold)
invaild_feature =[]
for thresh in threshhold:
    selection = SelectFromModel(model, thresh, prefit=True)

    select_x_train = selection.transform(x_train[:5])
    select_x_test = selection.transform(x_test[:5])

    print("에옹")

    select_model = XGBClassifier(n_jobs=-1)
    select_model.fit(select_x_train, y_train[:5])
    print(y_test[:5].shape)

    predict = select_model.predict(select_x_test)
    acc = accuracy_score(y_test[:5] , predict)

    print(type(thresh))

    print("Thresh=%.3f, n=%d, acc: %.5f%%" %(thresh, select_x_train.shape[1], acc))