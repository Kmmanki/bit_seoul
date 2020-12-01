import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.python.keras.utils.data_utils import Sequence
import autokeras as ak
from sklearn.model_selection import train_test_split

x=np.load('./homework/project1/npy/proejct1_x.npy')
y=np.load('./homework/project1/npy/proejct1_y.npy')

# print(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state =66)

#shape
print(x_train.shape)
print(y_train.shape)

clf = ak.ImageClassifier(
    overwrite=True,
    max_trials=1
)

clf.fit(x_train, y_train, epochs=50)
model = clf.export_model()
try:
    model.save('./homework/project1/models/autokeras1.h5')
except:
    model.save('./homework/project1/models/autokeras1', save_format="tf")
predicted_y = clf.predict(x_test)
print(predicted_y)

print(clf.evaluate(x_test, y_test))
