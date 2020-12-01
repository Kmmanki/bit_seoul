# pip install autokeras
# pip install git+https://github.com/keras-team/keras-tuner.git@1.0.2rc4
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.python.keras.utils.data_utils import Sequence
import autokeras as ak
from tensorflow.keras.models import load_model

x_train = np.load('./data/keras63_train_x.npy')
x_test = np.load('./data/keras63_test_x.npy')
y_train = np.load('./data/keras63_train_y.npy')
y_test = np.load('./data/keras63_test_t.npy')

#shape
print(x_train.shape)
print(y_train.shape)
print(y_train[:3])

clf = ak.ImageClassifier(
    overwrite=True,
    max_trials=1
)
clf.fit(x_train, y_train, epochs=50)
clf = load_model("./homework/project1/models/autokeras1.h5", custom_objects=ak.CUSTOM_OBJECTS)
model = clf.export_model()
predicted_y = clf.predict(x_test)
print(predicted_y)
try:
    model.save('./homework/project1/models/autokeras1.h5')
except:
    model.save('./homework/project1/models/autokeras1', save_format="tf")
print(clf.evaluate(x_test, y_test))

'''
[1.8198654651641846, 0.5416666865348816]

[1.7608370780944824, 0.5249999761581421]
'''