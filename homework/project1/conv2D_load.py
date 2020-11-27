import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model

x=np.load('./homework/project1/npy/proejct1_x.npy')
y=np.load('./homework/project1/npy/proejct1_y.npy')

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)

model = load_model('./homework/project1/models/model_Conv2D_train1_acc0.9897111058235168.h5')

model.summary()

loss, acc = model.evaluate(x_test, y_test)

print("loss", loss)
print("acc", acc)

