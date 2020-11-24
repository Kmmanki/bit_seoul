from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import numpy as np

cancer = load_breast_cancer()

x ,y = load_breast_cancer(return_X_y=True)
print(x.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, shuffle=True, train_size=0.8)

# model = DecisionTreeClassifier(max_depth=4)
# model = RandomForestClassifier(max_depth=4)
# model = GradientBoostingClassifier(max_depth=4)
model = XGBClassifier(max_depth=4)

model.fit(x_train, y_train)

acc = model.score(x_test, y_test)

print(acc)
print(model.feature_importances_) 
print(cancer.data.shape[1])

def plot_feature_importances_cancer(data_name, model):
    n_features = data_name.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_,
    align='center')
    plt.yticks(np.arange(n_features), data_name.feature_names)
    plt.xlabel('Feacutre Importance')
    plt.ylabel('Feature')
    plt.ylim(-1, n_features)

plot_feature_importances_cancer(cancer,model)
plt.show()
'''
DecisionTreeClassifier
acc 0.9385964912280702
총합은 1
[0.         0.         0.         0.         0.         0.
 0.         0.70458252 0.         0.         0.         0.00639525
 0.         0.01221069 0.         0.         0.         0.0162341
 0.         0.0189077  0.05329492 0.05959094 0.05247428 0.
 0.00940897 0.         0.         0.06690062 0.         0.        ]

RandomForestClassifier
총합은 1
acc 0.9385964912280702
[0.         0.         0.         0.         0.         0.
 0.         0.70458252 0.         0.         0.         0.00639525
 0.         0.01221069 0.         0.         0.         0.0162341
 0.         0.0189077  0.05329492 0.05959094 0.05247428 0.
 0.00940897 0.         0.         0.06690062 0.         0.        ]

GradientBoostingClassifier
acc: 0.956140350877193
[7.76584550e-06 4.35018642e-02 1.25786490e-04 5.74942454e-05
 2.50855217e-04 5.47308200e-04 9.03921649e-04 6.71611663e-01
 8.27929223e-04 6.11656786e-05 5.85725865e-03 1.92017782e-03
 3.69955119e-06 6.28758964e-03 1.32754336e-03 3.48643223e-04
 7.88182170e-03 1.53824214e-02 5.42112298e-04 1.09307689e-02
 5.80947359e-02 2.18344681e-02 5.50936037e-02 3.63666418e-03
 8.66421877e-03 2.12310255e-03 2.42599678e-03 7.93267827e-02
 3.90570999e-04 3.20657111e-05]

XGBClassifier
 0.9649122807017544
[6.5209707e-03 2.4647316e-02 5.1710028e-03 8.5555250e-03 3.9974600e-03 
 4.6066064e-03 2.6123745e-03 4.4031221e-01 3.4104299e-04 2.0658956e-03 
 1.2474300e-02 6.8988632e-03 1.7291201e-02 5.6377212e-03 3.1236352e-03 
 3.2256048e-03 2.7834374e-02 6.4999197e-04 7.1520107e-03 0.0000000e+00 
 7.2251976e-02 1.7675869e-02 8.9819685e-02 2.0090658e-02 1.0589287e-02 
 0.0000000e+00 1.2581742e-02 1.8795149e-01 5.9212563e-03 0.0000000e+00]
 '''