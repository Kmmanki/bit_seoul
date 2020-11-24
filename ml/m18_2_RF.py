from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

x ,y = load_breast_cancer(return_X_y=True)
print(x.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, shuffle=True, train_size=0.8)

# model = DecisionTreeClassifier(max_depth=4)
model = RandomForestClassifier(max_depth=4)

model.fit(x_train, y_train)

acc = model.score(x_test, y_test)

print(acc)
print(model.feature_importances_) 

'''
총합은 1
acc 0.9385964912280702
0.9649122807017544
[0.04920698 0.01279842 0.05063012 0.04445663 0.00425139 0.0094626
 0.07258388 0.12919914 0.00500238 0.00345677 0.01019225 0.00539526
 0.00784025 0.02994608 0.00507804 0.0047524  0.00936389 0.00372162
 0.00561693 0.00494707 0.12111562 0.02045491 0.08494974 0.10432133
 0.0090205  0.02160143 0.02719984 0.12271734 0.01564296 0.00507425]
 '''