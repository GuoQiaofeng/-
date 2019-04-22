import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import data_loading as dl
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, explained_variance_score, r2_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score

air = dl.load_data()

features = pd.DataFrame(air.data, columns=air.feature_names)
targets = air.target

X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.1, random_state=0)

rf = RandomForestRegressor(n_estimators=500, oob_score=True, random_state=0)

scores1 = -cross_val_score(rf, X_train, y_train, cv=9, scoring='neg_mean_squared_error')
print(scores1)
print(scores1.mean())

scores2 = cross_val_score(rf, X_train, y_train, cv=9, scoring='r2')
print(scores2)
print(scores2.mean())

""""""
rf.fit(X_train, y_train)

y_train_pred = rf.predict(X_train)
y_test_pred = rf.predict(X_test)
#y_pred = rf.predict(features)

y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)
y_train_pred = y_train_pred.reshape(-1, 1)
y_test_pred = y_test_pred.reshape(-1, 1)

"""
y_pred = y_pred.reshape(-1, 1)
targets = targets.reshape(-1, 1)

y_p = pd.DataFrame(y_pred)
y_p.to_csv('target.csv')
"""

print(rf.score(X_train, y_train))
print(rf.score(X_test, y_test))


"""
#训练集
plt.figure()
plt.scatter(y_train, y_train_pred, c='r', marker='o', alpha=0.5)
linear_regressor = linear_model.LinearRegression()
linear_regressor.fit(y_train, y_train_pred)
y_train_lin = linear_regressor.predict(y_train)
R2 = r2_score(y_train, y_train_pred)
a = linear_regressor.coef_
b = linear_regressor.intercept_
a = a[0][0]
b = b[0]
a = round(a, 2)
b = round(b, 2)
c = "Y = "+'%s'%float(a)+"X+"+'%s'%float(b)
d = "R2 ="+'%s'%float(round(R2, 4))
plt.text(0.125, 0.075, c, size=15, alpha=0.8)
plt.text(0.125, 0.050, d, size=15, alpha=0.8)
plt.plot(y_train, y_train_lin, color='black', linewidth=2, ls='--', alpha=0.5)
plt.title('rf_train')
plt.xlabel('Gravimetric_TSP (mg/m3)')
plt.ylabel('Calibrated_TSP (mg/m3)')



#测试集
plt.figure()
plt.scatter(y_test, y_test_pred, c='r', marker='o', alpha=0.5)
linear_regressor.fit(y_test, y_test_pred)
y_test_lin = linear_regressor.predict(y_test)
R2 = r2_score(y_test, y_test_pred)
a = linear_regressor.coef_
b = linear_regressor.intercept_
a = a[0][0]
b = b[0]
a = round(a, 2)
b = round(b, 2)
c = "Y = "+'%s'%float(a)+"X+"+'%s'%float(b)
d = "R2 ="+'%s'%float(round(R2, 4))
plt.text(0.125, 0.075, c, size=15, alpha=0.8)
plt.text(0.125, 0.050, d, size=15, alpha=0.8)
plt.plot(y_test, y_test_lin, color='black', linewidth=2, ls='--', alpha=0.5)
plt.title('rf_test')
plt.xlabel('Gravimetric_TSP (mg/m3)')
plt.ylabel('Calibrated_TSP (mg/m3)')


#测试集
plt.figure()
plt.scatter(targets, y_pred, c='r', marker='o', alpha=0.5)
linear_regressor.fit(targets, y_pred)
y_test_lin = linear_regressor.predict(targets)
R2 = r2_score(targets, y_pred)
a = linear_regressor.coef_
b = linear_regressor.intercept_
a = a[0][0]
b = b[0]
a = round(a, 2)
b = round(b, 2)
c = "Y = "+'%s'%float(a)+"X+"+'%s'%float(b)
d = "R2 ="+'%s'%float(round(R2, 4))
plt.text(0.125, 0.075, c, size=15, alpha=0.8)
plt.text(0.125, 0.050, d, size=15, alpha=0.8)
plt.plot(targets, y_test_lin, color='black', linewidth=2, ls='--', alpha=0.5)
plt.title('rf_all')
plt.xlabel('Gravimetric_TSP (mg/m3)')
plt.ylabel('Calibrated_TSP (mg/m3)')
plt.show()

"""