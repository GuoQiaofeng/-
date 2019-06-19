import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import data_loading as dl
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, explained_variance_score, r2_score
from sklearn.model_selection import cross_val_score

air = dl.load_data1()

features = pd.DataFrame(air.data, columns=air.feature_names)
targets = air.target

X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=0)

ss_x = StandardScaler()
X_train = ss_x.fit_transform(X_train)
X_test = ss_x.transform(X_test)

ss_y = StandardScaler()
y_train = ss_y.fit_transform(y_train.reshape(-1, 1))
y_test = ss_y.transform(y_test.reshape(-1, 1))


MLP = MLPRegressor(max_iter=400)
MLP.fit(X_train, y_train)

scores = cross_val_score(MLP, X_train, y_train, scoring="neg_mean_squared_error", cv=10)
r2 = cross_val_score(MLP, X_train, y_train, scoring="r2", cv=10)
rmse_scores = np.sqrt(-scores).mean()
print("MSE:", -scores.mean())
print("RMSE:", rmse_scores)
print("R2:", r2.mean())

y_train_pred = MLP.predict(X_train)
y_test_pred = MLP.predict(X_test)

y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)
y_train_pred = y_train_pred.reshape(-1, 1)
y_test_pred = y_test_pred.reshape(-1, 1)


"""
scores2 = cross_val_score(MLP, X_train, y_train, cv=9, scoring='r2')
print(scores2)
print(scores2.mean())

print(MLP.score(X_train, y_train))
print(MLP.score(X_test, y_test))
"""

X_test = ss_x.inverse_transform(X_test)
X_train = ss_x.inverse_transform(X_train)
X_test = pd.DataFrame(X_test, columns=['H', 'SO2', 'NO2', 'PM10', 'O3', 'CO', 'PM2.5', 'KSL', 'Tem', 'Hum'])
X_train = pd.DataFrame(X_train, columns=['H', 'SO2', 'NO2', 'PM10', 'O3', 'CO', 'PM2.5', 'KSL', 'Tem', 'Hum'])
y_test = ss_y.inverse_transform(y_test).reshape(-1, 1)
y_train = ss_y.inverse_transform(y_train).reshape(-1, 1)
y_test_pred = ss_y.inverse_transform(y_test_pred).reshape(-1, 1)
y_train_pred = ss_y.inverse_transform(y_train_pred).reshape(-1, 1)

"""
scores1 = -cross_val_score(MLP, X_train, y_train, cv=9, scoring='neg_mean_squared_error')
print(scores1)
print(scores1.mean())
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
plt.title('ANN_train')
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
plt.title('ANN_performance')
plt.xlabel('Gravimetric_TSP (mg/m3)')
plt.ylabel('Calibrated_TSP (mg/m3)')


plt.show()
