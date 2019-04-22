import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import data_loading as dl
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, explained_variance_score, r2_score

air = dl.load_data1()

features = pd.DataFrame(air.data, columns=air.feature_names)
targets = air.target

X_train, X_test, y_train, y_test = train_test_split(features, targets, train_size=0.8, random_state=0)

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False





"""
#LR

linear_regressor = linear_model.LinearRegression()
plt.figure()
plt.subplot(2,2,1)
plt.scatter(targets, features['Casella'], c='grey', marker='o', alpha=0.5)
x = np.array(targets).reshape(-1, 1)
y = np.array(features['Casella']).reshape(-1, 1)
linear_regressor.fit(x, y)
y_lin = linear_regressor.predict(x)
R2 = linear_regressor.score(x, y)
a = linear_regressor.coef_
b = linear_regressor.intercept_
a = a[0][0]
b = b[0]
a = round(a, 2)
b = round(b, 2)
c = "Y = "+'%s'%float(a)+"X+"+'%s'%float(b)
d = "R^2 ="+'%s'%float(round(R2, 4))
#plt.text(0.25, 0.075, c, size=15, alpha=0.8)
#plt.text(0.25, 0.050, d, size=15, alpha=0.8)
plt.plot(x, y_lin, color='r', linewidth=3, ls='-', alpha=1)
plt.title('线性回归模型')
plt.xlabel('Gravimetric_TSP (mg/m3)')
plt.ylabel('Calibrated_TSP (mg/m3)')
"""


#SVM

linear_regressor = linear_model.LinearRegression()

ss_x = StandardScaler()
X_train = ss_x.fit_transform(X_train)
X_test = ss_x.transform(X_test)

ss_y = StandardScaler()
y_train = ss_y.fit_transform(y_train.reshape(-1, 1))
y_test = ss_y.transform(y_test.reshape(-1, 1))

sv = svm.SVR(kernel='rbf', C=1)
sv.fit(X_train, y_train)

y_train_pred = sv.predict(X_train)
y_test_pred = sv.predict(X_test)

X_test = ss_x.inverse_transform(X_test)
X_train = ss_x.inverse_transform(X_train)
X_test = pd.DataFrame(X_test, columns=['H', 'SO2', 'NO2', 'PM10', 'O3', 'CO', 'PM2.5', 'KSL', 'Tem', 'Hum'])
X_train = pd.DataFrame(X_train, columns=['H', 'SO2', 'NO2', 'PM10', 'O3', 'CO', 'PM2.5', 'KSL', 'Tem', 'Hum'])
y_test = ss_y.inverse_transform(y_test).reshape(-1, 1)
y_train = ss_y.inverse_transform(y_train).reshape(-1, 1)
y_test_pred = ss_y.inverse_transform(y_test_pred).reshape(-1, 1)
y_train_pred = ss_y.inverse_transform(y_train_pred).reshape(-1, 1)

plt.subplot(2,2,1)
plt.scatter(y_test, y_test_pred, c='grey', marker='o', alpha=0.5)
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
#plt.text(0.125, 0.075, c, size=15, alpha=0.8)
#plt.text(0.125, 0.050, d, size=15, alpha=0.8)
plt.plot(y_test, y_test_lin, color='r', linewidth=3, ls='-', alpha=1)
plt.title('支持向量机模型')
plt.xlabel('Gravimetric_TSP (mg/m3)')
plt.ylabel('Calibrated_TSP (mg/m3)')

list1 = np.hstack((y_test, y_test_pred))
rf_data = pd.DataFrame(columns=['y_test', 'y_test_pred'], data=list1)
rf_data.to_csv('E:/shenxin_code/1.2 ML(10)/save/svm_data.csv')

#RF

rf = RandomForestRegressor(n_estimators=500, oob_score=True, random_state=0)
rf.fit(X_train, y_train)

y_train_pred = rf.predict(X_train)
y_test_pred = rf.predict(X_test)

y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)
y_train_pred = y_train_pred.reshape(-1, 1)
y_test_pred = y_test_pred.reshape(-1, 1)


plt.subplot(2,2,2)
plt.scatter(y_test, y_test_pred, c='grey', marker='o', alpha=0.5)
linear_regressor = linear_model.LinearRegression()
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
#plt.text(0.125, 0.075, c, size=15, alpha=0.8)
#plt.text(0.125, 0.050, d, size=15, alpha=0.8)
plt.plot(y_test, y_test_lin, color='r', linewidth=3, ls='-', alpha=1)
plt.title('随机森林模型')
plt.xlabel('Gravimetric_TSP (mg/m3)')
plt.ylabel('Calibrated_TSP (mg/m3)')

list2 = np.hstack((y_test, y_test_pred))
rf_data = pd.DataFrame(columns=['y_test', 'y_test_pred'], data=list2)
rf_data.to_csv('E:/shenxin_code/1.2 ML(10)/save/rf_data.csv')


#GBR
gbr = GradientBoostingRegressor(n_estimators=500, learning_rate=0.01, loss='ls')
gbr.fit(X_train, y_train)

y_train_pred = gbr.predict(X_train)
y_test_pred = gbr.predict(X_test)

y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)
y_train_pred = y_train_pred.reshape(-1, 1)
y_test_pred = y_test_pred.reshape(-1, 1)


plt.subplot(2,2,3)
plt.scatter(y_test, y_test_pred, c='grey', marker='o', alpha=0.5)
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
#plt.text(0.125, 0.075, c, size=15, alpha=0.8)
#plt.text(0.125, 0.050, d, size=15, alpha=0.8)
plt.plot(y_test, y_test_lin, color='r', linewidth=3, ls='-', alpha=1)
plt.title('梯度提升树模型')
plt.xlabel('Gravimetric_TSP (mg/m3)')
plt.ylabel('Calibrated_TSP (mg/m3)')

list3 = np.hstack((y_test, y_test_pred))
rf_data = pd.DataFrame(columns=['y_test', 'y_test_pred'], data=list3)
rf_data.to_csv('E:/shenxin_code/1.2 ML(10)/save/gbrt_data.csv')


#ANN

ss_x = StandardScaler()
X_train = ss_x.fit_transform(X_train)
X_test = ss_x.transform(X_test)

ss_y = StandardScaler()
y_train = ss_y.fit_transform(y_train.reshape(-1, 1))
y_test = ss_y.transform(y_test.reshape(-1, 1))


MLP = MLPRegressor(max_iter=400)
MLP.fit(X_train, y_train)

y_train_pred = MLP.predict(X_train)
y_test_pred = MLP.predict(X_test)

y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)
y_train_pred = y_train_pred.reshape(-1, 1)
y_test_pred = y_test_pred.reshape(-1, 1)

X_test = ss_x.inverse_transform(X_test)
X_train = ss_x.inverse_transform(X_train)
X_test = pd.DataFrame(X_test, columns=['H', 'SO2', 'NO2', 'PM10', 'O3', 'CO', 'PM2.5', 'Casella', 'Tem', 'Hum'])
X_train = pd.DataFrame(X_train, columns=['H', 'SO2', 'NO2', 'PM10', 'O3', 'CO', 'PM2.5', 'Casella', 'Tem', 'Hum'])
y_test = ss_y.inverse_transform(y_test).reshape(-1, 1)
y_train = ss_y.inverse_transform(y_train).reshape(-1, 1)
y_test_pred = ss_y.inverse_transform(y_test_pred).reshape(-1, 1)
y_train_pred = ss_y.inverse_transform(y_train_pred).reshape(-1, 1)

plt.subplot(2,2,4)
plt.scatter(y_test, y_test_pred, c='grey', marker='o', alpha=0.5)
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
#plt.text(0.125, 0.075, c, size=15, alpha=0.8)
#plt.text(0.125, 0.050, d, size=15, alpha=0.8)
plt.plot(y_test, y_test_lin, color='r', linewidth=3, ls='-', alpha=1)
plt.title('神经网络模型')
plt.xlabel('Gravimetric_TSP (mg/m3)')
plt.ylabel('Calibrated_TSP (mg/m3)')

list4 = np.hstack((y_test, y_test_pred))
rf_data = pd.DataFrame(columns=['y_test', 'y_test_pred'], data=list4)
rf_data.to_csv('E:/shenxin_code/1.2 ML(10)/save/ann_data.csv')



plt.show()