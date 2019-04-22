import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import data_loading as dl
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn import svm
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, explained_variance_score, r2_score

air = dl.load_data()

features = pd.DataFrame(air.data, columns=air.feature_names)
targets = air.target

X_train, X_test, y_train, y_test = train_test_split(features, targets, train_size=0.8, random_state=0)


ss_x = StandardScaler()
X_train = ss_x.fit_transform(X_train)
X_test = ss_x.transform(X_test)

ss_y = StandardScaler()
y_train = ss_y.fit_transform(y_train.reshape(-1, 1))
y_test = ss_y.transform(y_test.reshape(-1, 1))

"""
scaler = MinMaxScaler(feature_range=(0, 1))
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
y_train = scaler.fit_transform(y_train.reshape(-1, 1))
y_test = scaler.transform(y_test.reshape(-1, 1))
"""

sv = svm.SVR(kernel='rbf', C=1)
sv.fit(X_train, y_train)

y_train_pred = sv.predict(X_train)
y_test_pred = sv.predict(X_test)



"""
with open('save/svm.pickle', 'rb') as f:
    sv = pickle.load(f)
y_train_pred = sv.predict(X_train)
y_test_pred = sv.predict(X_test)
"""

print(sv.score(X_train, y_train))
print(sv.score(X_test, y_test))

X_test = ss_x.inverse_transform(X_test)
X_train = ss_x.inverse_transform(X_train)
#X_test = pd.DataFrame(X_test, columns=['H', 'SO2', 'NO2', 'PM10', 'O3', 'CO', 'PM2.5', 'KSL', 'Tem', 'Hum'])
#X_train = pd.DataFrame(X_train, columns=['H', 'SO2', 'NO2', 'PM10', 'O3', 'CO', 'PM2.5', 'KSL', 'Tem', 'Hum'])
y_test = ss_y.inverse_transform(y_test).reshape(-1, 1)
y_train = ss_y.inverse_transform(y_train).reshape(-1, 1)
y_test_pred = ss_y.inverse_transform(y_test_pred).reshape(-1, 1)
y_train_pred = ss_y.inverse_transform(y_train_pred).reshape(-1, 1)

"""
X_train = scaler.inverse_transform(X_train)
X_test = scaler.inverse_transform(X_test)
X_test = pd.DataFrame(X_test, columns=['H', 'SO2', 'NO2', 'PM10', 'O3', 'CO', 'PM2.5', 'KSL', 'Tem', 'Hum'])
X_train = pd.DataFrame(X_train, columns=['H', 'SO2', 'NO2', 'PM10', 'O3', 'CO', 'PM2.5', 'KSL', 'Tem', 'Hum'])
y_train = scaler.inverse_transform(y_train).reshape(-1, 1)
y_test = scaler.inverse_transform(y_test).reshape(-1, 1)
y_train_pred = scaler.inverse_transform(y_train_pred).reshape(-1, 1)
y_test_pred = scaler.inverse_transform(y_test_pred).reshape(-1, 1)
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
plt.text(35, 20, c, size=15, alpha=0.8)
plt.text(35, 15, d, size=15, alpha=0.8)
plt.plot(y_train, y_train_lin, color='black', linewidth=2, ls='--', alpha=0.5)
plt.title('svm_performance')
plt.xlabel('Gravimetric_TSP (mg/m3)')
plt.ylabel('Calibrated_TSP (mg/m3)')

"""
plt.figure()
plt.scatter(X_train['KSL'], y_train_pred, c='k', marker='o', alpha=0.5)
plt.title('svm_performance')
plt.xlabel('KSL (mg/m3)')
plt.ylabel('Calibrated_TSP (mg/m3)')
#plt.ylim(0,1)

plt.figure()
plt.scatter(X_train['Tem'], y_train_pred, c='g', marker='o', alpha=0.5)
plt.title('svm_performance')
plt.xlabel('Tem (C)')
plt.ylabel('Calibrated_TSP (mg/m3)')
plt.ylim(0,1)

plt.figure()
plt.scatter(X_train['Hum'], y_train_pred, c='b', marker='o', alpha=0.5)
plt.title('svm_performance')
plt.xlabel('Hum (%)')
plt.ylabel('Calibrated_TSP (mg/m3)')
plt.ylim(0,1)

plt.figure()
x = np.array(X_train['KSL']).reshape(-1, 1)
y = np.array(y_train_pred).reshape(-1, 1)
frh = np.divide(x, y)
plt.scatter(X_train['Hum'], frh, c='c', marker='o', alpha=0.5)
plt.title('svm_performance')
plt.xlabel('Hum (%)')
plt.ylabel('KSL/Calibrated_TSP')
#plt.ylim(0,1)

"""


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
plt.text(35, 20, c, size=15, alpha=0.8)
plt.text(35, 15, d, size=15, alpha=0.8)
plt.plot(y_test, y_test_lin, color='black', linewidth=2, ls='--', alpha=0.5)
plt.title('svm_performance')
plt.xlabel('Gravimetric_TSP (mg/m3)')
plt.ylabel('Calibrated_TSP (mg/m3)')

"""
plt.figure()
plt.scatter(X_test['KSL'], y_test_pred, c='k', marker='o', alpha=0.5)
plt.title('svm_performance')
plt.xlabel('KSL (mg/m3)')
plt.ylabel('Calibrated_TSP (mg/m3)')
#plt.ylim(0,1)

plt.figure()
plt.scatter(X_test['Tem'], y_test_pred, c='g', marker='o', alpha=0.5)
plt.title('svm_performance')
plt.xlabel('Tem (C)')
plt.ylabel('Calibrated_TSP (mg/m3)')
plt.ylim(0,1)

plt.figure()
plt.scatter(X_test['Hum'], y_test_pred, c='b', marker='o', alpha=0.5)
plt.title('svm_performance')
plt.xlabel('Hum (%)')
plt.ylabel('Calibrated_TSP (mg/m3)')
plt.ylim(0,1)

plt.figure()
x = np.array(X_test['KSL']).reshape(-1, 1)
y = np.array(y_test_pred).reshape(-1, 1)
frh = np.divide(x, y)
plt.scatter(X_test['Hum'], frh, c='c', marker='o', alpha=0.5)
plt.title('svm_performance')
plt.xlabel('Hum (%)')
plt.ylabel('KSL/Calibrated_TSP')
#plt.ylim(0,1)
"""

plt.show()

