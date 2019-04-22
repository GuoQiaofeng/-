# coding=utf-8
import numpy as np
import pandas as pd
import data_loading as dl
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import r2_score
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt

air = dl.load_data()

features = pd.DataFrame(air.data, columns=air.feature_names)
targets = air.target

X_train, X_test, y_train, y_test = train_test_split(features, targets, train_size=0.8, random_state=0)

scaler = MinMaxScaler(feature_range=(0, 1))
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)
y_train = scaler.fit_transform(y_train.reshape((-1, 1)))
y_test = scaler.fit_transform(y_test.reshape((-1, 1)))

model = Sequential()

model.add(Dense(units=100,
                input_dim=10,
                kernel_initializer='normal',
                activation='relu'))
"""
model.add(Dense(units=100,
                kernel_initializer='normal',
                activation='relu'))
model.add(Dense(units=100,
                kernel_initializer='normal',
                activation='relu'))
"""
model.add(Dense(units=1))

print(model.summary())

model.compile(loss='mse',
              optimizer='adam')

train_history = model.fit(x=X_train,
                          y=y_train,
                          validation_split=0.2,
                          epochs=150, batch_size=10, verbose=2)


def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train_History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')


show_train_history(train_history, 'loss', 'val_loss')

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)


X_train = scaler.inverse_transform(X_train)
X_test = scaler.inverse_transform(X_test)
y_train = scaler.inverse_transform(y_train).reshape(-1, 1)
y_test = scaler.inverse_transform(y_test).reshape(-1, 1)
y_train_pred = scaler.inverse_transform(y_train_pred).reshape(-1, 1)
y_test_pred = scaler.inverse_transform(y_test_pred).reshape(-1, 1)

print(r2_score(y_train, y_train_pred))
print(r2_score(y_test, y_test_pred))

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
plt.title('DNN_train')
plt.xlabel('Gravimetric_TSP (mg/m3)')
plt.ylabel('Calibrated_TSP (mg/m3)')

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
plt.title('DNN_test')
plt.xlabel('Gravimetric_TSP (mg/m3)')
plt.ylabel('Calibrated_TSP (mg/m3)')
plt.show()

