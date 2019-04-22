import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import data_loading as dl
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, explained_variance_score, r2_score

air = dl.load_data()

features = pd.DataFrame(air.data, columns=air.feature_names)
targets = air.target

X_train, X_test, y_train, y_test = train_test_split(features, targets, train_size=0.8, random_state=0)

rf = RandomForestRegressor(n_estimators=500, oob_score=True, random_state=1)
rf.fit(X_train, y_train)

y_train_pred = rf.predict(X_train)
y_test_pred = rf.predict(X_test)

y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)
y_train_pred = y_train_pred.reshape(-1, 1)
y_test_pred = y_test_pred.reshape(-1, 1)

"""
with open('save/rf.pickle', 'rb') as f:
    rf = pickle.load(f)
y_train_pred = rf.predict(X_train)
y_test_pred = rf.predict(X_test)
"""

print(rf.score(X_train, y_train))
print(rf.score(X_test, y_test))


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
plt.title('rf_performance')
plt.xlabel('Gravimetric_TSP (mg/m3)')
plt.ylabel('Calibrated_TSP (mg/m3)')

"""
plt.figure()
plt.scatter(X_train['KSL'], y_train_pred, c='k', marker='o', alpha=0.5)
plt.title('rf_performance')
plt.xlabel('KSL (mg/m3)')
plt.ylabel('Calibrated_TSP (mg/m3)')
#plt.ylim(0,1)

plt.figure()
plt.scatter(X_train['Tem'], y_train_pred, c='g', marker='o', alpha=0.5)
plt.title('rf_performance')
plt.xlabel('Tem (C)')
plt.ylabel('Calibrated_TSP (mg/m3)')
plt.ylim(0,1)

plt.figure()
x = np.array(X_train['KSL']).reshape(-1, 1)
y = np.array(y_train_pred).reshape(-1, 1)
frh = np.divide(x, y)
plt.scatter(X_train['Hum'], frh, c='c', marker='o', alpha=0.5)
plt.title('rf_performance')
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
plt.text(0.125, 0.075, c, size=15, alpha=0.8)
plt.text(0.125, 0.050, d, size=15, alpha=0.8)
plt.plot(y_test, y_test_lin, color='black', linewidth=2, ls='--', alpha=0.5)
plt.title('rf_performance')
plt.xlabel('Gravimetric_TSP (mg/m3)')
plt.ylabel('Calibrated_TSP (mg/m3)')


"""
plt.figure()
plt.scatter(X_test['KSL'], y_test_pred, c='k', marker='o', alpha=0.5)
plt.title('rf_performance')
plt.xlabel('KSL (mg/m3)')
plt.ylabel('Calibrated_TSP (mg/m3)')
#plt.ylim(0,1)

plt.figure()
plt.scatter(X_test['Tem'], y_test_pred, c='g', marker='o', alpha=0.5)
plt.title('rf_performance')
plt.xlabel('Tem (C)')
plt.ylabel('Calibrated_TSP (mg/m3)')
plt.ylim(0, 1)

plt.figure()
plt.scatter(X_test['Hum'], y_test_pred, c='b', marker='o', alpha=0.5)
plt.title('rf_performance')
plt.xlabel('Hum (%)')
plt.ylabel('Calibrated_TSP (mg/m3)')
plt.ylim(0,1)

plt.figure()
x = np.array(X_test['KSL']).reshape(-1, 1)
y = np.array(y_test_pred).reshape(-1, 1)
frh = np.divide(x, y)
plt.scatter(X_test['Hum'], frh, c='c', marker='o', alpha=0.5)
plt.title('rf_performance')
plt.xlabel('Hum (%)')
plt.ylabel('KSL/Calibrated_TSP')
#plt.ylim(0,1)
"""

plt.show()



def plot_feature_importances(feature_importances, title):

    feature_names = air.feature_names
    # 将得分从高到低排序
    index_sorted = np.flipud(np.argsort(feature_importances))

    # 让X坐标轴上的标签居中显示
    pos = np.arange(index_sorted.shape[0]) + 0.5

    # 画条形图
    plt.figure()
    plt.bar(pos, feature_importances, align='center')
    plt.xticks(pos, feature_names)
    plt.ylabel('Relative Importance')
    plt.title(title)
    plt.show()


plot_feature_importances(rf.feature_importances_, 'RF')

import pickle
with open('save/rf.pickle', 'wb') as f:
    pickle.dump(rf, f)