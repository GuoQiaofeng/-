import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import data_loading as dl
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, explained_variance_score, r2_score


plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False


air = dl.load_data()

features = pd.DataFrame(air.data, columns=air.feature_names)
targets = air.target

linear_regressor = linear_model.LinearRegression()
plt.figure()
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


targets = targets.reshape(-1, 1)
casella = features['Casella'].reshape(-1, 1)

list0 = np.hstack((targets, casella))
rf_data = pd.DataFrame(columns=['y_test', 'y_test_pred'], data=list0)
rf_data.to_csv('E:/shenxin_code/1.2 ML(10)/save/lr_data.csv')

plt.show()

