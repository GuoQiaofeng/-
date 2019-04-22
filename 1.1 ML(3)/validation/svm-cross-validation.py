import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import data_loading as dl
from sklearn.preprocessing import StandardScaler, scale
from sklearn import svm
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve


air = dl.load_data()

features = pd.DataFrame(air.data, columns=air.feature_names)
targets = air.target

ss_x = StandardScaler()
X = ss_x.fit_transform(features)

ss_y = StandardScaler()
y = ss_y.fit_transform(targets.reshape(-1, 1))

param_range = np.arange(0.1, 100, 1)
print(param_range)
train_loss, test_loss = validation_curve(
    svm.SVR(), X, y, param_name='C', param_range=param_range, cv=10,
    scoring="neg_mean_squared_error")
train_loss_mean = -np.mean(train_loss, axis=1)
test_loss_mean = -np.mean(test_loss, axis=1)

plt.plot(param_range, train_loss_mean, 'o-', color='r', label="Training")
plt.plot(param_range, test_loss_mean, 'o-', color='g', label="Cross-validation")

plt.xlabel("C")
plt.ylabel("Loss")
plt.legend(loc="best")
plt.show()


"""
sv = svm.SVR(kernel='rbf', C=1)

train_sizes, train_loss, test_loss = learning_curve(
    sv, features, targets, cv=10, scoring="neg_mean_squared_error", train_sizes=
    [0.1, 0.25, 0.5, 0.75, 1])
train_loss_mean = -np.mean(train_loss, axis=1)
test_loss_mean = -np.mean(test_loss, axis=1)

plt.plot(train_sizes, train_loss_mean, 'o-', color='r', label="Training")
plt.plot(train_sizes, test_loss_mean, 'o-', color='g', label="Cross-validation")

plt.xlabel("Training example")
plt.ylabel("Loss")
plt.legend(loc="best")
plt.show()
"""