import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import data_loading as dl
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve

air = dl.load_data()

features = pd.DataFrame(air.data, columns=air.feature_names)
targets = air.target


param_range = np.arange(10, 1000, 100)
print(param_range)
train_loss, test_loss = validation_curve(
    RandomForestRegressor(), features, targets, param_name='n_estimators', param_range=param_range, cv=10,
    scoring="neg_mean_squared_error")
train_loss_mean = -np.mean(train_loss, axis=1)
test_loss_mean = -np.mean(test_loss, axis=1)

plt.plot(param_range, train_loss_mean, 'o-', color='r', label="Training")
plt.plot(param_range, test_loss_mean, 'o-', color='g', label="Cross-validation")

plt.xlabel("Trees")
plt.ylabel("Loss")
plt.legend(loc="best")
plt.show()
