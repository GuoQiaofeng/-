
import pandas as pd
import pydotplus
from sklearn.model_selection import train_test_split
import data_loading as dl
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor


air = dl.load_data()

features = pd.DataFrame(air.data, columns=air.feature_names)
targets = air.target

X_train, X_test, y_train, y_test = train_test_split(features, targets, train_size=0.8, random_state=0)

rf = RandomForestRegressor(n_estimators=500, oob_score=True, random_state=0)
rf.fit(X_train, y_train)

# 提取单一的树
estimator = rf.estimators_[5]

from sklearn.externals.six import StringIO
dot_data = StringIO()
tree.export_graphviz(estimator,
                     out_file=dot_data,
                     feature_names=air.feature_names,
                     class_names=air.target_names,
                     filled=True,rounded=True,
                     impurity=False)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
# 输出pdf，显示整个决策树的思维过程
graph.write_pdf("tree.pdf")










