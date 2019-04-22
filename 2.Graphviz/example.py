from sklearn.datasets import load_iris

iris = load_iris()

# 模型（也可以使用单一决策树）
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=10)

# Train
model.fit(iris.data, iris.target)
# 提取单一的树
estimator = model.estimators_[5]

from sklearn.tree import export_graphviz

# Export as dot file
export_graphviz(estimator, out_file='tree.dot',
                feature_names=iris.feature_names,
                class_names=iris.target_names,
                rounded=True, proportion=False,
                precision=2, filled=True)

# 使用系统命令转换为png（需要Graphviz）
from subprocess import call

call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])

# 显示在jupyter笔记本
from IPython.display import Image

Image(filename='tree.png')