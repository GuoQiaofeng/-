import pandas as pd
import data_loading as dl
import matplotlib.pyplot as plt

air = dl.load_data()
features = air.data
targets = air.target

plt.figure()

data = features
labels = air.feature_names

plt.boxplot(data,  # 指定绘图数据

            labels=labels,

            patch_artist=True,  # 要求用自定义颜色填充盒形图，默认白色填充

            showmeans=True,  # 以点的形式显示均值

            boxprops={'color': 'black', 'facecolor': '#9999ff'},  # 设置箱体属性，填充色和边框色

            flierprops={'marker': 'o', 'markerfacecolor': 'red', 'color': 'black'},  # 设置异常值属性，点的形状、填充色和边框色

            meanprops={'marker': 'D', 'markerfacecolor': 'indianred'},  # 设置均值点的属性，点的形状、填充色

            medianprops={'linestyle': '--', 'color': 'orange'})  # 设置中位数线的属性，线的类型和颜色



#print(features.Casella.shape())

features = pd.DataFrame(air.data, columns=air.feature_names)

for i in range(0, 424):
    if features.Casella[i]>0.2:
        features.Casella[i] = 0.2
        #features.Casella[i] = ((features.Casella[i+1])+(features.Casella[i-1])/2)
    if features.so2[i]>15:
        features.so2[i] = 15
    if features.pm10[i]>82:
        features.pm10[i] = 82
"""
plt.figure()
plt.boxplot(x=features.Casella,  # 指定绘图数据

            patch_artist=True,  # 要求用自定义颜色填充盒形图，默认白色填充

            showmeans=True,  # 以点的形式显示均值

            boxprops={'color': 'black', 'facecolor': '#9999ff'},  # 设置箱体属性，填充色和边框色

            flierprops={'marker': 'o', 'markerfacecolor': 'red', 'color': 'black'},  # 设置异常值属性，点的形状、填充色和边框色

            meanprops={'marker': 'D', 'markerfacecolor': 'indianred'},  # 设置均值点的属性，点的形状、填充色

            medianprops={'linestyle': '--', 'color': 'orange'})  # 设置中位数线的属性，线的类型和颜色

"""
plt.show()