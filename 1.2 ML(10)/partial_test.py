from __future__ import print_function
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble.partial_dependence import plot_partial_dependence
from sklearn.ensemble.partial_dependence import partial_dependence
import data_loading as dl
from mpl_toolkits.mplot3d import Axes3D

#
def main():
    air = dl.load_data2()

    # split 80/20 train-test
    X_train, X_test, y_train, y_test = train_test_split(air.data,
                                                        air.target,
                                                        test_size=0.2,
                                                        random_state=1)
    """
    ss_x = StandardScaler()
    X_train = ss_x.fit_transform(X_train)
    X_test = ss_x.transform(X_test)

    ss_y = StandardScaler()
    y_train = ss_y.fit_transform(y_train.reshape(-1, 1))
    y_test = ss_y.transform(y_test.reshape(-1, 1))
    """
    names = air.feature_names

    print("Training GBRT...")
    clf = GradientBoostingRegressor(n_estimators=500)
    clf.fit(X_train, y_train)
    print(" done.")

    print('Convenience plot with ``partial_dependence_plots``')

    features = [(8, 9)]
    fig, axs = plot_partial_dependence(clf, X_train, features,
                                       feature_names=names,
                                       n_jobs=3, grid_resolution=100)
#    fig.suptitle('Partial dependence of TSP on Different Features\n'
#                 'for the dataset')
    plt.subplots_adjust(top=0.9)  # tight_layout causes overlap with suptitle

    print('Custom 3d plot via ``partial_dependence``')


    fig = plt.figure()

    target_feature = (7, 9)
    pdp, axes = partial_dependence(clf, target_feature,
                                   X=X_train, grid_resolution=50)
    XX, YY = np.meshgrid(axes[0], axes[1])
    Z = pdp[0].reshape(list(map(np.size, axes))).T
    ax = Axes3D(fig)
    surf = ax.plot_surface(XX, YY, Z, rstride=1, cstride=1,
                           cmap=plt.cm.BuPu, edgecolor='k')
    ax.set_xlabel(names[target_feature[0]])
    ax.set_ylabel(names[target_feature[1]])
    ax.set_zlabel('Partial dependence')
    #  pretty init view
    ax.view_init(elev=22, azim=122)
    plt.colorbar(surf)
    plt.suptitle('Partial dependence of TSP on \n'
                 'Grav and Hum')
    plt.subplots_adjust(top=0.9)

    plt.show()


# Needed on Windows because plot_partial_dependence uses multiprocessing
if __name__ == '__main__':
    main()
"""
import pickle
with open('save/gbr.pickle', 'wb') as f:
    pickle.dump(gbr, f)
"""


