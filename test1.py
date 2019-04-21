import numpy as np
import pandas as pd
from keras.utils import np_utils

from keras.datasets import mnist

(X_train_image, y_train_label), (X_test_image, y_test_label) = mnist.load_data()


print('train data=', len(X_train_image))


import matplotlib.pyplot as plt
def plt_image(image):
    fig = plt.gcf()
    fig.set_size_inches(2, 2)
    plt.imshow(image, cmap='binary')
    plt.show()


plt_image(X_train_image[0])

