import numpy as np
import matplotlib.pyplot as plt
from utilities import *
from Single_layer_Modele import *


X_train, y_train, X_test, y_test = load_data()
X_train_reshape = X_train.reshape(X_train.shape[0],-1)/X_train.max()
X_test_reshape = X_test.reshape(X_test.shape[0],-1)/X_test.max()

w, b = a_neuron_train_and_test(X_train_reshape , y_train , X_test_reshape , y_test , learning_rate=0.01 , n_iter=10000)


"""
print(X_train_reshape.shape)
print(X_test_reshape.shape)
print(X_train.shape)
print(y_train.shape)
print(np.unique(y_train, return_counts=True))
print(X_test.shape)
print(y_test.shape)
print(np.unique(y_test, return_counts=True))
"""


"""
plt.figure(figsize=(16, 8))
for i in range(1, 10):
    plt.subplot(4, 3, i)
    plt.imshow(X_train[i], cmap='gray')
    plt.title(y_train[i])
    plt.tight_layout()
plt.show()

"""

