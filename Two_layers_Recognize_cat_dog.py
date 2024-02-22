import numpy as np
import matplotlib.pyplot as plt
from utilities import *
from Two_layers_Modele import *


X_train, y_train, X_test, y_test = load_data()
X_train = X_train.T
y_train = y_train.T
X_test = X_test.T
y_test = y_test.T
X_train_reshape = X_train.reshape(-1 , X_train.shape[-1])/X_train.max()
X_test_reshape = X_test.reshape(-1 , X_test.shape[-1])/X_test.max()

parametres = nural_network_test(X_train_reshape , y_train, X_test_reshape , y_test , n1 =8 , learning_rate=0.01 , n_iter=10000)


"""
print(X_train_reshape.shape)
print(X_test_reshape.shape)
print(y_test.shape)
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

