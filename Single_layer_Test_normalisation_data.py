import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from Single_layer_Modele import *



X, y = make_blobs(n_samples=100, n_features=2, centers=2, random_state=0)
X[:,1] = X[:,1] *10
Y = y.reshape((y.shape[0],1))

#visualiser Dataset
plt.scatter(X[:,0],X[:,1], c=y, cmap='summer')
plt.show()

lim = 10
h = 100
w1 = np.linspace(-lim , lim , h)
w2 = np.linspace(-lim , lim , h)
w11 , w22 = np.meshgrid(w1 , w2)
w_final = np.c_[w11.ravel() , w22.ravel()].T

b=0
z=X.dot(w_final)+b
A=1/(1+np.exp(-z))

epsilon = 1e-15
L= 1/len(Y)*np.sum(-Y*np.log(A + epsilon) - (1-Y)*np.log(1-A + epsilon), axis=0).reshape(w11.shape)

history , b = A_neuron_normalisation(X, Y,  learning_rate=0.1, n_iter=200)

print(w11.shape)
print(w22.shape)
print(L.shape)

plt.figure(figsize=(12 ,4))

plt.subplot(1,2,1)
plt.contourf(w11 , w22 , L , 10 , cmap = 'magma')
plt.colorbar()

plt.subplot(1,2,2)
plt.contourf(w11 , w22 , L , 10 , cmap = 'magma')
plt.scatter(history[:,0], history[:,1], c=history[:,2]  ,cmap='Blues' ,marker='x' )
plt.show()

