
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from Multi_layers_Modele import *

#Dataset
X, y = make_circles(n_samples=100, noise = 0.1, factor= 0.3, random_state=0)
X = X.T
y = y.reshape((1 ,y.shape[0]))

print(X.shape)
print(y.shape)

#model training 
parametres = deep_neural_network(X, y, hidden_layers = (32, 32, 32), learning_rate = 0.1, n_iter = 3000)


#visualiser la frontiere predite par le model
lim = 1.5
h = 100
x1 = np.linspace(-lim , lim , h)
x2 = np.linspace(-lim , lim , h)
x11 , x22 = np.meshgrid(x1 , x2)
x_final = np.c_[x11.ravel() , x22.ravel()].T

activations = {'A0': x_final}
C = len(parametres) // 2
Z=[]
for c in range(1, C + 1):
    Z = parametres['W' + str(c)].dot(activations['A' + str(c - 1)]) + parametres['b' + str(c)]
    activations['A' + str(c)] = 1 / (1 + np.exp(-Z))

Z = Z.reshape(x11.shape)


fig2, ax2 = plt.subplots(layout='constrained')
CS3 = ax2.contourf(x11,x22 , Z, [-100, 0, 100], colors=('olivedrab', 'khaki'))
CS4 = ax2.contour(x11,x22 , Z, [-100, 0, 100], colors="red", linewidths=3)
ax2.set_title('La frontiere predite par le model. (En rouge Z = 0 ou A = 0.5)')
plt.scatter(X[0,:],X[1,:], c=y, cmap="summer" )
plt.show()

"""
for key, val in activations.items():
    print(key , val.shape)
"""    
