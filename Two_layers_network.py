
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from Two_layers_Modele import *

X, y = make_circles(n_samples=100, noise = 0.1, factor= 0.3, random_state=0)
X = X.T
y = y.reshape((1 ,y.shape[0]))



parametres = nural_network(X, y, n1=4,  learning_rate=0.1, n_iter=10000)

W1 = parametres['W1']
W2 = parametres['W2']
B1 = parametres['B1']
B2 = parametres['B2']

lim = 1.5
h = 100
x1 = np.linspace(-lim , lim , h)
x2 = np.linspace(-lim , lim , h)
x11 , x22 = np.meshgrid(x1 , x2)
x_final = np.c_[x11.ravel() , x22.ravel()].T



Z1=W1.dot(x_final)+B1
A1=1/(1+np.exp(-Z1))
Z2=W2.dot(A1)+B2
Z2 = Z2.reshape(x11.shape)



#visualiser la frontiere predite par le model
#plt.contourf(x11,x22 , Z2 , [0] , colors="red", linewidths=2)
fig2, ax2 = plt.subplots(layout='constrained')
CS3 = ax2.contourf(x11,x22 , Z2, [-100, 0, 100], colors=('olivedrab', 'khaki', 'r'))
CS4 = ax2.contour(x11,x22 , Z2, [-100, 0, 100], colors="red", linewidths=3)
ax2.set_title('La frontiere predite par le model')
plt.scatter(X[0,:],X[1,:], c=y, cmap="summer" )
plt.show()