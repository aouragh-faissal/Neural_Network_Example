import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import plotly.graph_objects as go
from utilities import *
from Single_layer_Modele import *


X, y = make_blobs(n_samples=100, n_features=2, centers=2, random_state=0)
Y = y.reshape((y.shape[0],1))

#visualise Dataset
#plt.scatter(X[:,0],X[:,1], c=y, cmap='summer')
#plt.show()
 
    
w, b = artificial_neuron(X , Y, learning_rate=0.1, n_iter=1000)


new_plaint = np.array([2,1])
print("la classe predite de point rouge est " + str(predict(new_plaint , w ,b)))

x0 = np.linspace(-1 , 4 , 100)
x1 = ( -w[0]*x0 - b)/w[1]

plt.plot(x0 , x1 , c='orange' , lw=2)
plt.scatter(X[:,0],X[:,1], c=y, cmap='summer')
plt.scatter(new_plaint[0],new_plaint[1], c='r')
plt.show()


#################################################################################
# visualisation 3D
#################################################################################

fig = go.Figure(data=[go.Scatter3d( 
    x=X[:, 0].flatten(),
    y=X[:, 1].flatten(),
    z=y.flatten(),
    mode='markers',
    marker=dict(
        size=5,
        color=y.flatten(),                
        colorscale='YlGn',  
        opacity=0.8,
        reversescale=True
    )
)])




fig.update_layout(template= "plotly_dark", margin=dict(l=0, r=0, b=0, t=0))
fig.layout.scene.camera.projection.type = "orthographic"
fig.show()

X0 = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
X1 = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
xx0, xx1 = np.meshgrid(X0, X1)
Z = w[0] * xx0 + w[1] * xx1 + b
A = 1 / (1 + np.exp(-Z))

fig = (go.Figure(data=[go.Surface(z=A, x=xx0, y=xx1, colorscale='YlGn', opacity = 0.7, reversescale=True)]))

fig.add_scatter3d(x=X[:, 0].flatten(), y=X[:, 1].flatten(), z=y.flatten(), mode='markers', marker=dict(size=5, color=y.flatten(), colorscale='YlGn', opacity = 0.9, reversescale=True))


fig.update_layout(template= "plotly_dark", margin=dict(l=0, r=0, b=0, t=0))
fig.layout.scene.camera.projection.type = "orthographic"
fig.show()


    







    





