import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from tqdm import tqdm


def Initialisation(n0 , n1 , n2):
    W1=np.random.randn(n1 , n0)
    B1=np.random.randn(n1 , 1)
    W2=np.random.randn(n2 , n1)
    B2=np.random.randn(n2 , 1)
    
    parametres = {
        'W1' : W1,
        'B1' : B1,
        'W2' : W2,
        'B2' : B2,  
        }
      
    return parametres


def forward_propagation(X, parametres):
    
    W1 = parametres['W1']
    W2 = parametres['W2']
    B1 = parametres['B1']
    B2 = parametres['B2']
    
    Z1=W1.dot(X)+B1
    A1=1/(1+np.exp(-Z1))
    Z2=W2.dot(A1)+B2
    A2=1/(1+np.exp(-Z2))
    
    activations = {
        'A1' : A1,
        'A2' : A2
        }
    return activations



def back_propagation(X,y ,activations , parametres):
    
    W2 = parametres['W2']
    A1 = activations['A1']
    A2 = activations['A2']
    
    m = y.shape[1]
    
    dZ2 = A2 - y
    dW2=1/m*dZ2.dot(A1.T)
    dB2=1/m*np.sum(dZ2 , axis = 1, keepdims = True)
    
    dZ1 = np.dot(W2.T , dZ2)*A1*(1-A1)
    dW1=1/m*dZ1.dot(X.T)
    dB1=1/m*np.sum(dZ1 , axis = 1, keepdims = True)
    
    gradients = {
        'dW1' : dW1,
        'dB1' : dB1,
        'dW2' : dW2,
        'dB2' : dB2,
        }
    
    return gradients


def Log_loss(A,y):
    epsilon = 1e-15
    return 1/len(y)*np.sum(-y*np.log(A + epsilon) - (1-y)*np.log(1-A + epsilon))



def update(gradients, parametres, learning_rate):
    
    W1 = parametres['W1']
    W2 = parametres['W2']
    B1 = parametres['B1']
    B2 = parametres['B2']
    
    dW1 = gradients['dW1']
    dW2 = gradients['dW2']
    dB1 = gradients['dB1']
    dB2 = gradients['dB2']
    

    W1 = W1 - dW1*learning_rate
    B1 = B1 - dB1*learning_rate
    W2 = W2 - dW2*learning_rate
    B2 = B2 - dB2*learning_rate
    
    parametres = {
        'W1' : W1,
        'B1' : B1,
        'W2' : W2,
        'B2' : B2,  
        }
    
    return parametres

def predict(X , parametres):
    activations = forward_propagation(X, parametres)
    A2 = activations['A2']
    return A2>=0.5


def nural_network(x_train, y_train, n1,  learning_rate=0.1, n_iter=1000):
    
    #initialisation w, b
    n0 = x_train.shape[0]
    n2 = y_train.shape[0]
    parametres = Initialisation(n0 ,n1 ,n2)
    
    train_loss = []
    train_acc = []


    #boucle d'apprentissage
    for i in tqdm(range(n_iter)):
        
        activations = forward_propagation(x_train, parametres)
        gradients = back_propagation(x_train, y_train, activations, parametres )
        parametres = update(gradients , parametres , learning_rate)
        
        if i%10 == 0:
             #Train
             train_loss.append( Log_loss(y_train , activations['A2']))
             y_pred = predict(x_train , parametres)
             current_accuracy = accuracy_score(y_pred.flatten(), y_train.flatten())
             train_acc.append(current_accuracy)
             
  
    
    plt.figure(figsize=(12,4))
    
    plt.subplot(1,2,1)
    plt.plot(train_loss, label = 'train loss')
    plt.legend()
    
    plt.subplot(1,2,2)
    plt.plot(train_acc, label = 'train acc')
    plt.legend()
    
    plt.show()
    return parametres



def nural_network_test(x_train, y_train, x_test, y_test, n1, learning_rate=0.1, n_iter=100):
    
    #initialisation w, b
    n0 = x_train.shape[0]
    n2 = y_train.shape[0]
    parametres = Initialisation(n0 ,n1 ,n2)
    
    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []

    #boucle d'apprentissage
    for i in tqdm(range(n_iter)):
        
        activations = forward_propagation(x_train, parametres)
        activations_test = forward_propagation(x_test, parametres)
        gradients = back_propagation(x_train, y_train, activations, parametres )
        parametres = update(gradients , parametres , learning_rate)
        
        if i%10 == 0:
             #Train
             train_loss.append( Log_loss(y_train , activations['A2']))
             y_pred = predict(x_train , parametres)
             current_accuracy = accuracy_score(y_pred.flatten(), y_train.flatten())
             train_acc.append(current_accuracy)
             
             #Test
             test_loss.append( Log_loss(y_test , activations_test['A2']))
             y_pred = predict(x_test , parametres)
             current_accuracy = accuracy_score(y_pred.flatten(), y_test.flatten())
             test_acc.append(current_accuracy)
             

     
    
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(train_loss, label = 'train loss')
    plt.plot(test_loss , label = 'test loss')
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(train_acc, label = 'train acc')
    plt.plot(test_acc , label = 'test acc')
    plt.legend()
    plt.show()
    return parametres