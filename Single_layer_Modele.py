import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from tqdm import tqdm


def Initialisation(x):
    w=np.random.randn(x.shape[1],1)
    b=np.random.randn(1)
    return(w,b)


def model(X,w,b):
    z=X.dot(w)+b
    A=1/(1+np.exp(-z))
    return A


def Log_loss(A,y):
    epsilon = 1e-15
    return 1/len(y)*np.sum(-y*np.log(A + epsilon) - (1-y)*np.log(1-A + epsilon))

def gradients(A,X,y):
    dw=1/len(y)*np.dot(X.T,A-y)
    db=1/len(y)*np.sum(A-y)
    return (dw,db)

def update(dw, db, w, b, learning_rate):
    w = w - dw*learning_rate
    b = b - db*learning_rate
    return (w , b)

def predict(X , w ,b):
    A = model(X, w ,b)
    return A>=0.5


def artificial_neuron(x_train, y_train,  learning_rate=0.1, n_iter=100):
    #initialisation w, b
    w, b = Initialisation(x_train)
    
    train_loss = []
    train_acc = []


    #boucle d'apprentissage
    for i in tqdm(range(n_iter)):
        A = model(x_train, w ,b)
        if i%10 == 0:
             #Train
             train_loss.append( Log_loss(A , y_train))
             y_pred = predict(x_train , w ,b)
             train_acc.append(accuracy_score(y_train,y_pred))
             

             
        #mise a jour
        dw , db = gradients(A , x_train , y_train)
        w , b = update(dw , db , w , b , learning_rate)
     
    
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(train_loss, label = 'train loss')
    plt.legend()
    
    plt.subplot(1,2,2)
    plt.plot(train_acc, label = 'train acc')
    plt.legend()
    plt.show()
    return (w , b)



def a_neuron_train_and_test(x_train, y_train, x_test, y_test,  learning_rate=0.1, n_iter=100):
    #initialisation w, b
    w, b = Initialisation(x_train)
    
    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []

    #boucle d'apprentissage
    for i in tqdm(range(n_iter)):
        A = model(x_train, w ,b)
        if i%10 == 0:
             #Train
             train_loss.append( Log_loss(A , y_train))
             y_pred = predict(x_train , w ,b)
             train_acc.append(accuracy_score(y_train,y_pred))
             
             #Test
             A_test = model(x_test, w ,b)
             test_loss.append( Log_loss(A_test , y_test))
             y_pred = predict(x_test , w ,b)
             test_acc.append(accuracy_score(y_test,y_pred))
             
        #mise a jour
        dw , db = gradients(A , x_train , y_train)
        w , b = update(dw , db , w , b , learning_rate)
     
    
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
    return (w , b)


def A_neuron_normalisation(x_train, y_train,  learning_rate=0.1, n_iter=100):
    #initialisation w, b
    w, b = Initialisation(x_train)
    w[0] , w[1] = -7.5 , -7.5
    
    nb=10
    j=0
    history = np.zeros((n_iter // nb , 5))
    
    loss = []
    


    #boucle d'apprentissage
    for i in tqdm(range(n_iter)):
        A = model(x_train, w ,b)
        loss.append( Log_loss(A , y_train))
        dw , db = gradients(A , x_train , y_train)
        w , b = update(dw , db , w , b , learning_rate)
        
        if i%nb == 0:
            history[j,0]=w[0] 
            history[j,1]=w[1] 
            history[j,2]=b 
            j+=1
            
    plt.plot(loss )
    plt.show()
    return history , b