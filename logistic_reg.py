import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



def sigmoid(z):
    return 1/(1+np.exp(-z))
    
    


def cost_func(X,Y,theta):
      
    h=sigmoid(X*theta)   
    return np.sum(Y.T*(np.log(h)) + (1-Y).T*(np.log(1-h)))/(-len(X))


def grad_desc(X,Y,theta,alpha,iters):
    costarr=[]
    
    theta=theta.T
    for i in range(iters):
        val1=X*theta-Y
        val2=((X.T)*val1)/(len(X))
        theta=theta-alpha*(val2)
        costarr.insert(i,cost_func(X,Y,theta))

    return theta,costarr


def accuracy(X,Y,thetanew):

    finalar=[]
    for i in X:
        if i*(thetanew.T)>=0.5:
            finalar.append(1)

        else:
            finalar.append(0)

    finalar=np.matrix(finalar).T
    error=finalar-Y
    numerror=0
    for i in error:
        if i!=0:
            numerror+=1

    acc=numerror/(len(X))
    return acc,finalar
    
    
    
