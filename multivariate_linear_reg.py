import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def cost_func(X,Y,theta):
    return (np.sum(np.square(X*theta-Y)))/(2*len(X))


def grad_desc(X,Y,theta,alpha,iters):
    costarr=[]
    for i in range(iters):
        val1=X*theta-Y
        val2=((X.T)*val1)/(len(X))
        theta=theta-alpha*(val2)
        costarr.insert(i,cost_func(X,Y,theta))

    return theta,costarr


path='C:\ML\ex1data2.txt'

df=pd.read_csv(path,header=None,names=['size','room','price'])
df=(df-df.mean())/(df.std())
df.insert(0,'Ones',1)


cols=df.shape[1]

X=np.matrix(df.iloc[:,0:cols-1])

Y=np.matrix(df.iloc[:,cols-1:])

theta=np.matrix(np.zeros(cols-1))
theta=theta.T
alpha=0.01
iters = 1000
theta,cost=grad_desc(X,Y,theta,alpha,iters)
plt.plot(np.arange(iters),cost)
plt.show()

