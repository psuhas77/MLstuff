import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt





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



path='C:\ML\ex2data1.txt'

df=pd.read_csv(path,header=None,names=['exam1','exam2','pass'])

passers=df.loc[df['pass']==1]
failers=df.loc[df['pass']==0]

#plt.scatter(passers.exam1,passers.exam2,s=5,c='green')
#plt.scatter(failers.exam1,failers.exam2,s=5,c='red',marker='v')
#plt.show()

df.insert(0,'ones',1)
X=np.matrix(df.iloc[:,0:3])
Y=np.matrix(df.iloc[:,3:])

theta=np.matrix([0.0001,-0.0023,0.002])
alpha=0.0001
iters=100000

print(theta.shape)
print(X.shape)
print(Y.shape)
#print(cost_func(X,Y,theta))
#thetanew,costarr=grad_desc(X,Y,theta,alpha,iters) 

#X_ax=np.arange(iters)
#plt.plot(X_ax,costarr)
#plt.show()


thetanew=np.matrix([-0.41526833,0.00872593,0.00719555])
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

print(numerror/(len(X)))
