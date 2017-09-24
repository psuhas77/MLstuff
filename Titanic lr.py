import pandas as pd
import numpy as np
import logistic_reg as lr
import matplotlib.pyplot as plt
import math
path='C:\\ML\\test.csv'

data=pd.read_csv(path)


lis=['Pclass','Sex','Age','SibSp','Parch']

X=data.loc[:,lis]

X.insert(0,'ones',1)
X.insert(2,'male',0)
X.insert(3,'female',0)

for i in range(len(X)):
    if X.loc[i,'Sex']=='male':
        X.loc[i,'male']=1
    else:
        X.loc[i,'female'] =1  

for i in range(len(X)):
    if math.isnan(X.loc[i,'Age']):
        X.loc[i,'Age']=0


X.drop('Sex',axis=1,inplace=True)

X=np.matrix(X)
Y=np.matrix(data.loc[:,'Survived'])





theta=np.matrix([0.0001,-0.0023,0.002,0.02,0.12,0.87,0.009])
Y=Y.T


alpha=0.001
iters=100000
theta=theta.T
thetanew,costarr=lr.grad_desc(X,Y,theta.T,alpha,iters)

print(thetanew)



  
  
finalar=[]
for i in X:
    if i*(thetanew.T)>=0.5:
        finalar.append(1)

    else:
        finalar.append(0)


#print(finalar)
finalar=pd.DataFrame(finalar)
print(finalar)   
data2=data.loc[:,'PassengerId']
frames=[data2,finalar]
frames=pd.concat(frames,1)
print(frames)
path2='C:\\ML\\testres2.csv'
frames.to_csv(path2)