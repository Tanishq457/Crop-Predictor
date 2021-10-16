# Importing the libraries
import numpy as np
import pandas as pd
import pickle

from scipy import stats
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('final.csv')
X = df.iloc[:,:]
  #Detect and handle Outliers using ZScore



z = np.abs(stats.zscore(X.iloc[:,2]))   #Do for PH
cols = np.where(z > 2.2)
X.drop(index=cols[0],inplace=True)

temp1 = X.reset_index()
X = temp1
X.drop(columns=['index'],inplace=True)

newX = X.iloc[:,0:4]
Y = X.iloc[:,4]


for i in range(len(Y)):
  Y[i] = Y[i].lower()


newY = pd.get_dummies(Y).iloc[: , 0:]

crops=['wheat','mungbean','tea','millet','rice','maize','lentil','jute','cofee','cotton','ground nut','peas','rubber','sugarcane','tobacco','kidney beans','moth beans','coconut','blackgram','adzuki beans','pigeon peas','chick peas','banana','grapes','apple','mango','muskmelon','orange','papaya','watermelon','pomegranate']
crops.sort()
data= pd.concat([X,newY],axis=1)
data.drop('label', axis=1,inplace=True)

train=data.iloc[:, 0:4].values
test=data.iloc[: ,4:].values


X_train,X_test,Y_train,Y_test=train_test_split(train,test,random_state=42)

clf=RandomForestClassifier(n_estimators=400,criterion='entropy')
clf.fit(X_train,Y_train)


def cropPredictor(p):
    for i in range(0,31):
        if(p[0][i]==1):
            c=crops[i]
            break
    print('The predicted crop is %s'%c)



pickle.dump(clf, open('Model.pkl','wb'))
