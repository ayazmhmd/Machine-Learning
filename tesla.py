# -*- coding: utf-8 -*-
"""
Created on Sun Oct  3 22:26:08 2021

@author: Ayaz Mehmood
"""

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import explained_variance_score
from sklearn.metrics import r2_score

data = pd.read_csv('TSLA.csv')
data.drop_duplicates(inplace=True)
print(data['Adj Close'])
features=data.drop(['Close','Adj Close','Date'],axis=1)
print(features)
#features['Date']=pd.to_datetime(features['Date'])
labels=data['Close']
print(data)
X_train,X_test,Y_train,Y_test=train_test_split(features, labels, random_state=0, test_size=0.33)

min_max_scaler =MinMaxScaler()
min_max_scaler.fit_transform(X_train)
X_train= min_max_scaler.transform(X_train)
X_test=min_max_scaler.transform(X_test)

lr=LinearRegression()

lr.fit(X_train,Y_train)

predict_val=lr.predict(X_test)
print()
a=explained_variance_score(Y_test,predict_val, multioutput='uniform_average')
b=r2_score(Y_test,predict_val)
print(b)

## check online how to implement linear regression.
