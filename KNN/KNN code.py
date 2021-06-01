# -*- coding: utf-8 -*-
"""
Created on Tue May 25 14:32:02 2021

@author: Sumit
"""

import pandas as pd
import numbpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

df = pd.read_csv("G:\\Data Science\\Assn 11 - KNN (nearest neighbour)\\Python code\\Zoo.csv")
df.info()
## we dont need animal name so dropping that colllum
df.drop(columns="animal name",inplace=True)

train_df,test_df = train_test_split(df,test_size = 0.2)
X_tr = train_df.iloc[:,:16]
Y_tr = train_df.iloc[:,16]

X_ts = test_df.iloc[:,:16]
Y_ts = test_df.iloc[:,16]

########## using k nearest neighbour now

from sklearn.neighbors import KNeighborsClassifier as KNC 

nh = KNC(n_neighbors=3)

nh.fit(X_tr,Y_tr)

#train accuracy
train_acc = np.mean(nh.predict(X_tr)==Y_tr)
#0.9625

#test_accuracy
test_acc = np.mean(nh.predict(X_ts)==Y_ts)
#0.9523

# lets have look at plot of model accuracy vs complexity
Acc_list = []
for k in range(2,10,1):
    nh = KNC(n_neighbors=k)
    nh.fit(X_tr,Y_tr)
    train_acc = np.mean(nh.predict(X_tr)==Y_tr)
    test_acc = np.mean(nh.predict(X_ts)==Y_ts)
    Acc_list.append([train_acc,test_acc,k])

plt.plot(np.arange(2,10,1),[i[0] for i in Acc_list],'bo-')
plt.plot(np.arange(2,10,1),[i[1] for i in Acc_list],'ro-')
plt.legend(["Test","train"])    