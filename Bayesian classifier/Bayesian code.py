# -*- coding: utf-8 -*-
"""
Created on Tue May 25 19:08:02 2021

@author: Sumit
"""

import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB as GB
from sklearn.naive_bayes import MultinomialNB as MB
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

train_df = pd.read_csv("G:\\Data Science\\Assn 12 - Bayesian Classifier\\Python code\\SalaryData_Train.csv")
test_df = pd.read_csv("G:\\Data Science\\Assn 12 - Bayesian Classifier\\Python code\\SalaryData_Test.csv")

train_df.info()
# first we create dummy and things and make the data presentable to algos


### below shit is long thing coz the factrial datatype of 1 colllumns makes everything manual again
#num_col = [i for i in train_df.columns if train_df[i].dtype != 'O']
#char_col = [i for i in train_df.columns if train_df[i].dtype == 'O']
#fact_col = char_col
#for i in ["sex","race","native"]:
#    fact_col = fact_col.remove(i)




#making dummies for train data
dum_c =["workclass","education",'maritalstatus',"occupation","race","relationship","native"]
dumms_tr = pd.DataFrame()
for i in dum_c:
    dumm_cat = pd.get_dummies(train_df[i]).rename(columns=lambda x: i + "_" + str(x) )
    dumms_tr = pd.concat([dumms_tr,dumm_cat],axis=1)
    

#making dummies for test data

dumms_ts = pd.DataFrame()
for i in dum_c:
    dumm_cat = pd.get_dummies(train_df[i]).rename(columns=lambda x: i + "_" + str(x) )
    dumms_ts = pd.concat([dumms_ts,dumm_cat],axis=1)

# now mapping binary charector variable by 1 and 0

train_df1 = train_df
train_df1.drop(columns=dum_c,axis=1,inplace=True)
train_df1 = pd.concat([dumms_tr,train_df1],axis=1)

test_df1 = test_df
test_df1.drop(columns=dum_c,axis=1,inplace=True)
test_df1 = pd.concat([dumms_ts,test_df1],axis=1)


#### mapping the binary category variable
train_df1.sex = pd.Series(np.where(train_df.sex.values == " Male", 1, 0),train_df.index)
train_df1.Salary = pd.Series(np.where(train_df.Salary.values == " >50K", 1, 0),train_df.index)


test_df1.sex = pd.Series(np.where(test_df.sex.values == " Male", 1, 0),test_df.index)
test_df1.Salary = pd.Series(np.where(test_df.Salary.values == " >50K", 1, 0),test_df.index)


# here we have to train the data and complete its training and testing on train_df itself and then we have finally test on test_df

tr_df,ts_df = train_test_split(train_df1,test_size = 0.25,random_state = 13)

X_tr = tr_df.iloc[:,:101]
Y_tr = tr_df.iloc[:,101]


#building default models
gnb = GB()
mnb = MB()


# Building and predicting at the same time 

pred_X_tr = gnb.fit(X_tr,Y_tr).predict(X_tr) #train accuracy
pred_Y_tr = gnb.fit(X_tr,Y_tr).predict(ts_df.iloc[:,:101]) #test accuracy

# Confusion matrix GaussianNB model
confusion_matrix(Y_tr,pred_X_tr) # GaussianNB model
pd.crosstab(Y_tr.values.flatten(),pred_X_tr) # confusion matrix using 
np.mean(Y_tr.values.flatten()==pred_X_tr) # 100%


### now we need to find test accuracy then again train on whole model then give caccurace on completestdf1
test_df1.dropna(inplace=True)
test_df1.isnull().sum()
Pred = gnb.fit(train_df1.iloc[:,:101],train_df1.iloc[:,101]).predict(test_df1.iloc[:,:101])

np.mean(test_df1.iloc[:,101].values.flatten()==Pred) 
# 68%

Pred1 = mnb.fit(train_df1.iloc[:,:101],train_df1.iloc[:,101]).predict(test_df1.iloc[:,:101])

np.mean(test_df1.iloc[:,101].values.flatten()==Pred1) 

# 77 % accuracy for complete test data








