# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 18:36:11 2021

@author: Sumit
"""

import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics     # for ROC curve plotting 



df = pd.read_csv('G:\\Github\\DS-assignments-python\\Logistic Regression\\affairs.csv')
df.head()
print(df.columns)
df0 = df
j=0
for i in df.affairs:
    if i>0:
        df0.iloc[j,0] = 1
        j+=1
    else:
        df0.iloc[j,0] = 0
        j+=1
        

# Getting the barplot for the categorical columns 
sb.countplot(x="age",data=df0,palette="hls")

pd.crosstab(df0.age,df0.gender).plot(kind="bar")


# looking for missing values

df.isnull().sum()
# gives collumn vise sum of missing values 
# no missing value found if found then we do ->
###### df["colname"].fillna(1,inplace=True)

# creating dummy variable for gender and childern
dumm_gen = pd.get_dummies(df0["gender"]).rename(columns = lambda x: "gender_" + str(x))
dumm_chi = pd.get_dummies(df0["children"]).rename(columns= lambda x : "children_" + str(x))

df1 = pd.concat([df0,dumm_gen,dumm_chi],axis =1)
df1.drop(["gender","children","gender_female","children_no"],axis =1 , inplace = True)
df1.head(5)
df1.columns

## train and test split
from sklearn.model_selection import train_test_split

train,test = train_test_split(df1,test_size=0.2)

#test.isnull().sum()
## preparing data for model
#df1.shape
#       df_train = df_train.reset_index().set_index('A')
#       to reset index after spliting

train = train.reset_index()
test = test.reset_index()
train.drop("index",axis = 1,inplace = True)
test.drop("index",axis = 1,inplace = True)

X_train = train.iloc[:,1:]
Y_train = train.iloc[:,0]

X_test = test.iloc[:,1:]
Y_test = test.iloc[:,0]

## prepeaireing logistic model for the data

m1 = LogisticRegression(solver='lbfgs', max_iter=1000)
m1.fit(X_train,Y_train)

m1.coef_ # coefficients of features 
pred_prob = m1.predict_proba (X_train) # Probability values 

## using probabilty values to create custom threshold point for prediction of 1 or 0 
pred_prob_df = pd.DataFrame(data = pred_prob,columns = ["Probability0","Probability1"])
prob_train = pred_prob_df.iloc[:,1]

prob_train = pd.DataFrame(prob_train)
prob_train["Predd"] = np.zeros(480)


## custom prediction
prob_train.loc[prob_train.Probability1>0.35,"Predd"] = 1

## Cross table
# confusion matrix 
confusion_matrix = pd.crosstab(Y_train,prob_train.Predd)
#accuracy
#confusion_matrix.sum().sum()
accuracy_mat = (confusion_matrix.iloc[0,0]+confusion_matrix.iloc[1,1])/confusion_matrix.sum().sum()
print(round(accuracy_mat,4)*100,"%")


# ROC curve 
#from sklearn import metrics
# fpr => false positive rate
# tpr => true positive rate
fpr, tpr, threshold = metrics.roc_curve(Y_train,prob_train.Probability1)

# the above function is applicable for binary classification class 

plt.plot(fpr,tpr);plt.xlabel("False Positive");plt.ylabel("True Positive")
print(threshold)
roc_auc = metrics.auc(fpr, tpr) # area under ROC curve 
print(roc_auc)
#0.63 aoc that means not a great job by model but accuracy is great so idk



