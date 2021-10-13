# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 18:13:55 2021


@author: Sumit
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt   ## for plotting boxplots 
import seaborn as sns ## for plotting correlation graphs and matrices
import statsmodels.formula.api as smf ## to using linear regression model
import statsmodels.api as sm  ## to use function to find the datapoints that influence the models the most
#from sklearn.linear_model import LinearRegression

 
df = pd.read_csv("G:\Data Science\Assn 5 - Multilinear Regression\Python code\Multilinear regression\Startups.csv")
df.head()
df.describe()

dummies = pd.get_dummies(df["State"]).rename(columns =lambda x: "State_" + str(x))
print(df.head(),df.shape)

df1 = pd.concat([df.loc[:,df.columns!="State"],dummies],axis = 1)
#print(df.loc[:,df.columns!="State"])
df1.columns = df1.columns.str.replace(' ','_')
df1.rename(columns={"R&D_Spend":"RnD_Spend"},inplace=True)
print(df1.head())
print(df1.columns,df1.shape)

for i in range(7):
    plt.figure()
    plt.boxplot(df1.iloc[:,i])
#Data has no outliers

# checkingfor 3 conditions before applying the model
""" 1. Predictors must be non random (the predictors should logically[commensense wise] make sense to be used as predictor for the given predicted variable)
    2. Predictors are measured without error
    3. Predictors which be independent of each other - not then COLLINEARITY problem
"""
# cheking for dependent variable independency

print(df1.corr())
sns.pairplot(df1)

# from the plot we can see profit and R&D spend shows strong positive correlation
# Thus we have COLLINEARITY proble m
#df1.corr(df1["R&D Spend"],df1["Profit"])
df1["RnD_Spend"].corr(df1["Profit"])
#np.corrcoef(df1["R&D Spend"],df1["Profit"])

df1["RnD_Spend"].corr(df1["Marketing_Spend"])
# R&D spend and Marketing also have good correlation between

#prepairing models for all the variables
#preparing string to put in the linear regression model
k = (df1.columns.values.tolist())
x1 = [i for i in k if i != 'Profit']
s = "+".join(x1)
lmnam = "Profit" + "~" + s
print(lmnam)

model1 = smf.ols(lmnam ,data=df1).fit()

# For getting coefficients of the varibles used in equation
model1.params

# P-values for the variables and R-squared value for prepared model
print(model1.summary())
##  Marketing_Spend , Administration are the only two insignificant
##  R-squared:                       0.951
##  Adj. R-squared:                  0.945

########### before deleting the columns and checking is there in change in significance we will first 


sm.graphics.influence_plot(model1)
# obsvervation 49, 48 and 46 influenses the most 
# lets see after removes does variables beco,es significant

df2 = df1.drop(df1.index[[49,48,46]],axis = 0)
model2 = smf.ols(lmnam,data = df2).fit()
print(model2.summary())
## still the variables remain insignificant
## R-squared:                       0.961
## Adj. R-squared:                  0.957

######################### so now we perform VIF on the variable to understand the degree of of collinearity
######################### they are having with output varibale 

# Calculating vif value for administration
k = (df1.columns.values.tolist())
x1 = [i for i in k if i != 'Profit' and i!="Administration"]
s = "+".join(x1)
sig1 = "Administration" + "~" + s
print(sig1)

rsq_adm = smf.ols(sig1,data = df2).fit().rsquared
vif_adm = 1/(1 - rsq_adm)
print("VIF value for Administration is ",vif_adm)
#### VIF value for Marketing Spend is  1.2373952119564537


## Calculating VIF value for Marketing spend
k = (df1.columns.values.tolist())
x1 = [i for i in k if i != 'Profit' and i!="Marketing_Spend"]
s = "+".join(x1)
sig2 = "Marketing_Spend" + "~" + s
print(sig2)

rsq_ms = smf.ols(sig2,data = df2).fit().rsquared
vif_ms = 1/(1 - rsq_ms)
print("VIF value for Marketing Spend is ",vif_ms)
#### VIF value for Marketing Spend is  2.7092939446787123

# since VIF for Marketing spend is higher we can remove that to see if model performs better 
############Multicollinearity can be detected via various methods. In this article, we will focus on the most common one – VIF (Variable Inflation Factors).
##########” VIF determines the strength of the correlation between the independent variables. It is predicted by taking a variable and regressing it against every other variable. “
############### the more the slope is horizontle the low is the ammount of contribution it does
## VIF = 1, no correlation between the independent variable and the other variables
## VIF exceeding 5 or 10 indicates high multicollinearity between this independent variable and the others


### VIF score of an independent variable represents how well the variable is explained by other independent variables.
k = (df1.columns.values.tolist())
x1 = [i for i in k if i != 'Profit' and i!="Marketing_Spend"]
s = "+".join(x1)
m3 = "Profit" + "~" + s
print(m3)

model3 = smf.ols(m3,data = df2).fit()
print(model3.summary()) 
## R-squared:                       0.959
## Adj. R-squared:                  0.955
### accuracy only improved by 0.8%

# Added varible plot 
plt.figure()
sm.graphics.plot_partregress_grid(model2)
### model shows that all of the dummy varible factors doesnt contribute towards output


k = (df1.columns.values.tolist())
x1 = [i for i in k if i != 'Profit' and i!="State_California" and i!="State_Florida"  and i!="State_New_York"]
s = "+".join(x1)
m4 = "Profit" + "~" + s
print(m4)

model4 = smf.ols(m4,data = df2).fit()
print(model4.summary()) 
# R-squared:                       0.961
# Adj. R-squared:                  0.959
#print(df2.columns)
### model4 is giving the highest R square calue and adjusted rsquare value
sm.graphics.plot_partregress_grid(model4)
m4_pred = model4.predict(df2)

plt.figure()
plt.scatter(df2.Profit,m4_pred,c="r"); plt.xlabel("observed Profit");plt.ylabel("Fitted profit")



plt.figure()
plt.hist(model4.resid_pearson)

import pylab          
import scipy.stats as st

# Checking Residuals are normally distributed
plt.figure()
st.probplot(model4.resid_pearson, dist="norm", plot=pylab)
# errors should be normally distributed

# checking the homoscedacity
#variance 
plt.figure()
plt.scatter(m4_pred,model4.resid_pearson,c="r"),plt.axhline(y=0,color='blue');plt.xlabel("fitted_values");plt.ylabel("residuals")
# the upper limit and the lower limit must be same in the figure that means the error dont have the homo problem


#splitting test and train data
from sklearn.model_selection import train_test_split
x_train,x_test  = train_test_split(df2,test_size = 0.2) # 20% size
 
# preparing the model on train data 

model_train = smf.ols(m4,data=df2).fit()

# train_data prediction
train_pred = model_train.predict(x_train)

# train residual values 
train_resid  = train_pred - x_train.Profit

# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid*train_resid))
print("Train rmse is    ",train_rmse)

# prediction on test data set 
test_pred = model_train.predict(x_test)

# test residual values 
test_resid  = test_pred - x_test.Profit

# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid*test_resid))
print("Test rmse is    ",test_rmse)

"""
Train rmse is     7188.9598214373245
Test rmse is     6102.913730542877

"""