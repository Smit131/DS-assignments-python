import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
"""
necessary transformations for input variables for getting better R^2 value for the model is to be prepared.
Delivery_time -> Predict delivery time using sorting time 
"""


############################################### Using statsmodels for Linear rgression model#####################################

df1 = pd.read_csv("G:\Data Science\Assn 4 - Linear Regression\Python code\Linear Regression\delivery_time.csv")
df1.columns = df1.columns.str.replace(" ","_")
df1.describe()
plt.figure(1)
plt.hist(df1["Delivery_Time"])

plt.figure(2)
plt.hist(df1["Sorting_Time"])

plt.figure(3)
plt.plot(df1["Delivery_Time"],df1["Sorting_Time"],"bo");plt.xlabel('Delivery_time');plt.ylabel("Sorting_time")
#there is a positive moderate strong relationship between them

print(df1["Delivery_Time"].corr(df1["Sorting_Time"]))
#0.82599
#there is near to strong correlationship (0.85 beibg the threshold for strong corr)
#thus we can use linear regression for prediction


np.corrcoef(df1["Delivery_Time"],df1["Sorting_Time"])
#above function can be used to find corr between multiple variables

import statsmodels.formula.api as smf
lr_model = smf.ols('Delivery_Time~Sorting_Time',data = df1).fit()

#Model parameters i.e. the coefficients of the equation
print(lr_model.params)

# Model summary to get the R2 values and r2 values 
print(lr_model.summary())
### R-squared:                       0.682

rmse = np.sqrt(np.mean((lr_model.resid)**2))
print("Rmse value is -> ",rmse)
#Rmse value is ->  2.7916503270617654

#stored in the resid attribute of the Results class
#Likewise there's a results.fittedvalues


lr_model1 = smf.ols('Delivery_Time~np.log(Sorting_Time)',data = df1).fit()
print(lr_model1.summary())#R-squared:                       0.695
rmse1 =np.sqrt(np.mean((lr_model1.resid)**2))
print("Rmse value is -> ",rmse1)# Rmse value is ->  2.733171476682066

lr_model2 = smf.ols('np.log(Delivery_Time)~Sorting_Time',data = df1).fit()
print(lr_model2.summary())#R-squared:                       0.711
tr_resid = np.exp(lr_model2.fittedvalues) - df1["Delivery_Time"]
rmse2 = np.sqrt(np.mean((tr_resid)**2))
print("Rmse value is -> ",rmse2)#Rmse value is ->  2.9402503230562007

lr_model3 = smf.ols('np.log(Delivery_Time)~np.log(Sorting_Time)',data = df1).fit()
print(lr_model3.summary())#R-squared:                       0.772
tr_resid = np.exp(lr_model3.fittedvalues) - df1["Delivery_Time"]
rmse3 = np.sqrt(np.mean((tr_resid)**2))
print("Rmse value is -> ",rmse3)#Rmse value is ->  2.745828897614548


lr_model21 = smf.ols('Delivery_Time~np.square(Sorting_Time)',data = df1).fit()
print(lr_model21.summary())#R-squared:                       0.630
rmse21 = np.sqrt(np.mean((lr_model21.resid)**2))
print("Rmse value is -> ",rmse21)#Rmse value is ->  3.0113772826462886

lr_model22 = smf.ols('np.square(Delivery_Time)~Sorting_Time',data = df1).fit()
print(lr_model22.summary())#R-squared:                       0.603
tr_resid = np.sqrt(lr_model22.fittedvalues) - df1["Delivery_Time"]
rmse22 = np.sqrt(np.mean((tr_resid)**2))
print("Rmse value is -> ",rmse22)#Rmse value is ->  2.7343261772614644

lr_model23 = smf.ols('np.square(Delivery_Time)~np.square(Sorting_Time)',data = df1).fit()
print(lr_model23.summary())#R-squared:                       0.588
tr_resid = np.sqrt(lr_model23.fittedvalues) - df1["Delivery_Time"]
rmse23 = np.sqrt(np.mean((tr_resid)**2))
print("Rmse value is -> ",rmse23)#Rmse value is ->  2.9072386947280413


# lowest Rmse is for lr_model1 when dependent variables where log ed 


"""
Salary_hike -> Build a prediction model for Salary_hike
"""
############################################# using linear regression model from sklearn library #######################################


df2 = pd.read_csv("G:\Data Science\Assn 4 - Linear Regression\Python code\Linear Regression\Salary_Data.csv")
from sklearn.linear_model import LinearRegression
model = LinearRegression()
X = df2["YearsExperience"].values.reshape(-1, 1)
Y = df2["Salary"]
model.fit(X,Y)
pred = model.predict(X)
print("Adjusted R-Squared value is",model.score(X,Y))#   0.9569566641435086
rmse =  np.sqrt(np.mean((pred - Y)**2))
print("Rmse value is -> ",rmse)#Rmse value is ->  5592.043608760662


model.fit(np.log(X),Y)
pred1 = model.predict(np.log(X))
print("Adjusted R-Squared value is",model.score(np.log(X),Y))#Adjusted R-Squared value is 0.853888882875697
rmse1 =  np.sqrt(np.mean((pred1 - Y)**2))
print("Rmse value is -> ",rmse1)#Rmse value is ->  10302.893706228302


model.fit(X,np.log(Y))
pred3 = model.predict(X)
pred3 = np.exp(pred3)
print("Adjusted R-Squared value is",model.score(X,np.log(Y)))#Adjusted R-Squared value is 0.9319671194084195
rmse3 =  np.sqrt(np.mean((pred3 - Y)**2))
print("Rmse value is -> ",rmse3)#Rmse value is ->  7213.235076620129


model.fit(np.log(X),np.log(Y))
pred2 = model.predict(np.log(X))
pred2 = np.exp(pred2)
print("Adjusted R-Squared value is",model.score(np.log(X),np.log(Y)))#Adjusted R-Squared value is 0.905215072581715
rmse2 =  np.sqrt(np.mean((pred2 - Y)**2))
print("Rmse value is -> ",rmse2)#Rmse value is ->  7219.716974372806



model.fit(np.sqrt(X),Y)
pred21 = model.predict(np.sqrt(X))
print("Adjusted R-Squared value is",model.score(np.sqrt(X),Y))#Adjusted R-Squared value is 0.9310009544993526
rmse21 =  np.sqrt(np.mean((pred21 - Y)**2))
print("Rmse value is -> ",rmse21)#Rmse value is ->  7080.095734983039

model.fit(X,np.sqrt(Y))
pred22 = model.predict(X)
pred22 = pred22**2
print("Adjusted R-Squared value is",model.score(X,np.sqrt(Y)))#Adjusted R-Squared value is 0.9498353533865289
rmse22 =  np.sqrt(np.mean((pred22 - Y)**2))
print("Rmse value is -> ",rmse22)#Rmse value is ->  5926.008666359513

model.fit(np.sqrt(X),np.sqrt(Y))
pred23 = model.predict(np.sqrt(X))
pred23 = pred23**2
print("Adjusted R-Squared value is",model.score(np.sqrt(X),np.sqrt(Y)))#Adjusted R-Squared value is 0.9419490138976825
rmse23 =  np.sqrt(np.mean((pred23 - Y)**2))
print("Rmse value is -> ",rmse23)#Rmse value is ->  5960.647096174308

########## the last model is the best model in terms of both adjusted R and rmse in which both input and output is squared
