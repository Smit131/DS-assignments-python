
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("G:\\Data Science\\Assn 13 - Decision Tree\\Python code\\Fraud_check.csv")
df.info()
df.isnull().sum()
# no null

##### here we are treating those who have taxable_income <= 30000 as "Risky" and others are "Good"
###ALSO OTHER TABLES CONVERSION TO NUMERICAL DATA
df["Taxable.Income"] = pd.Series(np.where(df["Taxable.Income"]<=30000,1,0),df.index)
df.Undergrad = pd.Series(np.where(df.Undergrad =="YES",1,0),df.index)
df.Urban = pd.Series(np.where(df.Urban =="YES",1,0),df.index)
dum_m = pd.get_dummies(df["Marital.Status"]).rename(columns = lambda x: "Marital_Status_" + str(x) )
df = pd.concat([df,dum_m],axis=1)
df.drop(["Marital.Status"],axis=1, inplace=True)
df.info()

## rearranging the collumn to the end
df1 = df[[i for i in df.columns if i != 'Taxable.Income'] + ["Taxable.Income"]]

df1.info()
df["Taxable.Income"].value_counts(normalize=True)

tr_df,ts_df = train_test_split(df1 , test_size = 0.2 , random_state = 13)

tr_df["Taxable.Income"].value_counts(normalize=True)
# here after the split the proportion of output category is nearly same for train and test data ...
# so no need to do the shinanigan of splitting data for output category taking praportion and then joining and all
# like doing things like -->
# #df2 = df1[df1["Taxable.Income"] == 'Risky']

tr_x = tr_df.iloc[:,:7]
tr_y = tr_df.iloc[:,7]


# building the model for decision tree
m1 = DecisionTreeClassifier(criterion="entropy")
m1.fit(tr_x,tr_y)

#Prediction on training set
pred_tr = m1.predict(tr_x)

#predicting on test data
pred_ts = m1.predict(ts_df.iloc[:,:7])

#checking the crosstable
pd.crosstab(tr_y,m1.predict(tr_x))

## finding the accuracy
# Train accuracy - 
np.mean(pred_tr == tr_y)
# 100%
#test accuracy
np.mean(pred_ts == ts_df.iloc[:,7])
# 75%
pd.crosstab(ts_df.iloc[:,7],pred_ts)
