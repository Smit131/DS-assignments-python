
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split


df =pd.read_csv("G:\\Data Science\\Assn 15 - Neural Network\\Python code\\50_Startups.csv")

df.info()
df.isnull().sum()

# as state in categorical we need to make it numerical using dumdums

dum = pd.get_dummies(df["State"]).rename(columns = lambda x : "State_" + str(x))

df1 = pd.concat([dum,df],axis = 1)
df1.drop(["State"],axis=1,inplace=True)
# we have stardize the values before putting it in the model

xy = StandardScaler()
# we need to fit the model according to Xtrain....idk exactly why... i think it can done using ytest also or data of X also
xy.fit(df1.iloc[:,:6])

tr_x,ts_x,tr_y,ts_y = train_test_split(df1.iloc[:,:6],df1.iloc[:,6], test_size = 0.2, random_state = 13)

tr_x = xy.transform(tr_x)
ts_x = xy.transform(ts_x)

########          applying NN

NN = MLPClassifier(hidden_layer_sizes=(15,15))

## again the the mlp is detecting labes as not int soooo
tr_y = tr_y.astype("int")
ts_y = ts_y.astype('int')

M1 = NN.fit(tr_x,tr_y)
pred_tr = NN.predict(tr_x)
pred_ts = NN.predict(ts_x)

##### train accuracy
np.corrcoef(tr_y,pred_tr)
#   0.831
rmse_tr = np.sqrt(np.mean((pred_tr - tr_y)**2))
#   9832.460

###### test accuracy
np.corrcoef(ts_y,pred_ts)
#   0.68
rmse_ts = np.sqrt(np.mean((pred_ts - ts_y)**2))
#   14361.15