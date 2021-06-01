
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

df = pd.read_csv("G:\\Data Science\\Assn 16 - SVM\\Python code\\forestfires.csv")

df.info()

# in the data for month and day already the dummy variables as present ...so we just neeed to drop the categry type ones
df.drop(["month","day"],axis = 1, inplace=True)

# we need convert the output category o.e size to numerical 1 and 0
df.size_category = pd.Series(np.where(df.size_category == "large",1,0),df.index) 

df.info()
#all are in numerical now

## building the svm model..

tr_x,ts_x,tr_y,ts_y = train_test_split( df.iloc[:,:28], df.iloc[:,28], test_size = 0.25, random_state = 13)

## creating svm models which includes - 
## linear , poly ,rbf , sigmoid.....


# linear
m1 = SVC(kernel="linear")
m1.fit(tr_x,tr_y)

pred_tr = m1.predict(tr_x)
pred_ts = m1.predict(ts_x)

# Train accuracy
np.mean(tr_y == pred_tr)
# 100%

# Test accuracy
np.mean(ts_y == pred_ts)
# 98.46%


# poly
m1 = SVC(kernel="poly")
m1.fit(tr_x,tr_y)

pred_tr = m1.predict(tr_x)
pred_ts = m1.predict(ts_x)

# Train accuracy
np.mean(tr_y == pred_tr)
# 77%

# Test accuracy
np.mean(ts_y == pred_ts)
#76.92%


#RBF
m1 = SVC(kernel="rbf")
m1.fit(tr_x,tr_y)

pred_tr = m1.predict(tr_x)
pred_ts = m1.predict(ts_x)

# Train accuracy
np.mean(tr_y == pred_tr)
# 76.22%

# Test accuracy
np.mean(ts_y == pred_ts)
# 73.84%

######Linear SVM is giving the max accuracy####