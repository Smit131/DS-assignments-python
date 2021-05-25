# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 17:26:30 2021

@author: Sumit
"""

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn.cluster	import	KMeans  #for k clustering
from scipy.spatial.distance import cdist   #for caluclating euclidean  distances between observations in n-dimensional space.


df = pd.read_csv("G:\\Data Science\\Assn 7 - Clustering\\Python Code\\crime_data1.csv")
df.info()

def norm_func(i):
    x = (i - i.mean())/(i.std())
    return(x)

# normalzing data except the city names . the citi names acts as ID as every entry has unique city name
df_norm = norm_func(df.iloc[:,1:])

## trying scree plot/ elbow plot

#Now we will apply kemans for different amounts of clusters and store there WSS and TSS in a list for plotting

k = list(range(2,15)) #includes 2 and doesnt include 15

SWSS = [] # sum of WSS
for i in k:
    k1 = KMeans(n_clusters = i)
    k1.fit(df_norm)
    print(k1)
    WSS = []
    for j in range(i):
        WSS.append(sum(cdist(df_norm.iloc[k1.labels_ == j,:],k1.cluster_centers_[j].reshape(1,df_norm.shape[1]),"euclidean")))
    SWSS.append(sum(WSS))
    

# Scree plot 
plt.plot(k,SWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS");plt.xticks(k)

# so elbow at 4 and 5 , so we cant take 4 or 5

# Selecting 4 clusters from the above scree plot which is the optimum number of clusters 
model=KMeans(n_clusters=4)
model.fit(df_norm)

model.labels_ # getting the labels of clusters assigned to each row 
md=pd.Series(model.labels_)

df["Cluster_Group_no"] = md
df = df.iloc[:,[5,0,1,2,3,4]]

### Displaying mean statistic associated with each clusters
df.iloc[:,2:].groupby(df.Cluster_Group_no).mean()

### from the above output we can see group three has the least ammount of crime rate
### thus the group can be said to be safe cities
# Sorting the cities according to their groups
df_new = df.sort_values(by = ['Cluster_Group_no'])   

# storing the clustered file
df_new.to_csv("G:\\Data Science\\Assn 7 - Clustering\\Python Code\\K_means_Clustered_cities.csv", index= False)