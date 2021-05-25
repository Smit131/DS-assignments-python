# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 15:23:42 2021

@author: Sumit
"""

import pandas as pd
import numpy as np
import scipy.cluster.hierarchy as sch          # to create dedogram
from scipy.cluster.hierarchy import linkage    # to define the type of linkage like complete linkage or single linkage for dendogram 
import matplotlib.pyplot as plt
from	sklearn.cluster	import	AgglomerativeClustering        # to create the actual clusters in dataset and saving em



df = pd.read_csv("G:\\Data Science\\Assn 7 - Clustering\\Python Code\\crime_data1.csv")
df.info()

len(df.City.unique())

# user based function to Stadardize the data
def stdard_func(i):
    x = (i - i.mean())/(i.std())
    return(x)

# normalzing data except the city names . the citi names acts as ID as every entry has unique city name
df_norm = stdard_func(df.iloc[:,1:])

help(linkage)
z = linkage(df_norm,method = "complete", metric = "euclidean")

# ploting the ded0gram
plt.figure(figsize=(15,5));plt.title("Dedogram figure");plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(    z,
    leaf_rotation=0.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)

##### from the figure we can see there that 4 major clusters can be made
##### now while clustering we will put 4 as value to form clusters

H_complete = AgglomerativeClustering(n_clusters=4, linkage = "complete",affinity = "euclidean").fit(df_norm)


cluster_labels=pd.Series(H_complete.labels_)
df["Cluster_Group_no"] = cluster_labels
df = df.iloc[:,[5,0,1,2,3,4]]

### Displaying mean statistic associated with each clusters
df.iloc[:,2:].groupby(df.Cluster_Group_no).mean()

### from the above output we can see group three has the least ammount of crime rate
### thus the group can be said to be safe cities
# Sorting the cities according to their groups
df_new = df.sort_values(by = ['Cluster_Group_no'])   

# storing the clustered file
df_new.to_csv("G:\\Data Science\\Assn 7 - Clustering\\Python Code\\H_Clustered_cities.csv", index= False)
