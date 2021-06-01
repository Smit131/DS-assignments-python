import pandas as pd 
import numpy as np
#import sklearn as skl
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from mpl_toolkits import mplot3d
from sklearn.cluster import KMeans

wdf = pd.read_csv("G:\\Data Science\\Assn 8 - PCA\\Python code\\wine.csv")
wdf.info()
wdf.describe()
plt.boxplot(wdf.Proline)
# need to find boxplot method for plotting all collumns 's boxplot in one figure
plt.boxplot(wdf)


#dropping the type collumns coz i thinks its already clusters dataset
wdf_n = wdf.iloc[:,1:]
#wdf_n.head(5)

#first we need to scale the data ...normalizing the data
wdf_nn = scale(wdf_n)

#lets plot the PClines

pcl = PCA(n_components=13)
pcl_values = pcl.fit_transform(wdf_nn)                  #these are the weights assigned (check it)

# the praportion of variance explained by each line is ->
pc_cap = pcl.explained_variance_ratio_
pc_cap
pcl.components_[0]

#only about 67% of the data is captured by pca1 ,pc2,pc3..
# thus datapoints has low homegenuity
pc_cap_perse = np.cumsum(np.round(pc_cap,decimals=4)*100)
pc_cap_perse

#variance plot for pca components?????
plt.plot(pc_cap_perse,color="red")

#ploting on pc1 and pc2

x = pcl_values[:,0]
y = pcl_values[:,1]
z = pcl_values[:,2]
plt.figure()
plt.scatter(x,y,c="red")

#from mpl_toolkits.mplot3d import Axes3D
#Axes3D.scatter(np.array(x),np.array(y),np.array(z),c=["green","blue","red"])
#gives error
#below one is easier and works great

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(x, y, z, color = 'red')

# it looks like 3 clusters can be formed but beware this is for only 66% of the data
# applying k - means 3 at the end to see things rooling

m1 = KMeans(n_clusters=3)
m1.fit(wdf_nn)

labels = pd.Series(m1.labels_)
wdf["Cluster number"] = labels

### Displaying mean statistic associated with each clusters
wdf.iloc[:,1:14].groupby(wdf["Cluster number"]).mean()

# the clusters formed doesn't seem to be significant as mentioned by the variance thingy above in PCs