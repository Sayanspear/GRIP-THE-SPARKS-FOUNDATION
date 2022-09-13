#!/usr/bin/env python
# coding: utf-8

# # GRIP: The Sparks Foundation
# ## Data Science & Business Analytics Internship
# ## Task 2: Prediction Using Supervised Machine Learning
# ### Author: Sayan Das

# Importing Libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Reading and exploring the data set

# In[7]:


iris=pd.read_csv("Downloads\Iris.csv")
iris.head()


# In[4]:


iris.info()


# In[5]:


iris.describe().T


# In[6]:


iris["Species"].value_counts()


# ### So we conclude that there are no missing values and there are 3 unique species.
# ### From the describe function we can conclue that there are no outliers as well.

# Finding Optimum Number of Clusters fot K-Means Clustering

# In[9]:


from sklearn.cluster import KMeans


# In[10]:


X = iris.iloc[:, [0, 1, 2, 3]].values
Sum_of_squared_distances = []
K = range(1,10)
for num_clusters in K :
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(X)
    Sum_of_squared_distances.append(kmeans.inertia_)
plt.plot(K,Sum_of_squared_distances,"bx-")
plt.xlabel("Values of K") 
plt.ylabel("Sum of squared distances/Inertia") 
plt.title("Elbow Method For Optimal k")
plt.show()


# In[11]:


kmeans = KMeans(n_clusters = 3, init = 'k-means++',max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(X)


# ## Visualization 

# In[12]:


plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], 
            s = 100, c = 'black', label = 'Iris-setosa')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], 
            s = 100, c = 'blue', label = 'Iris-versicolour')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1],
            s = 100, c = 'red', label = 'Iris-virginica')

# Plotting the centroids of the clusters
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], 
            s = 100, c = 'yellow', label = 'Centroids')

plt.legend()


# # THANK YOU
