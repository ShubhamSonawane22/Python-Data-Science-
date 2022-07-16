#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
import numpy as nm 
import scipy.cluster.hierarchy as sch 
from sklearn.cluster import AgglomerativeClustering 


# In[3]:


Univ=pd.read_csv("Universities.csv")


# In[4]:


Univ


# In[11]:


# Function Normalization for data standarddization 
# Using mean max method

def norm_func(i):
    x=(i-i.mean())/(i.max()-i.min())
    return (x)


# In[13]:


# Normalize the data
df_norm=norm_func(Univ.iloc[:,1:])


# In[15]:


df_norm


# In[19]:


# Creation of Dendograme 
dendogram=sch.dendrogram(sch.linkage(df_norm,method="complete"))


# In[20]:


dendrogram=sch.dendrogram(sch.linkage(df_norm,method="single"))


# In[21]:


dendrogram=sch.dendrogram(sch.linkage(df_norm,method="centroid"))


# In[25]:


dendrogram=sch.dendrogram(sch.linkage(df_norm,method="complete"))


# In[30]:


# Create Clustering 
hc=AgglomerativeClustering(n_clusters=3,affinity='euclidean',linkage="complete")


# In[31]:


y_hc=hc.fit_predict(df_norm)


# In[32]:


Univ["Cluster"]=y_hc


# In[33]:


Univ


# In[1]:


# K-Means 


# In[40]:


import pandas as pd
import numpy as nm 
import matplotlib.pyplot as plt 
from scipy.spatial.distance import cdist 
from sklearn.cluster import KMeans


# In[41]:


Univ=pd.read_csv("Universities.csv")


# In[42]:


Univ.head()


# In[43]:


def norm_func(i):
    x=(i-i.mean())/(i.max()-i.min())
    return (x)


# In[44]:


df_norm=norm_func(Univ.iloc[:,1:])


# In[45]:


df_norm.head()


# In[19]:


#  Elbow Curve for selction of Clusters 


# In[49]:


#selecting 5 clusters (k=4)


# In[51]:


model=KMeans(n_clusters=4)
model.fit(df_norm)


# In[52]:


model.labels_


# In[53]:


md=pd.Series(model.labels_)
Univ["Clust"]=md


# In[54]:


Univ


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




