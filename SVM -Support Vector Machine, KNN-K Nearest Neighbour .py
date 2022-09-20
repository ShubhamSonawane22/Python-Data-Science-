#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing data and module 


# In[2]:


import pandas as pd
import numpy as np
from sklearn import datasets 


# In[6]:


iris=datasets.load_iris()


# In[10]:


iris


# In[15]:


#Defining petal length and petal width for analysis 
X=iris.data[:,[2,3]]
Y=iris.target


# In[17]:


#Placing the data into Pandas dataframe 
iris_df=pd.DataFrame(iris.data[:,[2,3]],columns=iris.feature_names[2:])


# In[19]:


iris_df.head()


# In[42]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=0)


# In[43]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
sc.fit(X_train)
X_train_std=sc.transform(X_train)
X_test_std=sc.transform(X_test)


# In[44]:


X_train_std


# In[45]:


from sklearn.svm import SVC
svm=SVC(kernel="rbf",random_state=0,gamma=0.10,C=1.0)
svm.fit(X_train,Y_train)


# In[46]:


svm.score(X_train_std,Y_train)


# In[47]:


svm.score(X_test_std,Y_test)

