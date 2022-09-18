#!/usr/bin/env python
# coding: utf-8

# # Boosting  (XGBM and LGBM) _
# 

# # XGBM (Extream Gradient Boosting)

# In[1]:


get_ipython().system('pip install xgboost')


# In[3]:


from numpy import loadtxt 
from xgboost import XGBClassifier 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score 


# In[8]:


#Data load 
dataset =loadtxt("pima-indians-diabetes.data.csv",delimiter=",")
X=dataset[:,0:8]
Y=dataset[:,8]


# In[9]:


#Split the data into train and test set 
X_train,X_test,Y_train,Y_test=train_test_split (X,Y,test_size=0.30,random_state=6)


# In[11]:


#Fit model on training data 
model=XGBClassifier()
model.fit(X_train,Y_train)


# In[12]:


#Making prediction for test data_
y_pread=model.predict(X_test)


# In[13]:


y_pread


# In[16]:


#Prediction evaluation _
accuracy=accuracy_score(Y_test,y_pread)


# In[18]:


accuracy*100


# # LGBM (Light Gadient Boosting)

# In[20]:


import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 


# In[ ]:


#importing dataset_


# In[24]:


dataset=loadtxt("pima-indians-diabetes.data.csv",delimiter=",")
X=dataset[:,0:8]
Y=dataset[:,8]


# In[25]:


#Split the data into train and test set 
X_train,X_test,Y_train,Y_test=train_test_split (X,Y,test_size=0.33,random_state=0)


# In[29]:


get_ipython().system('pip install lightgbm')


# In[36]:


import lightgbm as lgb 
d_train=lgb.Dataset(X_train,label=Y_train)


# In[33]:


params={}
params['learning_rate']=0.003
params['bossting_type']="gbdt"
params['objective']='binary'
params['metric']='binary_logloss'
params['sub_feature']=0.5
params['num_leaves']=10
params['min_data']=50
params['max_depth']=10


# In[37]:


clf=lgb.train(params,d_train,100)


# In[41]:


#Prediction
y_pred=clf.predict(X_test)


# In[42]:


y_pred

