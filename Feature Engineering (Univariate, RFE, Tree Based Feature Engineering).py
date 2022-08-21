#!/usr/bin/env python
# coding: utf-8

# # Univariate Feature Selection#

# In[3]:


import pandas as pd
import numpy as nm 
from numpy import set_printoptions
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2 
from pandas import read_csv


# In[8]:


#Loading data set
filename="pima-indians-diabetes.data.csv"
name=["preg","plas","pres","skin","test","mass","pedi","age","class"]
dataset=read_csv(filename,names=name)
array=dataset.values
x=array[:,0:8]
y=array[:,8]


# In[9]:


#Feature Extraction 
test=SelectKBest(score_func=chi2,k=4)
fit=test.fit(x,y)


# In[14]:


fit.scores_


# In[ ]:


# Max Chi2 value suggest variable useful in predicting the class variable 
#(Useful variable "test","plas","class","preg")


# In[ ]:


# FOR REGRESSSION _
f_regression,mutual_info_regression # When Y variable is numeric

#FOR CLASSIFICATION _
chi2,f_classif,mutual_info_classif # When Y variable is categorical 


# # RFE (Recursive Feature Elimination)

# In[20]:


from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression 
#Loading data set
filename="pima-indians-diabetes.data.csv"
name=["preg","plas","pres","skin","test","mass","pedi","age","class"]
data=read_csv(filename,names=name)
array=data.values
x=array[:,0:8]
y=array[:,8]


# In[22]:


#Feature Extraction 
model=LogisticRegression(max_iter=500)
rfe=RFE(model,3)
fit=rfe.fit(x,y)


# In[23]:


rfe.support_
#True means supporting the model building prediction 
#Flase means not supporting the moldel building preduction 


# In[24]:


rfe.ranking_
# Ranking of varaible impacting model building prediction 


# # Feature importance using Decision Tree

# In[25]:


from sklearn.tree import DecisionTreeClassifier 


# In[26]:


#Loading data set
filename="pima-indians-diabetes.data.csv"
name=["preg","plas","pres","skin","test","mass","pedi","age","class"]
data=read_csv(filename,names=name)
array=data.values
x=array[:,0:8]
y=array[:,8]


# In[29]:


model=DecisionTreeClassifier()
model.fit(x,y)


# In[30]:


model.feature_importances_
#Max values having vabriable impacting model building preduction 


# In[ ]:





# In[ ]:





# In[ ]:




