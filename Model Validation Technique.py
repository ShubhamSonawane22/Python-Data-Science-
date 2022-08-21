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

# For regression _import DecisionTreeClassifier
# For classification_import DecisionTreeRegressor


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


# # Model Validation Technique

# # Train Test Split 

# In[35]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
#Loading data set
filename="pima-indians-diabetes.data.csv"
name=["preg","plas","pres","skin","test","mass","pedi","age","class"]
data=read_csv(filename,names=name)
array=data.values
X=array[:,0:8]
Y=array[:,8]


# In[36]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.30,random_state=7)


# In[37]:


model=LogisticRegression()
model.fit(X_train,Y_train)


# In[39]:


model.score(X_test,Y_test)*100


# # K Fold Cross Validation 

# In[53]:


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression


# In[54]:


#Loading data set
filename="pima-indians-diabetes.data.csv"
name=["preg","plas","pres","skin","test","mass","pedi","age","class"]
data=read_csv(filename,names=name)
array=data.values
X=array[:,0:8]
Y=array[:,8]


# In[58]:


num_folds =10
seed=7
kfold=KFold(n_splits=num_folds)
model=LogisticRegression(max_iter=400)
results=cross_val_score(model,X,Y,cv=kfold)


# In[59]:


results


# In[61]:


results.mean()*100


# In[63]:


results.std()*100


# # Leave one out Cross Validation

# In[66]:


from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

#Loading data set
filename="pima-indians-diabetes.data.csv"
name=["preg","plas","pres","skin","test","mass","pedi","age","class"]
data=read_csv(filename,names=name)
array=data.values
X=array[:,0:8]
Y=array[:,8]

lou=LeaveOneOut()
model=LogisticRegression(max_iter=400)
results=cross_val_score(model,X,Y,cv=lou)


# In[67]:


results


# In[68]:


results.mean()*100


# In[69]:


results.std()*100

