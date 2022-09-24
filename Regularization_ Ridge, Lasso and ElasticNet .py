#!/usr/bin/env python
# coding: utf-8

# # Ridge Regularization 

# In[ ]:


import pandas as pd
import numpy as nm 
import matplotlib.pyplot as plt 
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 


# In[15]:


#Loading predefined dataset 
boston_dataset=datasets.load_boston()


# In[20]:


#Converting data and loading X and Y variable 
boston_pd=pd.DataFrame(bostan_dataset.data)
boston_pd.columns=boston_dataset.feature_names 
boston_pd_target=nm.asarray(boston_dataset.target)
boston_pd["House PRICE"]=pd.Series(boston_pd_target)


# In[21]:


boston_pd


# In[23]:


# Input 
X=boston_pd.iloc[:, :-1]

#Output
Y=boston_pd.iloc[:, -1]


# In[25]:


X.head()


# In[26]:


Y.head()


# In[27]:


# Apply multiple linear regression model 
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.25)


# In[28]:


x_train.shape,x_test.shape,y_train.shape,y_test.shape


# In[29]:


ireg=LinearRegression()
ireg.fit(x_train,y_train)

#Prediction 
ireg_y_pred=ireg.predict(x_test)


# In[35]:


# Calculating the mean squared error (MSE)
mean_squared_error=nm.mean((ireg_y_pred-y_test)**2)
print("MEAN SQUARED ERROR ON TEST SET :" ,mean_squared_error)


# In[39]:


#Putting together the coefficent and their corresponsding variable name 

ireg_coefficient =pd.DataFrame()
ireg_coefficient["Columns"]=x_train.columns
ireg_coefficient["Coefficient Estimate"]=pd.Series(ireg.coef_)
print(ireg_coefficient)


# In[43]:


#Importing ridge from sklearn library 
from sklearn.linear_model import Ridge 
    
#Train the model

ridgeR=Ridge(alpha=1)
ridgeR.fit(x_train,y_train)
y_pred=ridgeR.predict(x_test)


# In[46]:


#Calculate mean squared error

mean_squared_error_ridge=nm.mean((y_pred-y_test)**2)
print(mean_squared_error_ridge)


# In[50]:


#Getting coefficient 

ridge_coefficient =pd.DataFrame()
ridge_coefficient["columns"]=x_train.columns
ridge_coefficient["coefficient estimate"]=pd.Series(ridgeR.coef_)
print (ridge_coefficient)


# # Lasso Regularization 

# In[55]:


from sklearn.linear_model import Lasso


# In[61]:


#Training the model 

lasso=Lasso(alpha=1)
lasso.fit(x_train,y_train)
y_pred=lasso.predict(x_test)

#Calculating mean squared error

mean_squared_errro=nm.mean((y_pred-y_test)**2)
print ("Mean Squared Error Value",mean_squared_errro)
lasso_coeff=pd.DataFrame()
lasso_coeff["Columns"]=x_train.columns
lasso_coeff['Coeff Estimate']=pd.Series(lasso.coef_)
print (lasso.coef_)


# # ElasticNet Regularization

# In[75]:


from sklearn.linear_model import ElasticNet 

#Train the model 
e_net =ElasticNet(alpha=0.4,l1_ratio=.5)
e_net.fit(x_train,y_train)

#Calculating the prediction and mean squared error 

y_pred_elastic=e_net.predict(x_test)
mean_squared_error=nm.mean((y_pred_elastic-y_test)**2)
print("Mean Squared Error",mean_squared_error)

e_net_coff=pd.DataFrame()
e_net_coff["Columns"]=x_train.columns
e_net_coff["coefficient estimate"]=pd.Series(e_net.coef_)
e_net_coff

