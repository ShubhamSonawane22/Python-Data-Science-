#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn import datasets
dir(datasets)


# In[2]:


import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as nm
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree 
from sklearn.metrics import classification_report
from sklearn import preprocessing 


# In[3]:


#Decision Tree C5.0#
#IRIS DATA SET#


# In[4]:


import pandas as pd
import numpy as nm
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn import preprocessing 
from sklearn.model_selection import train_test_split 
from sklearn import tree 
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report 


# In[5]:


iris=pd.read_csv("IRIS.csv",index_col=0)


# In[6]:


iris.head()


# In[7]:


# Converting data into numerical format:
#Species 


# In[8]:


label_encoder=preprocessing.LabelEncoder()
iris["Species"]=label_encoder.fit_transform(iris["Species"])


# In[9]:


#Define X and Y 


# In[10]:


x=iris.iloc[:,0:3]
y=iris["Species"]


# In[11]:


x


# In[12]:


y


# In[13]:


iris["Species"].unique()


# In[14]:


iris.Species.value_counts()


# In[15]:


#Splitting the data set into TEST And TRAIN data set#


# In[16]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=30)


# In[17]:


#Building decision tree classifier using Entropy criteria#


# In[18]:


model=DecisionTreeClassifier(criterion='entropy',max_depth=3)
model.fit(x_train,y_train)


# In[19]:


#Plot the DecisionTree


# In[20]:


tree.plot_tree(model);


# In[21]:


fn=("Sapal.Length","Spal.Width","Petal.Length","Petal.Width")
cn=("setosa","versocolor","vrginica")
fig,axes=plt.subplots(nrows=1,ncols=1,figsize=(4,4),dpi=100)
tree.plot_tree(model,feature_names=fn,class_names=cn,filled=True);


# In[22]:


# Predicting the test data#


# In[23]:


preds=model.predict(x_test)
pd.Series(preds).value_counts();


# In[24]:


preds


# In[25]:


pd.crosstab(y_test,preds)


# In[26]:


nm.mean(preds==y_test)*100


# In[27]:


#CART DECISION TREE#


# In[28]:


from sklearn.tree import DecisionTreeClassifier


# In[30]:


model_gini=DecisionTreeClassifier(criterion="gini",max_depth=3)


# In[32]:


model_gini.fit(x_train,y_train)


# In[33]:


tree.plot_tree(model_gini);


# In[34]:


preds_=model_gini.predict(x_test)


# In[35]:


preds_


# In[37]:


nm.mean(preds_==y_test)*100


# In[ ]:


# Decision Tree Regression#


# In[39]:


from sklearn.tree import DecisionTreeRegressor


# In[40]:


array=iris.values


# In[41]:


x=array[:,0:3]
y=array[:,3]


# In[47]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=2)


# In[49]:


model=DecisionTreeRegressor()


# In[50]:


model.fit(x_train,y_train)


# In[56]:


preds=model.predict(x_test);


# In[57]:


preds


# In[64]:


model.score(x_test,y_test)*100

