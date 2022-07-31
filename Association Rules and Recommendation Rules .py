#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as nm
import matplotlib.pyplot as plt
import seaborn as sns 


# In[2]:


data=pd.read_csv("Titanic.csv")


# In[3]:


data.head()


# In[9]:


pip install mlxtend


# In[10]:


import pandas as pd
import numpy as nm
import matplotlib.pyplot as plt
import seaborn as sns 
from mlxtend.frequent_patterns import apriori,association_rules
from mlxtend.preprocessing import TransactionEncoder 


# In[11]:


# Preproceesing _Categorical data into numeric form # 
df=pd.get_dummies(data)


# In[12]:


df.head()


# In[ ]:


# Apriori Algorithm 


# In[16]:


frequent_itemset=apriori(df,min_support=0.1,use_colnames=True)


# In[17]:


frequent_itemset


# In[18]:


#Create rule 


# In[19]:


rules=association_rules(frequent_itemset,metric="lift",min_threshold=0.7)


# In[20]:


rules


# In[21]:


# Selecting lift ration more than 1 


# In[23]:


rules=association_rules(frequent_itemset,metric="lift",min_threshold=1)


# In[24]:


rules


# In[25]:


# For max lift values 


# In[26]:


rules.sort_values("lift",ascending=False)[0:10]


# In[ ]:





# In[ ]:




