#!/usr/bin/env python
# coding: utf-8

# In[62]:


# Find probability of narmal distribution of random variable having mean=60 and std deviation is 10,
#Find probability that X is less than 70.


# In[63]:


from scipy import stats 
import pandas as pd
import numpy as nm


# In[64]:


stats.norm.cdf(70,loc=60,scale=10)


# In[65]:


# GMAT SCORE having mean=711 and std deviation=29  ,Find probability having score less than 680


# In[66]:


stats.norm.cdf(680,loc=711,scale=29)


# In[67]:


#GMAT SCORE having mean=711 and std deviation=29  ,Find probability having score bet 697 and 740


# In[68]:


stats.norm.cdf(740,loc=711,scale=29)-stats.norm.cdf(697,loc=711,scale=29)


# In[69]:


# Normal Distribution Example and application 
# Stock Market Exapmle 
# BEML and GLAXO DATA 


# In[70]:


import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
from scipy import stats


# In[71]:


beml_df=pd.read_csv("BEML.csv")


# In[72]:


glaxo_df=pd.read_csv("GLAXO.csv")


# In[73]:


beml_df.head()


# In[74]:


glaxo_df.head()


# In[75]:


beml_df=beml_df[["Date","Close"]]


# In[76]:


beml_df


# In[77]:


glaxo_df=glaxo_df[["Date","Close"]]


# In[78]:


glaxo_df


# In[79]:


# Data frames have date column so we can create DateTime Index from this column date
#It will ensure that date time column in asending order 


# In[80]:


beml_df=beml_df.set_index(pd.DatetimeIndex(beml_df["Date"]))
glaxo_df=glaxo_df.set_index(pd.DatetimeIndex(glaxo_df["Date"]))


# In[81]:


beml_df


# In[82]:


glaxo_df


# In[83]:


# Line Graph
plt.plot(beml_df.Close)


# In[84]:


plt.plot(glaxo_df.Close)


# In[85]:


# As the variation in BEML is more and risk factor is high so we need to invest in GLAXO shares


# In[86]:


# To calculate the % gain


# In[87]:


beml_df['gain']=beml_df.Close.pct_change (periods=1)


# In[88]:


beml_df["gain"]


# In[89]:


glaxo_df['gain']=glaxo_df.Close.pct_change (periods=1)


# In[90]:


glaxo_df['gain']


# In[91]:


plt.plot(beml_df.index,beml_df.gain)


# In[92]:


plt.plot(glaxo_df.index,glaxo_df.gain)


# In[93]:


sns.distplot(beml_df.gain)


# In[94]:


sns.distplot(glaxo_df.gain)


# In[95]:


print("Mean:",round(beml_df.gain.mean(),4))
print("Standard Deviation:",round(beml_df.gain.std(),4))


# In[96]:


print("Mean:",round(glaxo_df.gain.mean(),4))
print("Standard Deviation:",round(glaxo_df.gain.std(),4))


# In[97]:


# Probability of making 2% loss or higher in glaxo
# Probability of making 2% Gain or higher in glaxo 


# In[98]:


from scipy import stats 
stats.norm.cdf(-0.02,
                loc=glaxo_df.gain.mean(),
                scale=glaxo_df.gain.std())


# In[99]:


stats.norm.cdf(0.02,
                loc=glaxo_df.gain.mean(),
                scale=glaxo_df.gain.std())


# In[100]:


# Probability of making 2% loss or higher in glaxo
# Probability of making 2% Gain or higher in glaxo 


# In[101]:


from scipy import stats 
stats.norm.cdf(-0.02,
                loc=beml_df.gain.mean(),
                scale=beml_df.gain.std())


# In[102]:


from scipy import stats 
stats.norm.cdf(0.02,
                loc=beml_df.gain.mean(),
                scale=beml_df.gain.std())


# In[103]:


from scipy import stats
import pandas as pd 
import numpy as nm


# In[104]:


stats.norm.ppf(0.95)


# In[105]:


stats.t.ppf(0.975,df=139) # df=Degree of freedom

