#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd 
import seaborn as sns 


# In[5]:


data=pd.read_csv("NewspaperData.csv")


# In[7]:


data.head()


# In[8]:


data.info()


# In[9]:


data.corr()


# In[ ]:


# Fitting a Linear Regrassion Model 


# In[10]:


import statsmodels.formula.api as smf
model= smf.ols("sunday~daily",data=data).fit()


# In[11]:


sns.regplot(x="daily",y="sunday",data=data);


# In[12]:


model.summary()


# In[ ]:


# Prediction for new data points (Eg. 200 and 300)


# In[13]:


newdata=pd.Series([200,300])


# In[15]:


data_pred=pd.DataFrame(newdata,columns=['daily'])


# In[16]:


data_pred


# In[17]:


model.predict(data_pred)


# In[ ]:


# New Example_WC_AT DATA 


# In[18]:


data=pd.read_csv("WC_AT.csv")


# In[19]:


data.head()


# In[20]:


data.info()


# In[21]:


data.corr()


# In[22]:


import statsmodels.formula.api as smf
model= smf.ols("Waist~AT",data=data).fit()


# In[44]:


sns.regplot(x="AT",y="Waist",data=data);


# In[45]:


model.summary()


# In[ ]:


# Prediction for new data points (Eg. waist value 90,100,110 )


# In[46]:


newdata=pd.Series([90,100])


# In[50]:


data_pred=pd.DataFrame(newdata,columns=['AT'])


# In[51]:


data_pred


# In[52]:


model.predict(data_pred)


# In[ ]:


# Salary Example


# In[53]:


data=pd.read_csv('Salary_Data.csv')


# In[54]:


data.head()


# In[55]:


data.info()


# In[56]:


data.corr()


# In[57]:


import statsmodels.formula.api as smf
model= smf.ols("YearsExperience~Salary",data=data).fit()


# In[59]:


sns.regplot(x="YearsExperience",y="Salary",data=data);


# In[60]:


model.summary()


# In[1]:


# Multiple Regresion 


# In[10]:


import pandas as pd
import seaborn as sns 
import numpy as nm 
import matplotlib.pyplot as plt
from statsmodels.graphics.regressionplots import influence_plot 


# In[61]:


data=pd.read_csv("Cars.csv")


# In[4]:


data.head()


# In[7]:


data.info()


# In[8]:


data.corr()


# In[12]:


sns.pairplot(data)    # Pairplot for bet variables 


# In[32]:


import statsmodels.formula.api as smf
model= smf.ols("MPG~HP+VOL+SP+WT",data=data).fit()


# In[33]:


model


# In[15]:


model.summary()


# In[16]:


import statsmodels.formula.api as smf
model_v= smf.ols("MPG~VOL",data=data).fit()
model_v.summary()


# In[17]:


import statsmodels.formula.api as smf
model_w= smf.ols("MPG~WT",data=data).fit()
model_w.summary()


# In[18]:


import statsmodels.formula.api as smf
model_vw= smf.ols("MPG~VOL+WT",data=data).fit()
model_vw.summary()


# In[ ]:


# Calculating the VIF value (Varience inflation factor)


# In[25]:


model_m= smf.ols("MPG~VOL+WT+SP",data=data).fit().rsquared
vif_mpg=1/(1-model_m)
model_v= smf.ols("VOL~MPG+WT+SP",data=data).fit().rsquared
vif_vol=1/(1-model_v)
model_wt= smf.ols("WT~VOL+MPG+SP",data=data).fit().rsquared
vif_wt=1/(1-model_wt)
model_sp= smf.ols("SP~VOL+WT+MPG",data=data).fit().rsquared
vif_sp=1/(1-model_sp)


# In[26]:


d1={"variables" : ["HP","Weight","Vol","Spd"],"VIF":[vif_mpg,vif_vol,vif_wt,vif_sp]}


# In[29]:


d1frame=pd.DataFrame(d1)


# In[30]:


d1frame


# In[ ]:


# Test of Normality for residual 


# In[34]:


res=model.resid


# In[35]:


res


# In[45]:


import statsmodels.api as sm 
import numpy as nm
qqplot=sm.qqplot(res,line="q")
plt.title("Test for Normality for Residual -QQ Plot")


# In[47]:


list(nm.where(model.resid>10))


# In[ ]:


# Model Deletion Diagonstics


# In[49]:


model_influence=model.get_influence()
(c,_)=model_influence.cooks_distance


# In[50]:


c


# In[67]:


import numpy as np
fig=plt.subplots(figsize=(20,7))
plt.stem(np.arange(len(data)),np.round(c,3))
plt.show()


# In[ ]:


# To get index and value of influencer 


# In[72]:


np.argmax(c),np.max(c)


# In[ ]:


# Droping the data points 


# In[74]:


data_new=data.drop(data.index[[76,70]],axis=0).reset_index()


# In[75]:


data_new


# In[ ]:


# Creating a new model 


# In[78]:


final_model= smf.ols("MPG~HP+VOL+SP",data=data_new).fit()


# In[79]:


final_model.summary() # R-squated value improved 


# In[ ]:


# Again use of Cooks distance 


# In[88]:


model_influence=model.get_influence()
(c_v,_)=model_influence.cooks_distance


# In[89]:


c

