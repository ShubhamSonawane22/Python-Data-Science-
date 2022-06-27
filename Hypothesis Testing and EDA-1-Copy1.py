#!/usr/bin/env python
# coding: utf-8

# # Hypothesis Testing 

# In[12]:


import pandas as pd
import numpy as nm
from scipy import stats


# In[ ]:


# SUPER MARKET EXAMPLE


# In[5]:


stats.t.cdf(2.23,df=39)  
# To perform t-test -given data available 
# t test = (sample mean-std mean)/(sample std deviation/sqr.root of sample length)


# In[ ]:


probability =0.98
alpha=1-p
i.e 0.016 i.e 0.016<0.05
No need to take any action (Null hypothesis)


# In[ ]:


# CALL CENTER EXAMPLE 
#Two tail hypothesis testing 


# In[ ]:


# from given data :: t value is 1.41


# In[6]:


2*stats.t.cdf(-1.41,df=49)  


#  # One tail test 

# In[15]:


data=pd.Series([0.593,0.142,0.329,0.691,0.231,0.793,0.519,0.292,0.418])
p=scipy.stats.ttest_1samp(data,0.3)[1]


# In[21]:


p_value=p/2


# # Two tail test 

# In[23]:


control=pd.Series([91,87,99,77,88,91])
treat=pd.Series([101,110,103,93,99,104])


# In[24]:


stats.ttest_ind(control,treat)


# # Anova 

# In[25]:


import pandas as pd
import numpy as nm
from scipy import stats


# In[27]:


from sklearn import datasets
importing dataset


# In[33]:


iris=datasets.load_iris()


# In[35]:


df=pd.DataFrame(iris.data)


# In[36]:


df


# In[38]:


stats.f_oneway(df.iloc[:,0],df.iloc[:,1],df.iloc[:,2])


# In[ ]:


#pvalue is less than alphs value (i.e 0.05) then we reject null hypothesis and select alternate hypothesis 


# # Exploratory Data Analysis (EDA-1)

# In[42]:


pip install sweetviz


# In[43]:


import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt 
import numpy as np
import sweetviz as sv


# In[46]:


data1=pd.read_csv("data_clean.csv")


# In[50]:


data1


# In[53]:


data1.dtypes


# In[54]:


data1.info()


# In[58]:


data2=data1.iloc[:,1:]         #slicing unnamed column 


# In[59]:


data2


# In[60]:


data=data2.copy()    #copy method is used so that changes made in new dataframes don't change the original dataframes 


# In[61]:


data


# In[63]:


data['Month'].values    # "May" is categorical value 


# In[64]:


data["Month"]=pd.to_numeric(data["Month"],errors="coerce")    #Converting the "May" into null value   (errors="coerce")


# In[65]:


data["Month"].values


# In[66]:


data["Temp C"]=pd.to_numeric(data["Temp C"],errors="coerce")


# In[69]:


data["Temp C"].values


# In[71]:


data["weather"]=data["Weather"].astype('category')


# In[72]:


data["weather"].values


# In[ ]:


# Duplicate rows elimination 


# In[75]:


data[data.duplicated()]


# In[77]:


data_cleane1=data.drop_duplicates()


# In[78]:


data_cleane1


# In[ ]:


# Drop a column


# In[80]:


data_clean2=data_cleane1.drop("Temp C",axis=1)


# In[82]:


data_clean2


# In[ ]:


# Rename the column 


# In[93]:


data_clean3=data_clean2.rename({"Solar.R": "Solar"},axis=1)


# In[94]:


data_clean3


# In[ ]:


# Detecting outlier 


# In[97]:


data_clean3["Ozone"].hist()


# In[99]:


data_clean3.boxplot(column="Ozone")


# In[ ]:


# Descriptive stats 


# In[100]:


data_clean3["Ozone"].describe()


# In[103]:


data_clean3


# In[109]:


data_clean3["weather"].value_counts().plot.bar()


# In[2]:


import pandas as pd
import seaborn as sns 
import matplotlib as plt 
import numpy as nm


# In[4]:


data=pd.read_csv("data_clean.csv")


# In[5]:


data


# In[6]:


data.info()


# # Mean imputation 

# In[8]:


mean=data['Ozone'].mean()


# In[9]:


mean


# In[10]:


data=data.fillna(mean)


# In[11]:


data


# In[ ]:


# Missing value imputation for categorical values
#Get the object columns 


# In[12]:


obj_column=data['Weather']


# In[13]:


obj_column


# In[14]:


obj_column=obj_column.fillna(obj_column.mode().iloc[0])


# In[15]:


obj_column.mode()


# In[ ]:


#Join the data set with imputed data set 


# In[17]:


data=pd.concat([data,obj_column],axis=1)


# In[18]:


data


# In[17]:


import seaborn as sns
import pandas as pd
data=pd.read_csv("data_clean.csv")


# In[7]:


sns.pairplot(data)


# In[8]:


#correlation 
data.corr()


# In[11]:


# Transformations 
# Dummy variables 
data_clean=pd.get_dummies(data,columns=['Weather'])


# In[12]:


data_clean


# In[42]:


# Spped up the EDA Process 
import sweetviz as sv
data1=pd.read_csv("data_clean.csv")


# In[ ]:




