#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd


# In[4]:


#Use as stat calculation 
2+2


# In[1]:


get_ipython().system('pip install pandas')


# In[2]:


import pandas as pd


# In[8]:


get_ipython().system('pip install numpy')


# In[10]:


get_ipython().system('pip install matplotlib')


# In[12]:


get_ipython().system('pip install seaborn')


# In[14]:


get_ipython().system('pip install jupyter_contrib_nbextensions')


# In[17]:


get_ipython().system('jupyter contrib nbextension install --user')


# In[3]:


import pandas as pd


# In[4]:


import numpy as nm


# In[5]:


import seaborn as sns


# In[6]:


import matplotlib as mpl


# In[7]:


# Read the file 
df = pd.read_csv("Salaries.csv")


# In[15]:


df


# In[18]:


df.head(10)


# In[19]:


df.tail()


# In[20]:


df.tail(12)


# In[ ]:


# Determination of data frames data types 


# In[21]:


df.dtypes


# In[23]:


df.columns


# In[24]:


df.dtypes
# Describe the data type


# In[26]:


df.columns
#Describe the column names 


# In[27]:


df.axes
#Listr the rows lable and column names 


# In[28]:


df.ndim
#Number of diamesion data 


# In[29]:


df.size
#Size of data frames 


# In[30]:


df.shape
#Shape of data frame 


# In[15]:


df.values
#Discribe the values in Array formate


# In[17]:


df.mean()


# In[18]:


df.median()


# In[19]:


df.head()


# In[20]:


df.tail()


# In[21]:


df.describe()


# In[22]:


df.std()


# In[23]:


df.dropna()


# In[26]:


df.mode()


# In[27]:


df.min()


# In[28]:


df.max()


# In[29]:


df.head(50).mean()   # Mean of first 50 data#


# In[30]:


x=df.head(50)


# In[32]:


x.mean()


# In[33]:


df.salary   #Selection of column


# In[39]:


df["salary"]


# In[40]:


df["salary"].mean()


# In[41]:


#Data/coloumn in tabular formate
df[["salary"]]


# In[42]:


df[["salary","rank"]]  #Creation of table


# In[45]:


#groupby fucntion
abc=df.groupby(["sex"])


# In[46]:


abc.mean()


# In[51]:


#Group by in conmbination of fucntion 
df.groupby('rank')[['salary']].mean()


# In[53]:


df.groupby('rank')[["salary","phd"]].mean()


# In[54]:


#Relational data set :

abc=df[df['salary']==93000]


# In[55]:


abc


# In[58]:


abc=df[df['salary']<=90000]


# In[60]:


abc


# In[62]:


abd=df[df['sex'] !='male']    #sex is not equal to male 


# In[65]:


abd.shape   #shape of the column


# In[ ]:


# ilock Function ::


# In[10]:


df.iloc[0]  # 1st Row 


# In[11]:


df.iloc[-1]  # Last row


# In[16]:


df.iloc[:,0]                 # all row and 1st column


# In[17]:


df.iloc[0:7]   # selection of first 7 rows


# In[19]:


df.iloc[:,0:2]  #selection of first 2 column


# In[20]:


df.iloc[0:2,1:3]  


# In[ ]:


#Selection o perticular row and column
# Seclection of first and fifth row and 1 and 3rd column


# In[21]:


df.iloc[[1,5],[1,3]]


# In[ ]:


# Use of iloc in combination 

#define a variable
# + selection of salary band and mean of it 


# In[23]:


x=df.iloc[:,[1,3,4]][df.salary>120000].mean


# In[24]:


x


# In[ ]:


# Graphics to explore the data 


# In[ ]:


#Use of matplotlib.pyplot


# In[27]:


import matplotlib.pyplot as plt


# In[28]:


x=[10,2,4]
y=[-1,-3,4]


# In[29]:


plt.plot(x,y)


# In[34]:


fig=plt.figure(figsize=(5,5))
plt.plot(x,y)
plt.xlim(-5,12)
plt.ylim(-5,10)
plt.xlabel("X AXIS")
plt.ylabel("Y AXIS")
plt.title("MAIN GRAPH",size=10)
plt.suptitle("MAJOR CHANGE",size=20)


# In[36]:


mtcars=pd.read_csv("mtcars.csv")


# In[37]:


mtcars.dtypes


# In[41]:


mtcars.columns


# In[43]:


mtcars.axes


# In[44]:


mtcars.ndim


# In[45]:


mtcars.size


# In[47]:


mtcars.shape


# In[51]:


mtcars.head(5)


# In[52]:


# Cross Table (Pandas)


# In[53]:


pd.crosstab(mtcars.gear,mtcars.cyl)


# In[ ]:


# Creation of bar chart with the help of CROSS TAB FUNCTION


# In[61]:


pd.crosstab(mtcars.gear,mtcars.cyl).plot(kind="bar")
plt.ylim(0,15)
plt.xlabel("GEAR VALUE")
plt.ylabel("NO.OF CASES")
plt.title("BARCHART")


# In[67]:


# Creation of pie chart ::
# Need to count the values first 

mtcars["gear"].value_counts


# In[66]:


mtcars.gear.value_counts().plot(kind="pie")


# In[ ]:


# For boxplot(Whisker chart) chart -(Conti. and discrete data types)
Using Seaborn 


# In[68]:


import seaborn as sns


# In[69]:


sns.boxplot(x="gear",y="mpg",data=mtcars)


# In[ ]:


# Creation of pair plot 


# In[76]:


sns.pairplot(mtcars.iloc[:,0:3])


# In[77]:


sns.pairplot(mtcars)


# In[ ]:


# Scatter plot 


# In[79]:


plt.scatter(mtcars.qsec,mtcars.mpg)


# In[ ]:


# Histo Graph


# In[82]:


plt.hist(mtcars.mpg)


# In[87]:


plt.hist((mtcars.mpg),facecolor="black",edgecolor="red",bins =5)


# In[ ]:


#Boxplot


# In[89]:


plt.boxplot((mtcars.mpg),vert=False)


# In[92]:


plt.violinplot(mtcars.mpg)


# In[ ]:


## Visualization with Pandas 


# In[8]:


df=pd.DataFrame(index=["Aditya","Ajay","Anil","Ansh","Anand"],data={"Apple":[20,10,20,11,9],"Orange":[34,23,24,43,10]})


# In[9]:


df


# In[ ]:


#Ploting the Bar plot


# In[108]:


ax=df.plot.bar(color=["0.9","0.3"],figsize=(10,4))


# In[ ]:


# Density Plot 
# For detection of data type for making decision -4th Moment of business decision


# In[15]:


ax=df.plot.kde(color=["0.9","0.3"],figsize=(10,10))


# In[ ]:


# Subplots 
# Multiple Number of plots 


# In[22]:


fig,(ax1,ax2,ax3)=plt.subplots(1,3,figsize=(15,5))
fig.suptitle("TWO VARIABLE PLOTS")
df.plot.line(ax=ax1,title="LINE PLOT")
df.plot.scatter(x="Apple",y="Orange",ax=ax2,title="Scatterplot")
df.plot.bar(ax=ax3,title="BARPLOT")


# In[18]:


import matplotlib.pyplot as plt


# In[26]:


fig,(ax1,ax2,ax3)=plt.subplots(1,3,figsize=(15,5))
fig.suptitle("TWO VARIABLE PLOTS",size=20)
df.plot.box(ax=ax1,title="BOX PLOT")
df.plot.hist(ax=ax2,title="HISTOGRAM")
df.plot.kde(ax=ax3,title="KDE")


# In[27]:


fig,(ax1,ax2,ax3)=plt.subplots(1,3,figsize=(15,5))
fig.suptitle("TWO VARIABLE PLOTS",size=20)
df.plot.box(ax=ax1,title="BOX PLOT")
df.plot.hist(ax=ax2,title="HISTOGRAM")
df.plot.bar(ax=ax3,title="BARPLOT")


# In[ ]:


# Seaborn 
#Advanced data visualization 


# In[ ]:


# Calling internal data set 


# In[1]:


import pandas as pd


# In[2]:


import seaborn as sns 


# In[3]:


import matplotlib.pyplot as plt


# In[4]:


tips=sns.load_dataset("tips")


# In[5]:


tips


# In[ ]:


# Univariate :: DATA VISUALIZATION (One Variable )


# In[ ]:


# Strip Plot 


# In[11]:


sns.stripplot(y="tip",data=tips,jitter=True)


# In[12]:


sns.stripplot(y="tip",data=tips,jitter=False)


# In[ ]:


# Grouping in strip graphs 


# In[13]:


sns.stripplot(x="day",y="tip",data=tips,jitter=False)


# In[ ]:


# Swarm plot (Visualization on the base of 1 descrete data)


# In[18]:


sns.swarmplot(x='day',y='tip',data=tips,size=2,hue='sex')
# size =2 for size of plot 


# In[17]:


# for orientation change 
sns.swarmplot(x='day',y='tip',data=tips,hue='sex',orient="h")


# In[ ]:


# Boxplot and violne plot 


# In[21]:


sns.boxplot(x='day',y="tip",data=tips)


# In[24]:


sns.violinplot(x='day',y="tip",data=tips)


# In[ ]:


# USe of subplot 


# In[26]:


plt.subplot(1,2,1)
sns.violinplot(x='day',y="tip",data=tips)
plt.subplot(1,2,2)
sns.boxplot(x='day',y="tip",data=tips)


# In[ ]:


# Combination of Graphs 


# In[11]:


sns.violinplot(x='day',y='tip',data=tips,inner=None,color="lightgray")
sns.stripplot(x="day",y="tip",data=tips,jitter=True,size=5)


# In[12]:


# Joint Plot 


# In[14]:


sns.jointplot(x="total_bill",y="tip",data=tips)


# In[16]:


sns.jointplot(x="total_bill",y="tip",data=tips,kind="kde")


# In[17]:


#Pair Plot 


# In[23]:


sns.pairplot(tips)


# In[ ]:


# TITANIC DATA VISUALIZATION 


# In[29]:


import numpy as nm
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings 
warnings.filterwarnings("ignore")


# In[32]:


# Loading DATA SET 
data=pd.read_csv('train.csv')
data.head()


# In[38]:


data.dtypes


# In[42]:


data["Survived"].value_counts()


# In[ ]:


# Visual exploration of data


# In[44]:


sns.boxplot(x="Survived",y="Age",data=data)
# Survival by sex 


# In[ ]:


# Very less chances of 60+ age survival-Male and female 
# Most survival from 20 to 40 


# In[48]:


sns.barplot(x="Survived",y="Sex",data=data)


# In[50]:


sns.barplot(y="Pclass",x="Sex",hue="Survived",data=data)


# In[54]:


sns.boxplot(x="Sex",y="Age",hue="Survived",data=data)


# In[61]:


sns.barplot(x="Parch",y="Survived",data=data)


# In[63]:


sns.pairplot(data)


# In[71]:


sns.pairplot(data,hue="Sex",diag_kws={"bw":0.9})


# In[ ]:


# 2 Discrete data and 1 conti data 
# Multiple variable in 2 diamensional graph


# In[75]:


grid=sns.FacetGrid(data,col="Survived",row="Pclass",size=2.5,aspect=1.5)
grid.map(plt.hist,"Age",alpha=0.5,bins=20)
grid.add_legend()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




