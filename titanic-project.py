#!/usr/bin/env python
# coding: utf-8

# In[16]:


# Import the neccessary modules
import pandas as pd # manage dataframes
import numpy as np # 
import seaborn as sb


# In[17]:


# Read the dataset into a dataframe
df = pd.read_csv(r'C:\Users\User\OneDrive\Desktop\Data Mining Project\titanic.csv', sep='\t', engine='python')
df.head(10)


# In[18]:


# Drop some columns which is not relevant to the analysis (they are not numeric)
cols_to_drop = ['Name', 'Ticket', 'Cabin']
df = df.drop(cols_to_drop, axis=1)


# In[19]:


df.head(3)


# In[20]:


df.info()
sb.heatmap(df.isnull())


# In[21]:


# To replace missing values with interpolated values, for example Age
df['Age'] = df['Age'].interpolate()


# In[22]:


sb.heatmap(df.isnull())


# In[23]:


# Drop all rows with missin data
df = df.dropna() # drop not avaialable


# In[24]:


df.head()


# In[25]:


# First, create dummy columns from the Embarked and Sex columns
EmbarkedColumnDummy = pd.get_dummies(df['Embarked'])
SexColumnDummy = pd.get_dummies(df['Sex'])


# In[26]:


df = pd.concat((df, EmbarkedColumnDummy, SexColumnDummy), axis=1)


# In[27]:


df.head()


# In[28]:


# Drop the redundant columns thus converted
df = df.drop(['Sex','Embarked'],axis=1)


# In[29]:


# Seperate the dataframe into X and y data
X = df.values
y = df['Survived'].values

# Delete the Survived column from X
X = np.delete(X,1,axis=1)


# In[30]:


# Split the dataset into 70% Training and 30% Test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=0)


# In[31]:


# Using simple Decision Tree classifier
from sklearn import tree
dt_clf = tree.DecisionTreeClassifier(max_depth=5)
dt_clf.fit(X_train, y_train)
dt_clf.score(X_test, y_test)


# In[33]:


# Using Naive Bayes classifier
from sklearn.naive_bayes import GaussianNB
nb_clf = GaussianNB()
nb_clf.fit(X_train, y_train)
nb_clf.score(X_test, y_test)


# In[36]:


# Using KNN classifier
from sklearn.neighbors import KNeighborsClassifier
knn_clf = KNeighborsClassifier(n_neighbors=3)
knn_clf.fit(X_train, y_train)
knn_clf.score(X_test, y_test)


# In[36]:


# Using KNN classifier
from sklearn.neighbors import KNeighborsClassifier
knn_clf = KNeighborsClassifier(n_neighbors=3)
knn_clf.fit(X_train, y_train)
knn_clf.score(X_test, y_test)


# In[36]:


# Using KNN classifier
from sklearn.neighbors import KNeighborsClassifier
knn_clf = KNeighborsClassifier(n_neighbors=3)
knn_clf.fit(X_train, y_train)
knn_clf.score(X_test, y_test)

