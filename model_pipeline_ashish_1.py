#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier 


# In[2]:


df_model=pd.read_csv('credit_pipeline_1.csv',low_memory=False,index_col=0)


# In[3]:


df_model.head()


# In[4]:


df_model.reset_index(inplace=True)


# In[5]:


df_model.drop(['index'],axis=1,inplace=True)


# In[6]:


df_model


# In[7]:


X = df_model.drop(['Status'],axis=1)
y = df_model["Status"]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


# In[22]:


X.info()


# In[23]:


rf = RandomForestClassifier(bootstrap=True, max_depth=10, max_features= 'auto',min_samples_leaf=1,min_samples_split=2,n_estimators=200)


# In[10]:


rf.fit(X_train,y_train)


# In[12]:


pred=rf.predict(X_test)


# In[13]:


from sklearn.metrics import classification_report
class_report_RF=classification_report(y_test, pred)


# In[14]:


print(class_report_RF)


# In[15]:


import joblib
joblib.dump(rf, 'rf_jlib')


# In[16]:


pipe = joblib.load("rf_jlib")


# In[20]:


pipe.predict(X_test)


# In[21]:


X_test.shape


# In[ ]:




