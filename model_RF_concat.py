#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# To display all the columns of dataframe
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from itertools import product
import seaborn as sns


# In[3]:


df_preprocessed=pd.read_csv('Bondora_preprocessed.csv')


# In[4]:


df_preprocessed.head()


# In[5]:


df_preprocessed.shape


# In[6]:


df_preprocessed.dtypes


# In[7]:


df_preprocessed.isnull().sum()


# In[8]:


df_preprocessed.columns


# In[9]:


df_preprocessed["VerificationType"] = df_preprocessed["VerificationType"].replace(np.NaN, df_preprocessed["VerificationType"].mean())
print(df_preprocessed["VerificationType"][:10])


# In[10]:


df_preprocessed["Gender"] = df_preprocessed["Gender"].replace(np.NaN, df_preprocessed["Gender"].mean())
print(df_preprocessed["Gender"][:10])


# In[11]:


df_preprocessed["MonthlyPayment"] = df_preprocessed["MonthlyPayment"].replace(np.NaN, df_preprocessed["MonthlyPayment"].mean())
print(df_preprocessed["MonthlyPayment"][:10])


# In[12]:


X = df_preprocessed.copy()
y = X.pop("Status")

# Label encoding for categoricals
for colname in X.select_dtypes("object"):
    X[colname], _ = X[colname].factorize()

# All discrete features should now have integer dtypes
discrete_features = X.dtypes == int


# In[13]:


from sklearn.feature_selection import mutual_info_regression

def make_mi_scores(X, y, discrete_features):
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores

mi_scores = make_mi_scores(X, y, discrete_features)


# In[14]:


mi_scores


# In[14]:


def plot_mi_scores(scores):
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores)
    plt.yticks(width, ticks)
    plt.title("Mutual Information Scores")


plt.figure(dpi=100, figsize=(8, 8))
plot_mi_scores(mi_scores)


# In[16]:


plt.figure(figsize=(20, 20))
# define the mask to set the values in the upper triangle to True
mask = np.triu(np.ones_like(df_preprocessed.corr(), dtype=np.bool))
heatmap = sns.heatmap(df_preprocessed.corr(), mask=mask, vmin=-1, vmax=1, annot=True, cmap='BrBG')
heatmap.set_title('Triangle Correlation Heatmap', fontdict={'fontsize':18}, pad=40);


# In[17]:


df_numerical=df_preprocessed[['Age','AppliedAmount','Interest','LoanDuration','IncomeTotal','LiabilitiesTotal','AmountOfPreviousLoansBeforeLoan']]


# In[24]:


df_numerical


# In[19]:


df_categorical=df_preprocessed[['NewCreditCustomer','VerificationType','LanguageCode','Gender','Education','MaritalStatus','EmploymentStatus','EmploymentDurationCurrentEmployer','OccupationArea','Restructured','CreditScoreEsMicroL','Status']]


# In[20]:


df_categorical.shape


# In[21]:


from sklearn import preprocessing
scalar=preprocessing.StandardScaler()
df_numerical_std_1=scalar.fit_transform(df_numerical)


# In[26]:


df_numerical_std_2=pd.DataFrame(df_numerical_std_1, columns = ['Age','AppliedAmount','Interest','LoanDuration','IncomeTotal','LiabilitiesTotal','AmountOfPreviousLoansBeforeLoan'])


# In[21]:


from sklearn.decomposition import PCA

# Create principal components
pca = PCA(n_components=5)
X_pca = pca.fit_transform(df_numerical_std_1)

# Convert to dataframe
component_names = [f"PC{i+1}" for i in range(X_pca.shape[1])]
X_pca = pd.DataFrame(X_pca, columns=component_names)

X_pca.head()


# In[ ]:


aftr_pca=pd.concat([df_categorical,],axis=1)


# In[29]:


aftr_pca=pd.concat([df_categorical,df_numerical],axis=1)


# In[30]:


aftr_pca


# In[31]:


lab_encod = preprocessing.LabelEncoder()
aftr_pca[['NewCreditCustomer', 'Restructured','LanguageCode', 'Education','MaritalStatus','EmploymentStatus','EmploymentDurationCurrentEmployer','OccupationArea','CreditScoreEsMicroL']]= aftr_pca[['NewCreditCustomer', 'Restructured','LanguageCode', 'Education','MaritalStatus','EmploymentStatus','EmploymentDurationCurrentEmployer','OccupationArea','CreditScoreEsMicroL']].apply(lab_encod.fit_transform)


# In[32]:


aftr_pca.dtypes


# In[26]:


from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
model=ExtraTreesClassifier()
model.fit(X,y)
print(model.feature_importances_)


# In[27]:


feat_import=pd.Series(model.feature_importances_,index=X.columns)
feat_import.nlargest(10).plot(kind='barh')


# In[28]:


plt.figure(figsize=(20, 20))
# define the mask to set the values in the upper triangle to True
mask = np.triu(np.ones_like(aftr_pca.corr(), dtype=np.bool))
heatmap = sns.heatmap(aftr_pca.corr(), mask=mask, vmin=-1, vmax=1, annot=False, cmap='BrBG')
heatmap.set_title('Triangle Correlation Heatmap', fontdict={'fontsize':18}, pad=40);


# In[33]:


aftr_pca.columns


# In[34]:


aftr_pca.isnull().sum()


# In[35]:


aftr_pca.head()


# In[36]:


plt.figure(figsize=(20, 16))
# Store heatmap object in a variable to easily access it when you want to include more features (such as title).
# Set the range of values to be displayed on the colormap from -1 to 1, and set the annotation to True to display the correlation values on the heatmap.
heatmap = sns.heatmap(aftr_pca.corr(), vmin=-1, vmax=1, annot=True)
# Give a title to the heatmap. Pad defines the distance of the title from the top of the heatmap.
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':12}, pad=12);


# In[37]:


aftr_pca.shape


# In[60]:


X = aftr_pca.copy()
y = aftr_pca["Status"]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


# In[61]:


from sklearn.ensemble import RandomForestClassifier  
random_classifier= RandomForestClassifier(bootstrap = True, max_depth = 20, max_features = 4, n_estimators = 200)  
clf = random_classifier.fit(X_train,y_train)
random_pred = clf.predict(X_test)


# In[63]:


aftr_pca.to_csv("credit_pipeline_1.csv")


# In[ ]:




