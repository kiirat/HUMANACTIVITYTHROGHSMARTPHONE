#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
import warnings
warnings.filterwarnings('ignore')


# In[2]:


train=pd.read_csv("train.csv")
test=pd.read_csv("test.csv")


# In[3]:


train.head(10)


# In[48]:


test.head(10)


# In[49]:


train.info()


# In[50]:


test.info()


# In[51]:


train.isnull().sum()


# In[52]:


test.isnull().sum()


# In[53]:


train.describe()


# In[54]:


test.describe()


# In[55]:


train.shape


# In[56]:


test.shape


# In[57]:


#checking if there is imbalance in data 
plt.figure(figsize=(12,8))
sns.countplot(train["Activity"])
plt.title("Dataset")
plt.show()


# In[58]:


plt.figure(figsize=(10,7))
sns.boxplot(x='Activity', y='tBodyAccMag-mean()',data=train, showfliers=False)
plt.ylabel('Body Acceleration Magnitude mean')
plt.title("Boxplot of tBodyAccMag-mean() column across various activities")
plt.axhline(y=-0.7, xmin=0.05,dashes=(3,3))
plt.axhline(y=0.020, xmin=0.35, dashes=(3,3))
plt.xticks(rotation=90)


# In[59]:


plt.figure(figsize=(10,7))
sns.boxplot(x='Activity', y='angle(X,gravityMean)', data=train, showfliers=False)
plt.axhline(y=0.08, xmin=0.1, xmax=0.9,dashes=(3,3))
plt.ylabel("Angle between X-axis and gravityMean")
plt.title('Box plot of angle(X,gravityMean) column across various activities')
plt.xticks(rotation = 90)


# In[60]:


plt.figure(figsize=(10,7))
sns.boxplot(x='Activity', y='angle(Y,gravityMean)', data = train, showfliers=False)
plt.ylabel("Angle between Y-axis and gravityMean")
plt.title('Box plot of angle(Y,gravityMean) column across various activities')
plt.xticks(rotation = 90)
plt.axhline(y=-0.35, xmin=0.01, dashes=(3,3))


# In[61]:


plt.figure(figsize=(10,7))
sns.boxplot(x='Activity', y='angle(Y,gravityMean)', data = train, showfliers=False)
plt.ylabel("Angle between Y-axis and gravityMean")
plt.title('Box plot of angle(Y,gravityMean) column across various activities')
plt.xticks(rotation = 90)
plt.axhline(y=-0.35, xmin=0.01, dashes=(3,3))


# In[62]:


train.nunique()


# In[63]:


test.nunique()


# In[64]:


data=train.iloc[:,:-1].values
type(data)


# In[65]:


cov=np.cov(data,rowvar=False)


# In[66]:


cov.shape


# In[67]:


train["Activity"].unique()


# In[68]:


train.groupby("Activity").count()


# In[69]:


from sklearn.manifold import TSNE


# In[70]:


X_for_tsne = train.drop(['subject', 'Activity'], axis=1)


# In[71]:


get_ipython().run_line_magic('time', '')
tsne = TSNE(random_state = 42, n_components=2, verbose=1, perplexity=50, n_iter=1000).fit_transform(X_for_tsne)


# In[72]:


plt.figure(figsize=(12,8))
sns.scatterplot(x =tsne[:, 0], y = tsne[:, 1], hue = train["Activity"],palette="bright")


# In[73]:


y_train = train.Activity
X_train = train.drop(['subject', 'Activity'], axis=1)
y_test = test.Activity
X_test = test.drop(['subject', 'Activity'], axis=1)
print(X_train.shape)
print(X_test.shape)


# In[74]:


y_train


# In[75]:


#lets try regression model with cross validation and hyperparameter tuning


# In[76]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score,classification_report


# In[77]:


parameters = {'C':np.arange(10,30,10), 'penalty':['l2','l1']}
lr_classifier = LogisticRegression()
lr_classifier_rs = RandomizedSearchCV(lr_classifier,param_distributions=parameters,cv=5,random_state=42)
lr_classifier_rs.fit(X_train, y_train)
y_pred = lr_classifier_rs.predict(X_test)


# In[78]:


y_pred


# In[79]:


lr_accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
print("Accuracy using Logistic Regression : ", lr_accuracy)


# In[80]:


# function to plot confusion matrix
def plot_confusion_matrix(cm,lables):
    fig, ax = plt.subplots(figsize=(12,8)) # for plotting confusion matrix as image
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
    yticks=np.arange(cm.shape[0]),
    xticklabels=lables, yticklabels=lables,
    ylabel='True label',
    xlabel='Predicted label')
    plt.xticks(rotation = 90)
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]),ha="center", va="center",color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()


# In[81]:


cm = confusion_matrix(y_test.values,y_pred)
plot_confusion_matrix(cm, np.unique(y_pred))


# In[82]:


#function to get best random search attributes
def get_best_randomsearch_results(model):
    print("Best estimator : ", model.best_estimator_)
    print("Best set of parameters : ", model.best_params_)
    print("Best score : ", model.best_score_)


# In[83]:


get_best_randomsearch_results(lr_classifier_rs)


# In[84]:


get_ipython().system('pip install hmmlearn')


# In[85]:


from hmmlearn import hmm
model = hmm.GaussianHMM(n_components=2, covariance_type="full", n_iter=100)


# In[86]:


model.fit(X_train)


# In[ ]:




