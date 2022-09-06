#!/usr/bin/env python
# coding: utf-8

# # GRIP: The Sparks Foundation

# # Data Science & Business Analytics Internship 

#    ## Task 1: Prediction Using Supervised Machine Learning : *Students' Score Prediction*
#    

# ## Author: Sayan Das

# ### importing libraries

# In[104]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ### importing the data set

# In[105]:


score_df=pd.read_excel("Downloads\score.xlsx")


# In[106]:


score_df


# ### Data Wrangling

# In[107]:


score_df.info()


# In[108]:


score_df.describe().T


# In[109]:


sns.boxplot(score_df['Scores'])


# In[110]:


sns.boxplot(score_df['Hours'])


# In[111]:


sns.relplot(data=score_df, x="Hours", y="Scores", color='green')


# In[112]:


score_df.corr()


# ### Data Analysis using Linear Regression

# In[113]:


#train-test split


# In[114]:


from sklearn.model_selection import train_test_split


# In[115]:


x=score_df[['Hours']].values
y=score_df[['Scores']].values


# In[116]:


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3, random_state=0)


# In[117]:


#importing linearmodel


# In[118]:


from sklearn import linear_model
reg= linear_model.LinearRegression()


# In[119]:


#predictive analysis


# In[120]:


reg.fit(x_train,y_train)
print("intercept:", reg.intercept_)
print("Coefficient:", reg.coef_)


# In[121]:


sns.regplot(data=score_df, x="Hours", y="Scores", color='green')


# In[122]:


yhat= reg.predict(x_test)


# In[123]:


from sklearn import metrics  
from sklearn.metrics import r2_score
print('Mean Absolute Error:', 
      metrics.mean_absolute_error(y_test, yhat))
print('R-Squared:', r2_score(y_test, yhat))


# In[124]:


def score_predict():
    X=float(input("Enter The Study Hours"))
    Y=reg.intercept_+(reg.coef_*X)
    print("If a student scores",X, "hours a day, he can achieve", float(Y), "%")


# In[125]:


score_predict()


# # *----------------------------------------------*
