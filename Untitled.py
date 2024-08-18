#!/usr/bin/env python
# coding: utf-8

# # Building a Naive Bayes model to predict customer churn

# In[1]:


# Install the necessary libraries
get_ipython().system('pip install pandas')
get_ipython().system('pip install matplotlib')
get_ipython().system('pip install seaborn')
get_ipython().system('pip install scikit-learn')


# In[2]:


# import the required packages and libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# In[11]:


#import the dataset
url = 'https://raw.githubusercontent.com/sharmaroshan/Churn-Modelling-Dataset/master/Churn_Modelling.csv'
df_original = pd.read_csv(url)


# In[13]:


df_original.head()


# In[14]:


df_original.info()


# In[15]:


#Drop the feature with no significant impact on target variable (Exited)
churn_df = df_original.drop(['RowNumber', 'CustomerId', 'Surname', 'Gender'], axis=1)


# In[16]:


churn_df.head()


# In[17]:


# Create loyalty feature (Feature Extraction)
churn_df['Loyalty'] = churn_df['Tenure']/churn_df['Age']


# In[18]:


churn_df.head()


# In[20]:


# Transform the Geography feature from categorical into numeric (Feature Transformation)
# Unique value of Geography 
churn_df['Geography'].unique()


# In[21]:


# Dummy encode categorical variable
churn_df = pd.get_dummies(churn_df, drop_first=True)


# In[22]:


churn_df.head()


# In[23]:


# check the class balance
churn_df['Exited'].value_counts()


# In[25]:


# To adhere to the Naive bayes model assumption of independencies in the predictor variable, we drop the Tenure and Age as we have already created a new feature Loyalty from these two
churn_df = churn_df.drop(['Tenure', 'Age'], axis=1)


# In[26]:


churn_df.head()


# In[28]:


# Define the target(Y) variable
y = churn_df['Exited']

# Define the predictor(X) variable
x = churn_df.copy()
x = x.drop('Exited', axis=1)

# Split the data 
# Notice that we include the argument stratify equals Y. If our master data has a class split of 80/20, stratifying ensures that this proportion is maintained in both the training and test data. Equals Y tells the function that it should use the class ratio found in the Y variable, which is our target. The less data you have overall and the greater your class imbalance, the more important it is to stratify when you split the data.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, stratify=y, random_state = 42)


# In[30]:


# Fit the model
gnb = GaussianNB()
gnb.fit(x_train, y_train)

# Get the predictions on test data
y_preds = gnb.predict(x_test)


# In[31]:


# Evaluate the model based on the metrics
print("Accuracy: ", accuracy_score(y_test, y_preds))
print("Recall: ", recall_score(y_test, y_preds))
print("F1 score: ", f1_score(y_test, y_preds))
print("Precision: ", precision_score(y_test, y_preds))


# In[33]:


# Check unique value of predictions
np.unique(y_preds)


# Okay, the model predicted zero or not churned for every sample in the test data. 

# In[34]:


# Describe the input data
x.describe()


# Something that stands out is that the loyalty variable we created is on a vastly different scale than some of the other variables we have such as balance or estimated salary. The maximum value of loyalty is 0.56, while the maximum value for balance is over 250,000. Almost six orders of magnitude greater. One thing that you can try when modeling is scaling your predictor variables. Some models require you to scale the data in order for them to operate as expected, while others don't. Naive Bayes does not require data scaling. However, sometimes packages and libraries need to make assumptions and approximations in their calculations. We're already breaking some of these assumptions by using the GaussianNB classifier on this data set. And it may not be helping that some of our predictor variables are on very different scales. In general, scaling might not improve the model, but it probably won't make it worse.

# In[35]:


# Import the scaler the function
from sklearn.preprocessing import MinMaxScaler

# Instantiate the scaler
scaler = MinMaxScaler()

# Fit the scaler to the training data
scaler.fit(x_train)

# Scale the training data
x_train = scaler.transform(x_train)

# Scale the test data
x_test = scaler.transform(x_test)


# In[36]:


# Fit the model again, this time to the scaled data
gnb_scaled = GaussianNB()
gnb_scaled.fit(x_train, y_train)

# Get the predictions on test data
scaled_preds = gnb_scaled.predict(x_test)


# In[37]:


def conf_matrix_plot(model, x_data, y_data):
    model_pred = model.predict(x_data)
    cm = confusion_matrix(y_data, model_pred, labels=model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot()
    plt.show()


# In[38]:


conf_matrix_plot(gnb_scaled, x_test, y_test)


# In[40]:


# Evaluate the model again based on the metrics
print("Accuracy: ", accuracy_score(y_test, scaled_preds))
print("Recall: ", recall_score(y_test, scaled_preds))
print("F1 score: ", f1_score(y_test, scaled_preds))
print("Precision: ", precision_score(y_test, scaled_preds))


# This looks certainly better than the previous one!

# In[ ]:




