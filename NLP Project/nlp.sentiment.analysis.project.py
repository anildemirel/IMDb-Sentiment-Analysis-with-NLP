#!/usr/bin/env python
# coding: utf-8

# # IMD Sentiment Analysis with NLP

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from bs4 import BeautifulSoup
import re
import nltk
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords


# In[7]:


#load our data set
df = pd.read_csv('NLPlabeledData.tsv',  delimiter="\t", quoting=3)


# In[69]:


df.head()


# In[9]:


len(df)


# In[10]:


len(df["review"])


# In[11]:


#To clear stopwords, we need to download the stopwords word set from the nltk library to our computer.
# We do this with nltk

nltk.download('stopwords')


# ## Data Cleaning Operations 

# ## First, we will delete HTML tags from review sentences using the BeautifulSoup module.
# To explain how these processes are done, let's first select a single review and see how it is done for you:

# In[13]:


sample_review= df.review[0]
sample_review


# In[14]:


# After cleaning the HTML tags..
sample_review = BeautifulSoup(sample_review).get_text()
sample_review


# In[15]:


# we clean it from punctuation and numbers - using regex..
sample_review = re.sub("[^a-zA-Z]",' ',sample_review)
sample_review


# In[16]:


# convert to lowercase, We do this so that machine learning algorithms do not perceive words that 
# start with a capital letter as different words
sample_review = sample_review.lower()
sample_review


# In[17]:


# stopwords
# First, we split the words with split and convert them to a list. our goal is to remove stopwords
sample_review = sample_review.split()


# In[18]:


sample_review


# In[19]:


len(sample_review)


# In[20]:


# sample_review without stopwords
swords = set(stopwords.words("english"))                      # conversion into set for fast searching
sample_review = [w for w in sample_review if w not in swords]               
sample_review


# In[21]:


len(sample_review)


# In[22]:


# After describing the cleanup process, we now batch clean the reviews in our entire dataframe in a loop
# for this purpose we first create a function:


# In[23]:


def process(review):
    # review without HTML tags
    review = BeautifulSoup(review).get_text()
    # review without punctuation and numbers
    review = re.sub("[^a-zA-Z]",' ',review)
    # converting into lowercase and splitting to eliminate stopwords
    review = review.lower()
    review = review.split()
    # review without stopwords
    swords = set(stopwords.words("english"))                      # conversion into set for fast searching
    review = [w for w in review if w not in swords]               
    # splitted paragraph'ları space ile birleştiriyoruz return
    return(" ".join(review))


# In[24]:


# We clean our training data with the help of the above function:
# We can see the status of the review process by printing a line after every 1000 reviews.
train_x_tum = []
for r in range(len(df["review"])):        
    if (r+1)%1000 == 0:        
        print("No of reviews processed =", r+1)
    train_x_tum.append(process(df["review"][r]))


# ## Train, test split

# In[27]:


x = train_x_tum
y = np.array(df["sentiment"])

# train test split
train_x, test_x, y_train, y_test = train_test_split(x,y, test_size = 0.1)


# ## Create Bag of Words

# In[28]:


# Using the count vectorizer function in sklearn, we create a bag of words with a maximum of 5000 words
vectorizer = CountVectorizer( max_features = 5000 )

# we convert our train data to feature vector matrix
train_x = vectorizer.fit_transform(train_x)


# In[29]:


train_x


# In[30]:


# We're converting it to an array because it requires an array for the fit operation.
train_x = train_x.toarray()
train_y = y_train


# In[31]:


train_x.shape, train_y.shape


# In[32]:


train_y


# ## We create Random Forest Model and fit

# In[33]:


model = RandomForestClassifier(n_estimators = 100, random_state=42)
model.fit(train_x, train_y)


# ## Now it's time for our test data..

# In[34]:


# We convert our test data to feature vector matrix
# So we repeat the same operations (conversion to bag of words) this time for our test data:
test_xx = vectorizer.transform(test_x)


# In[35]:


test_xx


# In[36]:


test_xx = test_xx.toarray()


# In[37]:


test_xx.shape


# ## Prediction

# In[38]:


test_predict = model.predict(test_xx)
accuracy = roc_auc_score(y_test, test_predict)


# In[39]:


print("Accuracy : % ", accuracy * 100)


# In[40]:


print(test_xx[0])


# In[41]:


print(test_predict[0])


# In[45]:


print(test_xx[4])


# In[46]:


print(test_predict[4])


# In[47]:


print(test_xx[2])


# In[48]:


print(test_predict[2])


# In[54]:


print(test_xx[4])


# In[55]:


print(test_predict[4])


# In[56]:


print(test_xx[3])


# In[57]:


print(test_predict[3])


# In[87]:


print(test_xx[20])


# In[88]:


print(test_predict[20])


# In[81]:


print(test_xx[17])


# In[82]:


print(test_predict[17])


# In[77]:


print(test_xx[12])


# In[78]:


print(test_predict[12])


# In[64]:


print(test_xx[7])


# In[65]:


print(test_predict[7])


# In[70]:


df.head(24)


# In[ ]:




