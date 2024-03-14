#!/usr/bin/env python
# coding: utf-8

# <H1>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;NLP (Natural Language Processing) PROJECT</H1>
# 
# <H2>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; IMDB SENTIMENT ANALYSIS USING NLP </H2>
# 
# 
# You can download the dataset from the Kaggle link below: https://www.kaggle.com/ymanojkumar023/kumarmanoj-bag-of-words-meets-bags-of-popcorn 
# 
# Please download only: labeledTrainData.tsv file..
# 

# In[ ]:


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


# In[ ]:


# We load our dataset here..
df = pd.read_csv('labeledTrainData.tsv',  delimiter="\t", quoting=3)


# In[ ]:


# Let's look at the head of the dataset:
df.head()


# In[ ]:


len(df)


# In[ ]:


len(df["review"])


# In[ ]:


# Since stopwords aren't used in NLP we have to clear stopwords. For this purpose we must download stopword wordset from nltk library..
# We do this operation using nltk module..
nltk.download('stopwords')



# ## * * * * Data Cleaning Operations * * * *

# ### First, we will delete HTML tags from review sentences using the BeautifulSoup module.
# 
# To explain how these processes are done, first let's choose a single review and see how it is done for you:

# In[ ]:


sample = df.review[0]
sample


# In[ ]:


# Clear HTML tags..
sample = BeautifulSoup(sample).get_text()
sample  # After the HTML tags are cleared.


# In[ ]:


# we clean it from punctuation and numbers - using regex..
sample = re.sub("[^a-zA-Z]",' ',sample)
sample


# In[ ]:


# we convert it to lowercase, since our machine learning algorithms think capitalized letters with different words
# and this may result mistakes..
sample = sample.lower()
sample


# In[ ]:


# stopwords (stopwords are the words like the, is, are not to be used by AI. These are grammar words..)
# First we split the words with split and convert them to a list. Our goal is to remove stopwords..
sample = sample.split()
  


# In[ ]:


sample


# In[ ]:


len(sample)


# In[ ]:


# sample without stopwords
swords = set(stopwords.words("english"))                      # conversion into set for fast searching
sample = [w for w in sample if w not in swords]               
sample


# In[ ]:


len(sample)


# In[ ]:


# After describing the cleanup process, we now batch clean the reviews in our entire dataframe in a loop
# for this purpose we first create a function:


# In[ ]:


def process(review):
    # without HTML tags
    review = BeautifulSoup(review).get_text()
    # without punctuation and numbers
    review = re.sub("[^a-zA-Z]",' ',review)
    # lowercase and splitting to eliminate stopwords
    review = review.lower()
    review = review.split()
    # without stopwords
    swords = set(stopwords.words("english"))                      
    review = [w for w in review if w not in swords]               
    # we join splitted paragraphs with join before return..
    return(" ".join(review))


# In[ ]:


# We clean our training data with the help of the above function:
# We can see the status of the review process by printing a line after every 1000 reviews.

train_x_tum = []
for r in range(len(df["review"])):        
    if (r+1)%1000 == 0:        
        print("No of reviews processed =", r+1)
    train_x_tum.append(process(df["review"][r]))


# ### Train, test split...

# In[ ]:


# Now we are going to split our data as train and test..
x = train_x_tum
y = np.array(df["sentiment"])

# train test split
train_x, test_x, y_train, y_test = train_test_split(x,y, test_size = 0.1)


# ### We are building our Bag of Words !
# 
# We have cleaned our data, but for the artificial intelligence to work, it is necessary to convert this text-based data into numbers and a matrix called bag of words. Here we use the CountVectorizer tool in sklearn for this purpose:

# <IMG src="bag.jpg" width="900" height="900" >

# In[ ]:


# Using the countvectorizer function in sklearn, we create a bag of words with a maximum of 5000 words...
vectorizer = CountVectorizer( max_features = 5000 )

# We convert our train data to feature vector matrix
train_x = vectorizer.fit_transform(train_x)


# In[ ]:


train_x


# In[ ]:


# We are converting it to array because it wants array for fit operation..
train_x = train_x.toarray()
train_y = y_train


# In[ ]:


train_x.shape, train_y.shape


# In[ ]:


train_y


# ### We build and fit a Random Forest Model

# In[ ]:


model = RandomForestClassifier(n_estimators = 100, random_state=42)
model.fit(train_x, train_y)


# In[ ]:





# ### Now it's time for our test data..

# In[ ]:


# We convert our test data to feature vector matrix
test_xx = vectorizer.transform(test_x)


# In[ ]:


test_xx


# In[ ]:


test_xx = test_xx.toarray()


# In[ ]:


test_xx.shape


# #### Now let's predict..

# In[ ]:


test_predict = model.predict(test_xx)
acc = roc_auc_score(y_test, test_predict)


# In[ ]:


print("Accuracy: % ", acc * 100)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




