#!/usr/bin/env python
# coding: utf-8

# # Building a Movie Recommendation System 
# 
# 
# 
# In this project we will use well known Movielens 100k dataset.
# You can download the dataset from Kaggle from the link below:
# https://www.kaggle.com/imkushwaha/movielens-100k-dataset
# 
# You must download only 2 files from this repository:
# 
# u.item
# u.data
# u.item : Contains information about movies (movie id and name) u.data : Contains information about user reviews..
# 
# 

# 
# <IMG src="s.jpg" width="250" height="350" >

# In[ ]:


import pandas as pd


# In[ ]:


column_names = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv('u.data', sep='\t', names=column_names) #You can download the dataset(u.data and u.item) from Kaggle from the link: https://www.kaggle.com/imkushwaha/movielens-100k-dataset


# In[ ]:


df.head()


# In[ ]:


# Let's see how many records we have:
len(df)


# ### Now import u.item file:

# In[ ]:



m_cols = ['item_id', 'title']
movie_titles = pd.read_csv('u.item', sep='|', names=m_cols, usecols=range(2))
movie_titles.head()


# In[ ]:


# Let's see how many records we have:

len(movie_titles)


# In[ ]:


# Now lets merge u.data and u.item files based on item_id
df = pd.merge(df, movie_titles, on='item_id')
df.head()


# ### We Are Setting Up Our Recommendation System:
# 

# In[ ]:


# First, we set up a pivot table-like structure in Excel.
# According to this structure, each row will be a user (ie the index of our dataframe will be user_id)
# There will be movie names in the columns,
# We create a dataframe with rating values in the table!

moviepivot = df.pivot_table(index='user_id',columns='title',values='rating')
moviepivot.head()


# In[ ]:


type(moviepivot)


# ### Purpose: Making movie suggestions similar to Starwars movie

# Let's take a look at the user ratings of Star Wars (1977):

# In[ ]:


starwars_user_ratings = moviepivot['Star Wars (1977)']
starwars_user_ratings.head()


# Let's calculate the correlations with the Star wars movie using the corrwith() method:

# In[ ]:


similar_to_starwars = moviepivot.corrwith(starwars_user_ratings)


# In[ ]:


similar_to_starwars


# In[ ]:


type(similar_to_starwars)


# #### It throws a warning because some records have spaces, let's convert it to a dataframe named corr_starwars and clear the NaN records and see:

# In[ ]:


corr_starwars = pd.DataFrame(similar_to_starwars, columns=['Correlation'])
corr_starwars.dropna(inplace=True)
corr_starwars.head()


# ### Let's list the dataframe we obtained and see what is the closest movie it would recommend:

# In[ ]:


corr_starwars.sort_values('Correlation',ascending=False).head(10)


# #### As you can see, there are irrelevant results. When you do a little research on this subject, you will find that the reason for this is because these films received very few votes. To correct this situation, let's eliminate the films that received less than 100 votes. Let's keep the votes (ie the number of votes)...

# In[ ]:


df.head()


# We don't need timestamp column, so drop it..

# In[ ]:


df.drop(['timestamp'], axis = 1)


# In[ ]:


# Let's find the mean value rating of each movie
ratings = pd.DataFrame(df.groupby('title')['rating'].mean())

# Let's sort them from high to low...
ratings.sort_values('rating',ascending=False).head()


# #### Attention: While calculating these averages, we did not look at how many votes it received, so there were movies like this that we did not know at all..

# In[ ]:


# Now let's find the number of votes each movie received.
ratings['rating_count'] = pd.DataFrame(df.groupby('title')['rating'].count())
ratings.head()


# In[ ]:


# Now let's sort the movies with the most votes, from largest to smallest...
ratings.sort_values('rating_count',ascending=False).head()


# In[ ]:


# Let's go back to our main goal and add the rating_count column to our corr_starwars dataframe (with join)


# In[ ]:


corr_starwars.sort_values('Correlation',ascending=False).head(10)


# In[ ]:


corr_starwars = corr_starwars.join(ratings['rating_count'])
corr_starwars.head()


# ### And the result:

# In[ ]:


corr_starwars[corr_starwars['rating_count']>100].sort_values('Correlation',ascending=False).head()


# As a result we have a reasonable movie recommendations for Star Wars movie.. Similarly you can try and see what our system will recommend you for other movies..

# In[ ]:





# In[ ]:





# In[ ]:




