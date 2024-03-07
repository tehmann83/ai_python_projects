#!/usr/bin/env python
# coding: utf-8

# # Calculating Employee Salaries using Polynomial Linear Regression
# 
# <IMG src="sal.jpg" width="300" height="300">
# 
# In this project we are going to build a machine learning model for exact calculation of employee salaries.
# 
# Polynomial Linear Regression General Formula:
# 
# y = a + b1*x + b2*x^2 + b3*x^3 + b4*x^4 + ....... + bN*x^N
# 

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# We import our dataset using pandas into df dataframe..
df = pd.read_csv("salaries_dataset.csv",sep = ";")


# In[2]:


df


# In[3]:



# Veri setimize bir bakalım
plt.scatter(df['experience_level'],df['salary'])
plt.xlabel('Experience level')
plt.ylabel('Salary')
plt.savefig('1.png', dpi=300)  # You can save the figure if you want!
plt.show()


# In[6]:


# As you can see data is not distributed linearly..
# If we apply linear regression to the dataset we see an incorrect model graph, let's see:

reg = LinearRegression()
reg.fit(df[['experience_level']],df['salary'])

plt.xlabel('Experience Level)')
plt.ylabel('Salary')

plt.scatter(df['experience_level'],df['salary'])   

xekseni = df['experience_level']
yekseni = reg.predict(df[['experience_level']])
plt.plot(xekseni, yekseni,color= "green", label = "linear regression")
plt.legend()
plt.show()


# #### Very bad model prediction, so: It is not correct to apply linear regression for this dataset. Remember, you will choose a model according to your data set! 
# 
# #### First of all, you should have a very good understanding of your dataset !!!
# 

# ### We decided that polynomial regression, one of the regression types, should be applied for this data set. Now let's see how we implement it:

# We adapt our x value to fit the polynomial function above
# 
# So => 1, x, x^2 (N=2) 

# In[13]:


# We call the PolynomialFeatures function to create a polynomial regression object.
# We specify the degree (N) of the polynomial when calling this function:
polynomial_regression = PolynomialFeatures(degree = 4)

x_polynomial = polynomial_regression.fit_transform(df[['experience_level']])


# In[14]:


# We fit the x_polynomial and y axes by creating our reg object, which is our regression model object,
# and calling its fit method.
# So we train our regression model with data:
reg = LinearRegression()
reg.fit(x_polynomial,df['salary'])


# ### Now that our model is ready, let's see how our model generates a result graph based on the available data:

# In[15]:


y_head = reg.predict(x_polynomial)
plt.plot(df['experience_level'],y_head,color= "red",label = "polynomial regression")
plt.legend()

# Let's scatter our data set as points and see if it fits polynomial regression:
plt.scatter(df['experience_level'],df['salary'])   

plt.show()


# As you can see, we can say that it definitely fits, polynomial regression is the right choice.
# Now let's make N=3 or 4 and see if we increase the polynomial degree, will it fit better?

# ### Calculate of a new employee who has experience level 4.5 

# In[16]:



x_polynomial1 = polynomial_regression.fit_transform([[4.5]])
reg.predict(x_polynomial1)


# #### The salary he will receive fits the company policy very well ! :)

# In[ ]:





# So using our new machine learning model, our HR department can easily calculate new employee salaries perfectly.

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




