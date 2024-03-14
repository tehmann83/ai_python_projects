#!/usr/bin/env python
# coding: utf-8

# # Predicting Diabets using ANN with 2 Hidden Layers and 25 Neurons
#    
# 

# ##### Our 8 Inputs are: Pregnancies	Glucose	BloodPressure	SkinThickness	Insulin	BMI	DiabetesPedigreeFunction	Age
# 
# We will use Prima Indians Dataset from Kaggle. You can download dataset from Kaggle: https://www.kaggle.com/uciml/pima-indians-diabetes-database
# 
# 
# <IMG src="Example2ArtificialNeuralNetwork.jpg" width="500" height="500">

# In[1]:


# We start by importing our modules.
import numpy as np
from tensorflow import keras
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.layers import Dense
from keras.layers import Dense
from keras import Sequential


# In[2]:


# Outcome = 1 Have Diabetes
# Outcome = 0 Healthy
df = pd.read_csv("diabetes.csv")
df.head()


# ### Train/test split

# In[3]:


X_train, X_test, y_train, y_test = train_test_split(df[['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction', 'Age']],df.Outcome,test_size=0.2, random_state=25)


# In[4]:


#Standard normalization
X_train_scaled = X_train.copy()
X_train_scaled = StandardScaler().fit_transform(X_train_scaled)

X_test_scaled = X_test.copy()
X_test_scaled = StandardScaler().fit_transform(X_test_scaled)



# ### We are building the model with 2 Hidden Layers and Total 25 Neurons
# 
# 

# In[5]:


# We are building the model with a 2 hidden layers with total 25 neurons:
model = Sequential()
model.add(Dense(16, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))   # this is output layer - should have as many neurons as there are outputs to the classification problem.

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train_scaled, y_train, epochs=500)



# ### Lets look at the accuracy of our model:

# In[6]:


model.evaluate(X_test_scaled,y_test)


# In[7]:


y_test


# In[8]:


model.predict(X_test_scaled)


# In[ ]:





# In[11]:


# New Prediction: 

sc = StandardScaler()
sc.fit_transform(X_train)

new_prediction = model.predict(sc.transform(np.array([[6,148,72,35,0,33.6,0.627,50]])))
new_prediction[0]


# In[ ]:


# With a probability of 0.9951345 we can say the new person has diabets.


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




