#!/usr/bin/env python
# coding: utf-8

# # Predicting Diabets using 1 Neuron Artificial Neural Network 
# 
# 
# We are going to build a solution ANN model with only 1 Neuron:  
# 

# ##### Our 8 Inputs are: Pregnancies	Glucose	BloodPressure	SkinThickness	Insulin	BMI	DiabetesPedigreeFunction	Age
# We will use Prima Indians Dataset from Kaggle.
# You can download dataset from Kaggle: https://www.kaggle.com/uciml/pima-indians-diabetes-database
# 
# 

# In[ ]:


# We start by importing our modules.
import numpy as np
from tensorflow import keras
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# In[ ]:


# Outcome = 1 Has Diabetes
# Outcome = 0 Healthy
# You can download dataset from Kaggle: https://www.kaggle.com/uciml/pima-indians-diabetes-database

df = pd.read_csv("diabetes.csv")
df.head()


# ### Train/test split

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(df[['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction', 'Age']],df.Outcome,test_size=0.2, random_state=25)


# In[ ]:


#Standard normalization
X_train_scaled = X_train.copy()
X_train_scaled = StandardScaler().fit_transform(X_train_scaled)

X_test_scaled = X_test.copy()
X_test_scaled = StandardScaler().fit_transform(X_test_scaled)



# ### We are building the model with only 1 neuron!

# In[ ]:


# We are building the model with only 1 neuron!
model = keras.Sequential([
    keras.layers.Dense(1, input_shape=(8,), activation='sigmoid', kernel_initializer='ones', bias_initializer='zeros')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X_train_scaled, y_train, epochs=1000)




# ### Lets look at the accuracy of our model:

# In[ ]:


model.evaluate(X_test_scaled,y_test)


# In[ ]:


y_test


# In[ ]:


model.predict(X_test_scaled)


# Lets get the coefficients of our model: (w1, w2, ... w8 and bias)

# In[ ]:


coef, intercept = model.get_weights()


# In[ ]:


print(coef, intercept)


# That's all for simple 1 neuron neural network!

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




