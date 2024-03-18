#!/usr/bin/env python
# coding: utf-8

# # Recognition & Classification of Images using Deep Learning Algorithms
# 
#     
#     

# In[ ]:


# We start coding by importing our libraries..
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# #### We load our CIFAR10 dataset: (You must have an Internet connection for the download process). If you do not have a connection, you can also download the dataset from the Internet.

# In[ ]:


(X_train, y_train), (X_test,y_test) = datasets.cifar10.load_data()


# In[ ]:


X_train.shape


# #### Our array is like this because each photo has a square size of 32 pixels-32 pixels and has 3 channel RGB information in color.
# 
# <IMG src="cifar10_images.jpg" width="400" height="400">
# 

# In[ ]:


X_test.shape


# In[ ]:


y_train[:3]


# y_train and y_test are kept as a 2-dimensional array in the cifar10 dataset.
# We make this data one-dimensional to understand it visually more easily.
# We use reshape() to make a 2-dimensional array one-dimensional.

# In[ ]:


y_test = y_test.reshape(-1,)


# In[ ]:


y_test 


# #### Let's take a look at the data. for this purpose we create an array ourselves:

# In[ ]:


image_classes = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]


# In[ ]:


def plot_sample(X, y, index):
    plt.figure(figsize = (15,2))
    plt.imshow(X[index])        
    plt.xlabel(image_classes[y[index]])
    plt.show()


# In[ ]:


plot_sample(X_test, y_test, 0)


# In[ ]:


plot_sample(X_test, y_test, 1)


# In[ ]:


plot_sample(X_test, y_test, 3)


# ### Normalization
# 
# We need to normalize our data. Otherwise, CNN algorithms may give wrong results. Since the photos have 3 channels in RGB and each pixel has a value between 0-255, it is enough to simply divide each pixel value by 255 for normalization.

# In[ ]:


X_train = X_train / 255
X_test = X_test / 255


# ### We are designing our Deep Learning Algorithm using Convolutional Neural Network:

# <IMG src="deep7.png" width="800" height="400">

# In[ ]:


deep_learning_model = models.Sequential([
    # The first part is the Convolution layer..
    # In this part, we extract the features from the photos to be able to identify them...
    layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # The second part is a classical Articial Neural Network layer.
    # We will train our ANN model according to the above features and training information.
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])


# In[ ]:


deep_learning_model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# ### Let's start training our model now...

# In[ ]:


deep_learning_model.fit(X_train, y_train, epochs=5)


# In[ ]:


deep_learning_model.evaluate(X_test,y_test)


# In[ ]:


y_pred = deep_learning_model.predict(X_test)
y_pred[:3]


# In[ ]:


y_predictions_siniflari = [np.argmax(element) for element in y_pred]
y_predictions_siniflari[:3]


# In[ ]:


y_test[:3]


# In[ ]:


plot_sample(X_test, y_test,0)


# In[ ]:


image_classes[y_predictions_siniflari[0]]


# In[ ]:


plot_sample(X_test, y_test,1)


# In[ ]:


image_classes[y_predictions_siniflari[1]]


# In[ ]:


plot_sample(X_test, y_test,2)


# In[ ]:


image_classes[y_predictions_siniflari[2]]


# In[ ]:





# In[ ]:


plot_sample(X_test, y_test,12)


# In[ ]:


image_classes[y_predictions_siniflari[12]]


# In[ ]:




