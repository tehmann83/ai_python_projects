#!/usr/bin/env python
# coding: utf-8

# # Image Classification using Transfer Learning
# 
# In this project we will make image classification using Transfer Learning. For this purpose we will use the InceptionResNetV2 model which is trained on the imageNet data set. Lets use a submarine image for classification. I found this image in Internet, this image is not included in imageNet dataset which was used for training InceptionResNetV2.
# 
# In transfer learning we use a model that has been previously trained on a dataset and contains the weights and biases that represent the features of whichever dataset it was trained on. 
# 
# Inception and ResNet have been among the best image recognition performance models in recent years, with very good performance at a relatively low computational cost. Inception and ResNet combines the Inception architecture, with residual connections.

# In[21]:


import numpy as np
from PIL import Image # Python Imaging Library - For operations like: Image open, resize image, etc..
from IPython.display import Image as show_image  # For displaying our test images to you
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input, decode_predictions


# #### The InceptionResNetV2 pre-trained model expects inputs of 299x299 resolution.  
# ##### InceptionResNetV2 model will classify images into one of 1,000 possible categories.
# 
# 

# In[22]:


# Let's load our image and rescale it to the resolution of 299x299 which InceptionResNetV2 requires..
img = Image.open("sportscar.jpg").resize((299,299))


# In[23]:


# We must convert it to array for operations...
img = np.array(img)


# In[24]:


# Let's look it's shape..
img.shape


# In[25]:


print(img.ndim)


# In[26]:


# We have to add an extra dimension to our array so we will reshape it.. 
img = img.reshape(-1,299,299,3)   # with reshape(-1,..) I'm adding 1 extra dimension..
                                  # I do this because my model requires 4 dim array!


# In[27]:


# Let's look it's shape..
img.shape


# In[28]:


print(img.ndim)


# In[7]:


# I will scale input pixels between -1 and 1 using my model's preprocess_input
# InceptionResNetV2 model requires it..
img = preprocess_input(img)   


# Let's load up the model itself:

# In[8]:


incresv2_model = InceptionResNetV2(weights='imagenet', classes=1000)   # InceptionResNetV2 will classify images into one of 
                                                                       # 1,000 possible categories.


# #### Lets inspect InceptionResNetV2 model

# In[9]:


# Now look at it's layers:
print(incresv2_model.summary())
print(type(incresv2_model))


# In[10]:


# Before prediction let's see our image with our eyes first:
show_image(filename='sportscar.jpg') 


# It's already trained with weights learned from the Imagenet data set. Now we will use it by calling incresv2_model's predict() method:

# In[11]:


preds = incresv2_model.predict(img)
print('Predicted categories:', decode_predictions(preds, top=2)[0]) # decode the results into a list of tuples 


# ### Lets make another prediction.. I downloaded all these images from the web. These images are not in ImageNet. you can try yourself, download any image from the web and make a prediction using the model...

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ### reshape

# In[13]:


import numpy as np

arr = np.arange(24)

print(arr)


# In[14]:


arr.shape


# In[15]:


arr.ndim


# As you can see above array arr has only 1 dimension. It is the simplest array since it has the min possible dimension size:1..

# ### Increasing the dimension size of a numpy array using reshape(-1,..)

# In[16]:


arr = arr.reshape(-1,24) #We added an extra dimension to array a with reshape(-1,24)


# In[17]:


arr.shape


# In[18]:


arr


# In[19]:


arr.ndim


# As you can see above we have increased the dimension of our array arr. Now it is 2 dimensional... Original values have become the first row now!

# In[ ]:




