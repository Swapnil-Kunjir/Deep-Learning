#!/usr/bin/env python
# coding: utf-8

# In[1]:


import keras


# In[2]:


#importing dataset directly from keras datasets available
(x_train, y_train), (x_test, y_test)=keras.datasets.mnist.load_data(path="mnist.npz")


# In[3]:


x_train.shape


# In[4]:


y_train.shape


# In[8]:


#visualization of first 10 images with their labels
import matplotlib.pyplot as plt
fig, axes = plt.subplots(ncols=10, sharex=False, 
    sharey=True, figsize=(20, 4))
for i in range(10):
    axes[i].set_title(y_train[i])
    axes[i].imshow(x_train[i], cmap='gray')
    axes[i].get_xaxis().set_visible(False)
    axes[i].get_yaxis().set_visible(False)
plt.show()


# In[9]:


#converting labels into one hot encoded values (this is required for applying ANN)
from keras.utils import to_categorical
import numpy as np
temp = []
for i in range(len(y_train)):
               temp.append(to_categorical(y_train[i], num_classes=10))
y_train = np.array(temp)
# Convert y_test into one-hot format
temp = []
for i in range(len(y_test)):
               temp.append(to_categorical(y_test[i], num_classes=10))
y_test = np.array(temp)


# In[10]:


y_train.shape


# In[11]:


y_test.shape


# In[12]:


from keras.layers import Dense, Flatten
from keras.models import Sequential


# In[13]:


#Model Initiation
model=Sequential()
#As input is in format of 2D array and Converting them into 1D
model.add(Flatten(input_shape=(28,28)))
#number of neurons and activation function are hyperparameters you can choose whichever can give you optimal solution
model.add(Dense(16,activation="sigmoid"))
#activation function is softmax as our output is multiclass not binary
model.add(Dense(10,activation="softmax"))


# In[14]:



model.summary()


# In[15]:


# Model Compilation
# As we are using NN for classification we have to use loss function as cross entropy. optimizer and metrics can be of your choice
model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["acc"])


# In[16]:


# Model Training
model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=10)


# In[17]:


prediction=model.predict(x_test)


# In[18]:


prediction


# In[21]:


# Argmax returns the indices of the maximum values along an axis which in our case are nothing but number between 0 to 9 with highest probability predicted by ANN
pred=np.argmax(prediction,axis=1)
pred


# In[23]:


# Visualization of test images with predicted outcomes
fig, axes = plt.subplots(ncols=10, sharex=False,
                         sharey=True, figsize=(20, 4))
for i in range(10):
    axes[i].set_title(pred[i])
    axes[i].imshow(x_test[i], cmap='gray')
    axes[i].get_xaxis().set_visible(False)
    axes[i].get_yaxis().set_visible(False)
plt.show()


# In[ ]:




