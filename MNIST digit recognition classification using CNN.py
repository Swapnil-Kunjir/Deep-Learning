#!/usr/bin/env python
# coding: utf-8

# In[1]:


import keras
#importing dataset directly from ones available in keras 
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()


# In[2]:


x_train.shape


# In[3]:


#Reshaping dataset as CNN requires 3 dimensions
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
#here xtrain.shape[0] is no. of images
#28x28 is (28,28) image size in pixel and 1 because it is black and white image
#(if it was colored we would have written 3 instead of 1) 
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)


# In[4]:


# again we convert images into floating type as images originally have uint8 format on which mathmatical operations cannaot be performed
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')


# In[5]:


#normalization of pixels into 0-1 form as they are in 0-255 form
x_train /= 255
x_test /= 255


# In[11]:


#converting labels into one hot encoded value as NN requirement (for only multiclass classification not for regression or binary classification)
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


# In[26]:


from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
#initiating the model
model = Sequential()
#convolution and max pooling layers are applied for feature extraction as well as size reduction of original image
model.add(Conv2D(32, kernel_size=(3,3), input_shape=input_shape,activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
#flattening image as NN requires 1D array
model.add(Flatten())
# number of neurons in hidden layer as well as activation function can be selected according to you
# You can select any which gives you optimal solution
model.add(Dense(128, activation="relu"))
# Dropout just drops x% of neurons which are selected randomly
model.add(Dropout(0.2))
#in output layer though activation function and neurons depend on your question in concern (eg binary/multiclass classification/regression)
model.add(Dense(10,activation="softmax")


# In[27]:


model.summary()


# In[28]:


#model compilation loss is crossentropy if we are performing classification, metrics and optimizer can be choosed according to user.
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# In[29]:


# model training and validation
model.fit(x=x_train,y=y_train, epochs=10,validation_data=(x_test,y_test),batch_size=30)


# In[30]:


# prediction
prediction=model.predict(x_test)
prediction


# In[31]:


# argmax returns index of most probable outcome according to our prediction
pred=np.argmax(prediction,axis=1)
pred


# In[32]:


# visualization with predicted labels
import matplotlib.pyplot as plt
fig, axes = plt.subplots(ncols=10, sharex=False,
                         sharey=True, figsize=(20, 4))
for i in range(10):
    axes[i].set_title(pred[i])
    axes[i].imshow(x_test[i].reshape(28, 28), cmap='gray')
    axes[i].get_xaxis().set_visible(False)
    axes[i].get_yaxis().set_visible(False)
plt.show()


# In[ ]:




