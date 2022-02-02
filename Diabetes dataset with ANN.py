#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from keras.models import Sequential
from keras.layers import Dense


# In[2]:


# Diabetes dataset is also directly available in keras library
df=pd.read_csv(r'C:\Users\DELL\Downloads\archive\diabetes.csv')
df


# In[3]:


x=df.drop("Outcome",axis=1)
y=df["Outcome"]


# In[4]:


# One hot encoding of label variable (Note-it is not required for binary classification but for multiclass. Ijust did it for practice and experimentation.)
from keras.utils import to_categorical
import numpy as np
temp=[]
for i in range(len(y)):
               temp.append(to_categorical(y[i], num_classes=2))
y = np.array(temp)


# In[5]:


# Correlation Matrix used for knowing relation between features and label and to check if there is any collinearity among features
import seaborn as sns
sns.heatmap(df.corr(), annot = True)


#  we can see features like Glucose, Age and BMI contribute more than others.

# We cannot see much of collinearity in data apart from pregnancy and age we are trying to ignore this for some time right now but you can try to remove collinearity and try performing again.

# In[6]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# In[7]:


#plot to check distribution of each 
import matplotlib.pyplot as plt
x_train
number_of_columns=8
l=x_train.columns.values
number_of_rows=len(l)-1/number_of_columns

plt.figure(figsize=(2*number_of_columns,5*number_of_rows))
for i in range(0,len(l)):
    plt.subplot(number_of_rows + 1,number_of_columns,i+1)
    sns.distplot(x_train[l[i]],kde=True)


#  As you can see features like Bloodpressure, Glucose, BMI are having normal distribution (some entries of there features are showing zero but lets consider those as inconsistent as values for those features cannot be zero for actual human case so those are because of human error). (Note-You can try deleting these tuples to make data more consistent and again try to fit algorithm results will be better.) Ohter features do not follow gaussian distribution so we will use standerdization.

# In[8]:


x_train.shape[0]
print(l)


# In[9]:


# Normalization or standerdization is done for feature scling.
# As KNN does not require features in guassian form we used standerdization (mean=0, standerd deviation=1) otherwise you can do normalization(all values between 0 to 1)
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)


# In[11]:


x_train


# In[12]:


#Model initialization
model=Sequential()
# You can change number of neurons and activation function according to whichever gave you optimal result(not applicable for output layer.
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
# In output as I have encoded thats why used 2 neurons otherwise one neuron is sufficient and use softmax for multiclass classification and number of neurons according to number of classes.
model.add(Dense(2, activation='sigmoid'))


# In[13]:


# For classification loss function is cross entropy whether it is multiclass or binary.
# Optimizer and metrics convinient to you (you can experiment here as well.)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[14]:


# Model training and validation(as there is no validation set i have used test set for validation as well but you can againg split data into validation and training.)
model.fit(x_train, y_train, epochs=15, batch_size=30,validation_data=(x_test,y_test))


# we can see accuracy settles more or less at 75%. 
# You can always try different combination of neurons and activation function ( might get better results)

# In[24]:


predictions = model.predict(x_test)
pred=np.argmax(predictions,axis=1)
pred


# In[26]:


y_test=np.argmax(y_test,axis=1)
y_test


# In[27]:


from sklearn.metrics import classification_report
print(classification_report(y_test,pred))


# Here we can see precision by each output (in this case 0 and 1). 
# We can say that our model is more precise for 0 which in this case is no diabetes than 1 i.e. Diabetes patient.
# So there is scope of improvement (I'll try to make improvements but you can always try somethings that I have mentioned that might get better results)

# In[ ]:




