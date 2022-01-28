#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
houseData=pd.read_csv("bostonhousing.csv")


# In[2]:


houseData


# In[3]:


from keras.models import Sequential
from keras.layers import Dense,Dropout

model= Sequential()
model.add(Dense(16,input_dim=13,activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(16,activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(1,activation="linear"))
model.summary()


# In[4]:


model.compile(loss="mse",optimizer="rmsprop",metrics=['mae'])


# In[5]:


from sklearn.model_selection import train_test_split

houseDataX=houseData.loc[:,'crim':'lstat']
houseDataY=houseData.loc[:,['medv']]

X_train, X_test, y_train, y_test = train_test_split(houseDataX, houseDataY, test_size=0.1,random_state=42)


# In[6]:


print(houseDataY.size)
print(y_train.size)
print(y_test.size)


# In[7]:


result=model.fit(X_train,y_train,validation_split=0.2,epochs=50,batch_size=8)


# In[8]:


import matplotlib.pyplot as plt
plt.plot(result.history['mae'],label="tranining")
plt.plot(result.history['val_mae'],label="validation")
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.legend()
plt.show()


# In[9]:


model.evaluate(X_test,y_test,batch_size=8)


# # 標準化

# In[10]:


from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()
houseDataX=houseData.loc[:,'crim':'lstat']
scaledHouseDataX=pd.DataFrame(scaler.fit_transform(houseDataX),columns=houseDataX.columns)

houseDataY=houseData.loc[:,['medv']]
X_train, X_test, y_train, y_test = train_test_split(scaledHouseDataX, houseDataY, test_size=0.1,random_state=42)


# In[11]:


modelScaled= Sequential()
modelScaled.add(Dense(16,input_dim=13,activation="relu"))
modelScaled.add(Dropout(0.5))
modelScaled.add(Dense(16,activation="relu"))
modelScaled.add(Dropout(0.5))
modelScaled.add(Dense(1,activation="linear"))
modelScaled.compile(loss="mse",optimizer="rmsprop",metrics=['mae'])
resultScaled=modelScaled.fit(X_train,y_train,validation_split=0.2,epochs=50,batch_size=8)


# In[12]:


plt.plot(resultScaled.history['mae'],label="tranining")
plt.plot(resultScaled.history['val_mae'],label="validation")
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.legend()
plt.show()


# In[13]:


modelScaled.evaluate(X_test,y_test,batch_size=8)


# In[14]:


modelScaled.save('housepredicterWeight.h5')


# # モデルの実装

# In[15]:


predictHousePriceData= pd.read_csv("BostonHousePricePredict.csv")
predictHousePriceData

scaler=StandardScaler()
housePredDataX=predictHousePriceData.loc[:,:]
scaledPrediHouseDataX=pd.DataFrame(scaler.fit_transform(housePredDataX),columns=housePredDataX.columns)
scaledPrediHouseDataX


# In[16]:


from keras.models import load_model

# Recreate the model from file
loadedModel = load_model('housepredicterWeight.h5')

#predict
predictedHousePrice=loadedModel.predict(scaledPrediHouseDataX)


# In[17]:


predictedHousePrice


# In[18]:


outputPrice= pd.DataFrame(pd.concat([housePredDataX,pd.DataFrame(predictedHousePrice,columns=["PredPrice"])],axis=1))
outputPrice


# In[19]:


outputPrice.to_csv("predictedPriceOutput.csv")

