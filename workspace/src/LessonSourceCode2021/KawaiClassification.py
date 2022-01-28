#!/usr/bin/env python
# coding: utf-8

# #sklearnのライブラリーを import する

# In[17]:


#pandas & numpy
import pandas as pd
import numpy as np

#Preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

#the classification algorithms
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

#for parameter optimization
from sklearn.model_selection import GridSearchCV

#for evaluation 
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn import metrics

#ignore all warnings
import warnings
warnings.filterwarnings("ignore")


# ##データセットをアップロードして、読み込む

# In[18]:


#read data file

#load data to pandas
data = pd.read_csv("incomedata.csv")


# ##データを検討する

# In[19]:


print (data.columns)
  


# In[20]:


data


# In[21]:


data.describe()


# #前処理 (Pre-Processing)

# ##One-Hot-Encoding
# 

# In[22]:



#deal with categoricial variables 
print ("one-hot encoder for gender")
ohe = OneHotEncoder(categories='auto')
feature_arr = ohe.fit_transform(data[['Gender']]).toarray()
genderData= pd.concat([data['Gender'],pd.DataFrame(feature_arr)],axis=1)
genderData


# In[23]:


#deal with categoricial variables 
ohe = OneHotEncoder(categories='auto')
feature_arr = ohe.fit_transform(data[['Employ_type', 'Education', 'Marriage', 'Occupation', 'Relationship','Race', 'Gender']]).toarray()

#move income column to the end 
incomeColumn = data['Income']
data.drop(labels=['Income'], axis=1,inplace = True)

#add to original data
convertedData= pd.concat([data, pd.DataFrame(feature_arr),incomeColumn], axis=1)
convertedData.head()


# In[24]:


#remove the original categorical data 
convertedData=convertedData.drop(columns=['Employ_type', 'Education', 'Marriage', 'Occupation', 'Relationship','Race', 'Gender'])
convertedData


# ##データの標準化
# 

# In[25]:


scaler=StandardScaler()
convertedData.iloc[:, 0:5]=scaler.fit_transform(convertedData.iloc[:, 0:5])
convertedData


# #機械学習の分類機をトレーニング

# ##X,Y配列の作成

# In[27]:


X = convertedData.iloc[:, 0:63].values 
Y = convertedData["Income"].values

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train,Y_test = train_test_split(X, Y, test_size=0.20, random_state=42)
print(X_train)
print(Y_train)


# ##モデルの宣言

# In[28]:


#SVM
svmCLF= SVC(C=1000,kernel="rbf")

#NaiveBayes
NBCLF= GaussianNB()

#Random Forest
RFCLF= RandomForestClassifier(max_depth=80,n_estimators=100)


# ##Cross-Validation の F-score の計算

# In[29]:


#SVM
scores = cross_val_score(svmCLF, X_train,Y_train, scoring="f1_macro", cv=5)
print("SVM Score")
print(scores)
print("SVM F1 Macro: %.2f%% (%.2f%%)" % (scores.mean()*100, scores.std()*100))

#Naive Bayes
scores = cross_val_score(NBCLF, X_train,Y_train, scoring="f1_macro", cv=5)
print("NB Score")
print(scores)
print("NB F1 Macro: %.2f%% (%.2f%%)" % (scores.mean()*100, scores.std()*100))

#Random Forest
scores = cross_val_score(RFCLF,X_train,Y_train, scoring="f1_macro", cv=5)
print("RF Score")
print(scores)
print("RF F1 Macro: %.2f%% (%.2f%%)" % (scores.mean()*100, scores.std()*100))


# #Modelを実験する

# In[30]:


#SVM
svmCLF= SVC(C=10,kernel="linear")
#NaiveBayes
NBCLF= GaussianNB()
#Random Forest
RFCLF= RandomForestClassifier(max_depth=100,n_estimators=100)

#新しいデータを予測する
svmCLF.fit(X_train,Y_train)
NBCLF.fit(X_train,Y_train)
RFCLF.fit(X_train,Y_train)

#実際にデータを予測する
svmPred=svmCLF.predict(X_test)
NBPred=NBCLF.predict(X_test)
RFPred=RFCLF.predict(X_test)

#予測したデータを元のデータに追加する
testDataWithPrediction= pd.concat([pd.DataFrame(Y_test,columns={"true"}),pd.DataFrame(svmPred,columns={"svmPred"})
                                   ,pd.DataFrame(NBPred,columns={"NBPred"}),pd.DataFrame(RFPred,columns={"RFPred"})],axis=1)

#ファイルをダウンロードする
PredictFile= pd.concat([pd.DataFrame(X_test,columns= convertedData.iloc[:, 0:63].columns),testDataWithPrediction],axis=1)
PredictFile.to_csv('outputClassification.csv')

#予測したデータ出力する
testDataWithPrediction


# In[ ]:




