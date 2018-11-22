# -*- coding: utf-8 -*-
"""
Created on Sun Sep  9 14:59:14 2018

@author: win10
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


verile = pd.read_csv('tel1.csv',sep=";",engine ='python')
print(verile)


verile = verile[pd.to_numeric(verile['MonthlyCharges'], errors='coerce').notnull()]
verile['MonthlyCharges']=verile['MonthlyCharges'].astype('float')
print(verile['MonthlyCharges'].dtype)
print(verile)


#verile['TotalCharges'] = verile['TotalCharges'].astype('float')
#verile['MonthlyCharges'] = verile['MonthlyCharges'].astype('float')
#print(verile)


#verile["MonthlyCharges"].value_counts(dropna =False)
#verile["TotalCharges"].value_counts(dropna =False)

#print(verile)


#eksikveriler
from sklearn.preprocessing import Imputer

imputer= Imputer(missing_values='NaN', strategy = 'mean', axis=0 )  
eksikveriler = verile.iloc[:,18:19].values
print(eksikveriler)


imputer = imputer.fit(eksikveriler)
eksikveriler = imputer.transform(eksikveriler)
print(eksikveriler)




#encoder:  Kategorik -> Numeric

from sklearn.preprocessing import LabelEncoder

veriler2 = verile.apply(LabelEncoder().fit_transform)
print(veriler2)

from sklearn.preprocessing import OneHotEncoder

PaymentMethod = veriler2.iloc[:,17:18]
ohe = OneHotEncoder(categorical_features='all')
PaymentMethod=ohe.fit_transform(PaymentMethod).toarray()
print(PaymentMethod)

contract = veriler2.iloc[:,15:16]
ohe = OneHotEncoder(categorical_features='all')
contract=ohe.fit_transform(contract).toarray()
print(contract)

Multiplelines = veriler2.iloc[:,7:8]
ohe = OneHotEncoder(categorical_features='all')
Multiplelines=ohe.fit_transform(Multiplelines).toarray()
print(Multiplelines)

InternetService = veriler2.iloc[:,8:9]
ohe = OneHotEncoder(categorical_features='all')
InternetService=ohe.fit_transform(InternetService).toarray()
print(InternetService)

OnlineSecurty = veriler2.iloc[:,9:10]
ohe = OneHotEncoder(categorical_features='all')
OnlineSecurty=ohe.fit_transform(OnlineSecurty).toarray()
print(OnlineSecurty)

OnlineBackup = veriler2.iloc[:,10:11]
ohe = OneHotEncoder(categorical_features='all')
OnlineBackup=ohe.fit_transform(OnlineBackup).toarray()
print(OnlineBackup)

DeviceProtection = veriler2.iloc[:,11:12]
ohe = OneHotEncoder(categorical_features='all')
DeviceProtection=ohe.fit_transform(DeviceProtection).toarray()
print(DeviceProtection)

TechSupport = veriler2.iloc[:,12:13]
ohe = OneHotEncoder(categorical_features='all')
TechSupport=ohe.fit_transform(TechSupport).toarray()
print(TechSupport)

StreamingTV = veriler2.iloc[:,13:14]
ohe = OneHotEncoder(categorical_features='all')
StreamingTV=ohe.fit_transform(StreamingTV).toarray()
print(StreamingTV)

StreamingMovies = veriler2.iloc[:,14:15]
ohe = OneHotEncoder(categorical_features='all')
StreamingMovies=ohe.fit_transform(StreamingMovies).toarray()
print(StreamingMovies)

#kolon oluşturma
Payment= pd.DataFrame(data = PaymentMethod, columns=['Electronic check','Mailed check','Bank transfer','Credit card'])
Streaming= pd.DataFrame(data = StreamingMovies , columns=['stremng_yes','streamng_no','stremng_no internet'])
StreamngTv= pd.DataFrame(data = StreamingTV , columns=['streamngtv_yes','streamngtv_no','streamngtv_no internet'])
Techspprt= pd.DataFrame(data = TechSupport , columns=['techspprt_yes','techspprt_no','techspprt_no internet'])
Deviceprtction= pd.DataFrame(data = DeviceProtection , columns=['Deviceprtction_yes','Deviceprtction_no','Deviceprtction_no internet'])
OnlineBckup= pd.DataFrame(data = OnlineBackup , columns=['OnlineBackup_yes','OnlineBackup_no','OnlineBackup_no internet'])
OnlineScurty= pd.DataFrame(data = OnlineSecurty , columns=['OnlineSecurty_yes','OnlineSecurty_no','OnlineSecurty_no internet'])
Multiplelins= pd.DataFrame(data = Multiplelines , columns=['Multiplelines_yes','Multiplelines_no','Multiplelines_no internet'])
InternetSrvce= pd.DataFrame(data = InternetService , columns=['DSL','FiberOptic','No'])
cntract= pd.DataFrame(data = contract , columns=['Month-to-month','One year','Two year'])

#dataframelerin birleşimi
sonveriler = pd.concat([Payment,Streaming,StreamngTv,Techspprt,Deviceprtction,OnlineBckup,OnlineScurty,Multiplelins,InternetSrvce,cntract],axis = 1)
sonveriler1=pd.concat([sonveriler,verile.iloc[:,2:3]], axis= 1 )
sonveriler2=pd.concat([sonveriler1,verile.iloc[:,5:6]], axis= 1 )
sonveriler3=pd.concat([sonveriler2,eksikveriler], axis= 1 )


sonuc=pd.concat([sonveriler3,veriler2.iloc[:,3:5]],axis=1)
sonuc1=pd.concat([sonuc,veriler2.iloc[:,6:7]],axis=1)
sonuc2=pd.concat([sonuc1,veriler2.iloc[:,16:17]],axis=1)
sonsonuc=pd.concat([sonuc2,veriler2.iloc[:,-1:]],axis=1)



#verilerin egitim ve test icin bolunmesi
from sklearn.cross_validation import train_test_split
x_train, x_test,y_train,y_test = train_test_split(sonsonuc.iloc[:,:-1],sonsonuc.iloc[:,-1:],test_size=0.33, random_state=0)


#veri olcekleme
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)


#backward elimination
import statsmodels.formula.api as sm 

X = np.append(arr = np.ones((7043,1)).astype(int), values=sonsonuc.iloc[:,:-1], axis=1 )
X_l = sonsonuc.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]].values
r_ols = sm.OLS(endog = sonsonuc.iloc[:,-1:], exog =X_l)
r = r_ols.fit()
print(r.summary())














