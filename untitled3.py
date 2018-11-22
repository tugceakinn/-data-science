# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 17:00:25 2018

@author: win10
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 04:18:20 2018

@author: sadievrenseker
"""

#1. kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# veri yukleme

veri = pd.read_csv('tel.csv',sep=";",engine ='python')
print(veri)

#veri["TotalCharges"].value_counts(dropna =False)

#verionisleme

#veri["TotalCharges"].value_counts(dropna =False)
#veri["MonthlyCharges"].value_counts(dropna =False)
#veri['MonthlyCharges'] = veri.MonthlyCharges.astype('int')

veri= veri[pd.to_numeric(veri['MonthlyCharges'], errors='coerce').notnull()]
print(veri['MonthlyCharges'].dtype)

veri = veri[pd.to_numeric(veri['TotalCharges'], errors='coerce').notnull()]

veri['MonthlyCharges']=veri['MonthlyCharges'].astype('float')

veri['TotalCharges']=veri['TotalCharges'].astype('float')



print(veri['MonthlyCharges'].dtype)
print(veri['TotalCharges'].dtype)
print(veri)

veri.dtypes


"""
#veri['MonthlyCharges'] = veri['MonthlyCharges'].astype('float')
#eksikveriler
from sklearn.preprocessing import Imputer

imputer= Imputer(missing_values='NaN', strategy = 'mean', axis=0 )  
eksikveriler = veri.iloc[:,18:20].values
print(eksikveriler)


imputer = imputer.fit(eksikveriler)
eksikveriler = imputer.transform(eksikveriler)
print(eksikveriler)
"""


#encoder:  Kategorik -> Numeric
from sklearn.preprocessing import LabelEncoder

veriler2 = veri.apply(LabelEncoder().fit_transform)
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
sonveriler1=pd.concat([sonveriler,veri.iloc[:,2:3]], axis= 1 )
sonveriler2=pd.concat([sonveriler1,veri.iloc[:,5:6]], axis= 1 )
sonveriler3=pd.concat([sonveriler2,veri.iloc[:,18:19]], axis= 1 )
sonveriler4=pd.concat([sonveriler3,veri.iloc[:,19:20]], axis= 1 )

sonuc=pd.concat([sonveriler4,veriler2.iloc[:,3:5]],axis=1)
sonuc1=pd.concat([sonuc,veriler2.iloc[:,6:7]],axis=1)
sonuc2=pd.concat([sonuc1,veriler2.iloc[:,16:17]],axis=1)
sonsonuc=pd.concat([sonuc2,veriler2.iloc[:,-1:]],axis=1)

a=sonsonuc.fillna(value=0)


#eksikveriler
from sklearn.preprocessing import Imputer

imputer= Imputer(missing_values='NaN', strategy = 'mean', axis=0 )  
eksik = sonsonuc.iloc[:,0:30].values
print(eksik)


imputer = imputer.fit(eksik)
eksik = imputer.transform(eksik)
print(eksik)



#verilerin egitim ve test icin bolunmesi
from sklearn.cross_validation import train_test_split
x_train, x_test,y_train,y_test = train_test_split(a.iloc[:,:-1],a.iloc[:,-1:],test_size=0.33, random_state=0)



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


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=10, criterion = 'entropy')
rfc.fit(X_train,y_train)


from sklearn.metrics import confusion_matrix
y_pred = rfc.predict(X_test)
cm = confusion_matrix(y_test,y_pred)
print('RFC')
print(cm)















