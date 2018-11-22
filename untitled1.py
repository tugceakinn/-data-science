# -*- coding: utf-8 -*-
"""
Created on Fri Sep  7 15:05:17 2018

@author: win10
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# veri yukleme

verile = pd.read_csv('tel1.csv',sep=";",engine ='python')
print(verile)

veriler.info()
veriler["tenure"].value_counts(dropna =False)
veriler["MonthlyCharges"].value_counts(dropna =False)
veriler["TotalCharges"].value_counts(dropna =False)
veriler["SeniorCitizen"].value_counts(dropna =False)

veriler["TotalCharges"].dropna(inplace = True)
veriler["MonthlyCharges"].dropna(inplace = True)

print(veriler)
#verionisleme

veriler.dtypes
#eksik veriler
veriler['TotalCharges'] = veriler['TotalCharges'].astype('float')

from sklearn.preprocessing import Imputer

imputer= Imputer(missing_values='NaN', strategy = 'mean', axis=0 )    

eksikveriler = veriler.iloc[:,18:20].values
print(eksikveriler)

imputer = imputer.fit(eksikveriler)
eksikveriler = imputer.transform(eksikveriler)
print(eksikveriler)