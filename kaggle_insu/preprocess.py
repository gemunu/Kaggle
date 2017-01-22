import pandas as pd
import zipfile
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
#get_ipython().magic('matplotlib inline')
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestRegressor,BaggingRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from ml_metrics import quadratic_weighted_kappa
import numpy as np
import xgboost as xgb
from sklearn.cross_validation import train_test_split

train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')

temp1=[]
for i in test.columns:
    if (test[i].describe()[0]) != 19765:
        print (i,test[i].dtype,test[i].describe()[0])
        temp1.append(i)
"""#temp=[]
for i in train.columns:
    if (train[i].describe()[0]) != 59381:
        print (i,train[i].dtype,train[i].describe()[0])
        #temp.append(i)
"""
meds=['Medical_History_1','Medical_History_10','Medical_History_24','Medical_History_32','Medical_History_15']
for i in meds:
    temp1.remove(i)
print (temp1,meds)


for i in temp1:
    men=train[i].mean()
    test[i].fillna(men,inplace=True)
    train[i].fillna(men,inplace=True)

for i in meds:
    mo=test[i].mode()[0]
    print (i,mo)
    test[i].fillna(mo,inplace=True)
    train[i].fillna(mo,inplace=True)

y=train.pop('Response')
ids=test.pop('Id')
train.drop('Id',axis=1,inplace=True)

#train.drop('Insurance_History_5',axis=1,inplace=True)
#test.drop('Medical_History_24',axis=1,inplace=True)
#test.drop('Medical_History_32',axis=1,inplace=True)
#train.drop('Medical_History_10',axis=1,inplace=True)
#train.drop('Medical_History_24',axis=1,inplace=True)
#train.drop('Medical_History_32',axis=1,inplace=True)
#temp1.remove('Insurance_History_5')
#temp1.remove('Medical_History_10')
#temp.remove('Medical_History_10')
#temp.remove('Insurance_History_5')
#train.drop('Medical_History_10',axis=1,inplace=True)
#test.drop('Medical_History_10',axis=1,inplace=True)

for i in temp1:
    men=train[i].mean()
    test[i].fillna(men,inplace=True)
    train[i].fillna(men,inplace=True)

for i in meds:
    mo=test[i].mode()[0]
    print (i,mo)
    test[i].fillna(mo,inplace=True)
    train[i].fillna(mo,inplace=True)

le=LabelEncoder()
train['Product_Info_2']=le.fit_transform(train['Product_Info_2'])
test['Product_Info_2']=le.fit_transform(test['Product_Info_2'])

x_train, x_test, y_train, y_test = train_test_split(train,y,test_size=0.33,random_state=42)

print (x_train,x_test,y_train,y_test)

dtrain=xgb.DMatrix(train,label=y)
dtest=xgb.DMatrix(test)
