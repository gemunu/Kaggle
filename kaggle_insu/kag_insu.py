#!/usr/bin/env

import pandas as pd
import zipfile
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
#%matplotlib inline
from matplotlib import pyplot as plt
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

zf=zipfile.ZipFile('train.csv.zip')
zft=zipfile.ZipFile('test.csv.zip')
train=pd.read_csv(zf.open('train.csv'))
test=pd.read_csv(zft.open('test.csv'))

temp1=[]
for i in test.columns:
    if (test[i].describe()[0]) != 19765:
        #print (i,test[i].dtype,test[i].describe()[0])
        temp1.append(i)

meds=['Medical_History_1','Medical_History_10','Medical_History_24','Medical_History_32','Medical_History_15']

for i in meds:
    temp1.remove(i)

for i in temp1:
    men=test[i].mean()
    test[i].fillna(men,inplace=True)

for i in meds:
    mo=train[i].mode()[0]
    train[i].fillna(mo,inplace=True)

y=train.pop('Response')
ids=test.pop('Id')
train.drop('Id',axis=1,inplace=True)

le=LabelEncoder()
train['Product_Info_2']=le.fit_transform(train['Product_Info_2'])
test['Product_Info_2']=le.fit_transform(test['Product_Info_2'])

dtrain=xgb.DMatrix(train,label=y,missing=-1.)
dtest=xgb.DMatrix(test,missing=-1.)
param = {'bst:max_depth':7, 'bst:eta':0.05, 'silent':1, 'objective':'multi:softprob','num_class':9}
param["colsample_bytree"] = 0.30
param["subsample"] = 0.60
param["min_child_weight"] = 40
plst=list(param.items())

param = {'bst:max_depth':7, 'bst:eta':0.05, 'silent':1, 'objective':'multi:softprob','num_class':9,
    "colsample_bytree" : 0.30,"subsample": 0.60,"min_child_weight": 40}
plst=list(param.items())


bst=xgb.train(plst,dtrain,num_boost_round=80)
bst_pred2=bst.predict(dtest)

bc=RandomForestClassifier(n_estimators=600,max_depth=7)
bc.fit(train,y)
bcpred3=bc.predict_proba(test)

resps=[]
for i in range(len(bst_pred2)):
    max_p=[]
    for j in range(8):
        max_p.append((bst_pred2[i][j]+bst_pred1[i][j])+bcpred3[i][j]/3)
    #print (max_p)
    resps.append(max_p.index(max(max_p))+1)

sub = pd.DataFrame(np.column_stack((ids, resps)), columns=['Id', 'Response'])
sub.to_csv('sub.csv',index=False)
pd.read_csv('sub.csv')
