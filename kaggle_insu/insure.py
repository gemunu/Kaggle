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


zf=zipfile.ZipFile('train.csv.zip')
zft=zipfile.ZipFile('test.csv.zip')
train=pd.read_csv(zf.open('train.csv'))
test=pd.read_csv(zft.open('test.csv'))

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

print ("Temp1:%"%(k))

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

y=train.pop('Response')
ids=test.pop('Id')
train.drop('Id',axis=1,inplace=True)

le=LabelEncoder()
train['Product_Info_2']=le.fit_transform(train['Product_Info_2'])
test['Product_Info_2']=le.fit_transform(test['Product_Info_2'])

import xgboost as xgb

dtrain=xgb.DMatrix(train,label=y)
dtest=xgb.DMatrix(test)

param = {'bst:max_depth':9, 'bst:eta':0.7, 'silent':1, 'objective':'multi:softmax','num_class':9}
param["colsample_bytree"] = 0.30
param["subsample"] = 0.60
param["min_child_weight"] = 1
param['nthread'] = 4
plst=list(param.items())

def eval_wrapper(yhat, y):
    y = np.array(y)
    y = y.astype(int)
    yhat = np.array(yhat)
    yhat = np.clip(np.round(yhat), np.min(y), np.max(y)).astype(int)
    return quadratic_weighted_kappa(yhat, y)
bst=xgb.train(plst,dtrain,num_boost_round=100)
bst_pred=bst.predict(dtest)
bst_pred=bst_pred.tolist()

for i in range(len(bst_pred)):
    bst_pred[i].pop(0)

xgbc=xgb.sklearn.XGBClassifier(max_depth=7,n_estimators=200,learning_rate=0.05,objective='multi:softmax',subsample=0.6,missing=-1.,colsample_bytree=0.3,min_child_weight=40)
xgbc.fit(train,y)

bstpred=bst.predict(dtrain)
eval_wrapper(y,bstpred)

rfc=RandomForestClassifier(n_estimators=600,max_depth=9,criterion='entropy',max_features=0.33)
rfc.fit(train,y)

bc=GradientBoostingClassifier(n_estimators=100,max_features=0.33)
bc.fit(train,y)

from sklearn.cross_validation import cross_val_score
cross_val_score(bc,train,y).mean()


resps=[]
for i in range(len(bst_pred)):
    max_p=[]
    for j in range(8):
        max_p.append((bst_pred[i][j]+bc_pred[i][j])+rfc_pred[i][j]/3)
    #print (max_p)
    resps.append(max_p.index(max(max_p))+1)

sps1=[]
for i in range(len(bst_pred)):
    max_p=[]
    for j in range(8):
        max_p.append((bst_pred[i][j]*bc_pred[i][j])*rfc_pred[i][j])
    #print (max_p)
    max_p=[i/sum(max_p) for i in max_p]
    resps1.append(max_p.index(max(max_p))+1)
"""
k=[i for i,j in zip(resps1,resps) if i!=j]
len(k)
imp_feats=pd.Series(bc.feature_importances_,index=train.columns)
imp_feats.sort()
imp_feats.plot(kind='barh')
sorted(zip(map(lambda x: round(x, 4), bc.feature_importances_), train.columns),
             reverse=True)
sorted(zip(map(lambda x: round(x, 4), bc.feature_importances_), train.columns),
             reverse=True)
"""

sub = pd.DataFrame(np.column_stack((ids, resps1)), columns=['Id', 'Response'])
sub.to_csv('sub.csv',index=False)
pd.read_csv('sub.csv')
