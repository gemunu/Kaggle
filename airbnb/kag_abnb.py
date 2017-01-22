#!/usr/bin/python3

import pandas as pd
import seaborn as sns
#get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import numpy as np
import xgboost as xgb

train=pd.read_csv('train_users.csv')
test=pd.read_csv('test_users.csv')
sessions=pd.read_csv('sessions.csv')

test=pd.merge(test,sessions,left_on='id',right_on='user_id',how='left')
train=pd.merge(train,sessions,left_on='id',right_on='user_id',how='left')
test.to_csv('test.csv')
train.to_csv('train.csv')

target=train.pop('country_destination')

"""train.ix[train.age > 122, 'age'] = None
test.age.max()
train['age']=pd.cut(train['age'],12,labels=[i for i in range(1,13)])
test.ix[test.age > 122, 'age'] = None
test.age.max()
test.info()
test['age']=test['age'].astype(object)
train['age']=train['age'].astype(object)
train['age'].fillna(0,inplace=True)
test['age'].fillna(0,inplace=True)
"""
train.drop('timestamp_first_active',axis=1,inplace=True)
test.drop('timestamp_first_active',axis=1,inplace=True)
train.drop('date_first_booking',axis=1,inplace=True)
test.drop('date_first_booking',axis=1,inplace=True)

id_test=test['id']

train.drop(['id','user_id'],axis=1,inplace=True)
test.drop(['id','user_id'],axis=1,inplace=True)

from  pandas.tseries.common import DatetimeProperties as dt
#get_ipython().run_cell_magic('time', '', "train['month_created']=train['date_account_created'].apply(pd.to_datetime).dt.month\ntrain['day_created']=train['date_account_created'].apply(pd.to_datetime).dt.dayofweek")
#get_ipython().run_cell_magic('time', '', "test['month_created']=test['date_account_created'].apply(pd.to_datetime).dt.month\ntest['day_created']=test['date_account_created'].apply(pd.to_datetime).dt.dayofweek")
train['month_created']=train['date_account_created'].apply(pd.to_datetime).dt.month
train['day_created']=train['date_account_created'].apply(pd.to_datetime).dt.dayofweek
test['month_created']=test['date_account_created'].apply(pd.to_datetime).dt.month
test['day_created']=test['date_account_created'].apply(pd.to_datetime).dt.dayofweek

train.drop('date_account_created',axis=1,inplace=True)
test.drop('date_account_created',axis=1,inplace=True)


cat_vars=['first_affiliate_tracked','gender', 'signup_method', 'language',
       'affiliate_channel', 'affiliate_provider',
       'signup_app', 'first_device_type', 'first_browser','action','action_type','action_detail','device_type']


train['first_affiliate_tracked'].fillna('missed',inplace=True)
test['first_affiliate_tracked'].fillna('missed',inplace=True)
train['action_detail'].fillna('missed',inplace=True)
test['action_detail'].fillna('missed',inplace=True)
train['action'].fillna('missed',inplace=True)
test['action'].fillna('missed',inplace=True)
train['action_type'].fillna('missed',inplace=True)
test['action_type'].fillna('missed',inplace=True)
train['device_type'].fillna('missed',inplace=True)
test['device_type'].fillna('missed',inplace=True)

le=LabelEncoder()
for i in cat_vars:
    train[i]=le.fit_transform(train[i])
    test[i]=le.fit_transform(test[i])

y=le.fit_transform(target)
test.fillna(-1,inplace=True)
train.fillna(-1,inplace=True)
dtest=xgb.DMatrix(test,missing=-1)
dtrain=xgb.DMatrix(train,label=y,missing=-1)

param = {'bst:max_depth':9, 'bst:eta':0.7, 'silent':1, 'objective':'multi:softprob','num_class':12}
param["colsample_bytree"] = 0.30
param["subsample"] = 0.60
param["min_child_weight"] = 1
param['nthread'] = 4
plst=list(param.items())

#model= GradientBoostingClassifier(n_estimators=10)
#model.fit(train,y)

bst=xgb.train(plst,dtrain,num_boost_round=10)
bst_pred=bst.predict(dtest)

v1,v2,v3,v4,v5=[],[],[],[],[]
for i in range(len(bst_pred)):
    #v1,v2,v3=bst_pred[i][0],bst_pred[i][1],bst_pred[i][0]
    v1.append(np.argsort(bst_pred[i])[::-1][:5][0])
    v2.append(np.argsort(bst_pred[i])[::-1][:5][1])
    v3.append(np.argsort(bst_pred[i])[::-1][:5][2])
    v4.append(np.argsort(bst_pred[i])[::-1][:5][3])
    v5.append(np.argsort(bst_pred[i])[::-1][:5][4])

result_df=pd.DataFrame({'v1':v1,'v2':v2,'v3':v3,'v4':v4,'v5':v5})

result_df['id']=id_test
res_group=result_df.groupby(by='id')

vf=[]
for i in range(len(res_group)):
    vf.extend((int(res_group['v1'].mean()[i]),int(res_group['v2'].mean()[i]),int(res_group['v3'].mean()[i]),int(res_group['v4'].mean()[i]),int(res_group['v5'].mean()[i])))

ids=result_df.id.unique().tolist()

print (len(vf))
idss,cts=[],[]
for i in range(len(ids)):
    idx = ids[i]
    idss += [idx] * 5

cts = le.inverse_transform(vf)

sub = pd.DataFrame(np.column_stack((idss, cts)), columns=['id', 'country'])
sub.to_csv('sub.csv',index=False)
pd.read_csv('sub.csv')


"""
#get_ipython().run_cell_magic('time', '', "model=RandomForestClassifier(n_estimators=600,max_depth=7,criterion='gini',oob_score=True,max_features=None,warm_start=True)\nmodel.fit(train,y)")
model.oob_score_
imp_feats=pd.Series(model.feature_importances_,index=train.columns)
imp_feats.sort()
imp_feats.plot(kind='barh')
pred=model.predict(test)
pred_prob=model.predict_proba(test)
len(pred_prob),len(id_test)
get_ipython().run_cell_magic('time', '', 'bc=GradientBoostingClassifier(n_estimators=100,max_features=0.33)\nbc.fit(train,y)')
bs_prob=bc.predict_proba(test)
resps1=[]
for i in range(len(bst_pred)):
    max_p=[]
    for j in range(12):
        max_p.append((bst_pred[i][j]*pred_prob[i][j])*bs_prob[i][j])
    #print (max_p)
    max_p=[i/sum(max_p) for i in max_p]
    resps1.append(max_p)#.index(max(max_p))+1)
resps=[]
for i in range(len(bst_pred)):
    max_p=[]
    for j in range(12):
        max_p.append((bst_pred[i][j]+pred_prob[i][j])+bs_prob[i][j]/3)
    #print (max_p)
    resps.append(max_p)#.index(max(max_p))+1)
from sklearn.calibration import CalibratedClassifierCV
mp=CalibratedClassifierCV(base_estimator=model,cv='prefit')
ids,cts_resp=[],[]
for i in range(len(id_test)):
    idx = id_test[i]
    ids += [idx] * 3
    cts_resp += le.inverse_transform(np.argsort(resps[i])[::-1])[:3].tolist()
k=[i for i,j in zip(cts_resp1,cts_resp) if i!=j]
"""
