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
from scipy.optimize import fmin_powell

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

#x_train, x_test, y_train, y_test = train_test_split(train,y,test_size=0.33,random_state=42)

xdtest=xgb.DMatrix(test)
xdtrain=xgb.DMatrix(train,label=y)

params = {}
params["objective"] = "reg:linear"
params["eta"] = 0.3
params["min_child_weight"] = 120
params["subsample"] = 1
params["colsample_bytree"] = 0.67
params["silent"] = 0
params["max_depth"] = 4
params["eta"]=0.15
plst=list(params.items())

ets=[400]
for i in (ets):
    #params["eta"] = i
    plst=list(params.items())
    bst=xgb.train(plst,xdtrain,num_boost_round=i)

    offsets = np.array([0.1, -1, -2, -1, -0.8, 0.02, 0.8, 1])

    def eval_wrapper(yhat, y):
            y = np.array(y)
            y = y.astype(int)
            yhat = np.array(yhat)
            yhat = np.clip(np.round(yhat), np.min(y), np.max(y)).astype(int)
            return quadratic_weighted_kappa(yhat, y)

    def apply_offset(data, bin_offset, sv, scorer=eval_wrapper):
        # data has the format of pred=0, offset_pred=1, labels=2 in the first dim
        data[1, data[0].astype(int)==sv] = data[0, data[0].astype(int)==sv] + bin_offset
        score = scorer(data[1], data[2])
        return score

    train_preds=bst.predict(xdtrain,ntree_limit=bst.best_iteration)
    test_preds=bst.predict(xdtest,ntree_limit=bst.best_iteration)
    test_preds = np.clip(test_preds, -0.99, 8.99)
    train_preds = np.clip(train_preds, -0.99, 8.99)

    data = np.vstack((train_preds, train_preds, y))
    for j in range(8):
            data[1, data[0].astype(int)==j] = data[0, data[0].astype(int)==j] + offsets[j]
    for j in range(8):
            train_offset = lambda x: -apply_offset(data, x, j)
            offsets[j] = fmin_powell(train_offset, offsets[j])

    def apply_offset(data, bin_offset, sv, scorer=eval_wrapper):
            # data has the format of pred=0, offset_pred=1, labels=2 in the first dim
            data[1, data[0].astype(int)==sv] = data[0, data[0].astype(int)==sv] + bin_offset
            score = scorer(data[1], data[2])
            return score

    for j in range(8):
            train_offset = lambda x: -apply_offset(data, x, j)
            offsets[j] = fmin_powell(train_offset, offsets[j])

    k=pd.Series([0]*(len(test)))

    data = np.vstack((test_preds, test_preds, k))
    for j in range(8):
            data[1, data[0].astype(int)==j] = data[0, data[0].astype(int)==j] + offsets[j]

    resps = np.round(np.clip(data[1], 1, 8)).astype(int)
    #print (i,eval_wrapper(resps,y_test))
    sub = pd.DataFrame(np.column_stack((ids, resps)), columns=['Id', 'Response'])
    sub.to_csv('sub.csv',index=False)
    pd.read_csv('sub.csv')
