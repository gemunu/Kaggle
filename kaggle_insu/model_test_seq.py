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
from scipy.optimize import fmin
import random
from scpy.stats import poisson

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
#meds=['Medical_History_1','Medical_History_10','Medical_History_24','Medical_History_32','Medical_History_15']
meds=['Medical_History_1','Medical_History_10','Medical_History_24','Medical_History_32','Medical_History_15','Family_Hist_5','Family_Hist_3']


for i in meds:
    temp1.remove(i)
print (temp1,meds)

"""for i in temp1:
    men=train[i].mean()
    test[i].fillna(men,inplace=True)
    train[i].fillna(men,inplace=True)"""

for i in temp1:
    mu,sigma=train[i].mean(),train[i].std()
    n=len(train)-train[i].count()
    kk=train[train[i].isnull()].index
    s=pd.Series(poisson.rvs(mu,size=n),index=kk)
    train[i].fillna(s,inplace=True)

    mu,sigma=test[i].mean(),test[i].std()
    n=len(test)-test[i].count()
    kk=test[test[i].isnull()].index
    s=pd.Series(poisson.rvs(mu,n),index=kk)
    test[i].fillna(s,inplace=True)

for i in meds:
    mo=test[i].mode()[0]
    print (i,mo)
    test[i].fillna(-1,inplace=True)
    train[i].fillna(-1,inplace=True)


y=train.pop('Response')
ids=test.pop('Id')
train.drop('Id',axis=1,inplace=True)

"""train['Product_Info_2_char'] = train.Product_Info_2.str[0]
train['Product_Info_2_num'] = train.Product_Info_2.str[1]

# factorize categorical variables
train['Product_Info_2'] = pd.factorize(train['Product_Info_2'])[0]
train['Product_Info_2_char'] = pd.factorize(train['Product_Info_2_char'])[0]
train['Product_Info_2_num'] = pd.factorize(train['Product_Info_2_num'])[0]

test['Product_Info_2_char'] = test.Product_Info_2.str[0]
test['Product_Info_2_num'] = test.Product_Info_2.str[1]

# factorize categorical variables
test['Product_Info_2'] = pd.factorize(test['Product_Info_2'])[0]
test['Product_Info_2_char'] = pd.factorize(test['Product_Info_2_char'])[0]
test['Product_Info_2_num'] = pd.factorize(test['Product_Info_2_num'])[0]
"""
le=LabelEncoder()
train['Product_Info_2']=le.fit_transform(train['Product_Info_2'])
test['Product_Info_2']=le.fit_transform(test['Product_Info_2'])

x_train, x_test, y_train, y_test = train_test_split(train,y,test_size=0.33,random_state=0)

print (x_train,x_test,y_train,y_test)

x_test['BMI_Age'] = x_test['BMI'] * x_test['Ins_Age']
x_train['BMI_Age'] = x_train['BMI'] * x_train['Ins_Age']
x_test['var1']=x_test['BMI'] **0.75
x_train['var1']=x_train['BMI'] **0.75
x_train['var2']=x_train['Ins_Age']**8.5
x_test['var2']=x_test['Ins_Age']**8.5

#x_test['ht_wt'] = x_test['Ins_Age'] * x_test['Employment_Info_1']*x_test['BMI_Age']
#x_train['ht_wt'] = x_train['Ins_Age'] * x_train['Employment_Info_1']*x_train['BMI_Age']
#x_test['Ht_Age'] = x_test['Medical_History_32'] * x_test['BMI']
#x_train['Ht_Age'] = x_train['Medical_History_32'] * x_train['BMI']

med_keyword_columns_x =x_train.columns[x_train.columns.str.startswith('Medical_Keyword_')]
x_train['Med_Keywords_Count'] = x_train[med_keyword_columns_x].sum(axis=1)
x_test['Med_Keywords_Count'] = x_test[med_keyword_columns_x].sum(axis=1)

#med_keyword_columns_x =x_train.columns[x_train.columns.str.startswith('Medical_History_')]
#x_train['Med_hist_Count'] = x_train[med_keyword_columns_x].product(axis=1)
#x_test['Med_hist_Count'] = x_test[med_keyword_columns_x].product(axis=1)
#df_med=pd.DataFrame(x_train[med_keyword_columns_x])
#x_train['med_std']=df_med.std(axis=1)
#x_test['med_std']=df_medt.std(axis=1)

"""
for i in med_keyword_columns_x:
    test.drop(i,axis=1,inplace=True)
    train.drop(i,axis=1,inplace=True)

to_drop=['Family_Hist_5','Medical_History_10','Medical_History_24','Medical_History_32','Medical_History_15']
for i in to_drop:
        x_test.drop(i,axis=1,inplace=True)
        x_train.drop(i,axis=1,inplace=True)"""

xdtest=xgb.DMatrix(x_test,y_test,missing=-1)
xdtrain=xgb.DMatrix(x_train,label=y_train,missing=-1)

params = {}
params["objective"] = "count:poisson" #"reg:linear"
params["min_child_weight"] = 120
params["subsample"] = 1
params["colsample_bytree"] = .75
params["silent"] = 0
params["max_depth"] = 4
params["eta"]=0.15
params["gamma"]=1
plst=list(params.items())
#0.6510793788705647
best_paras=[]
ets=[0.150] #"reg:linear", ]
nrounds=[400]
trp=[]
tep=[]
for etam in (ets):
    for rawuma in nrounds:
        params["eta"] =etam
        plst=list(params.items())
        bst=xgb.train(plst,xdtrain,num_boost_round=rawuma)

        offsets = np.array([0.1, -1, -0.1, -0.1, -0.8, 0.02, 0.8, 1])
        #offsets = np.array([0.1, -1, -2, -1, -0.9, 0.05, 0., 1])

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

        data1 = np.vstack((train_preds, train_preds, y_train))
        for j in range(8):
                data1[1, data1[0].astype(int)==j] = data1[0, data1[0].astype(int)==j] + offsets[j]
        for j in range(8):
                train_offset = lambda x: -apply_offset(data1, x, j)
                offsets[j] = fmin(train_offset, offsets[j])

        ll=data1[0].round().astype(int)
        trp.append(ll)
        k=pd.Series([0]*(len(x_test)))

        data = np.vstack((test_preds, test_preds, k))
        for j in range(8):
                data[1, data[0].astype(int)==j] = data[0, data[0].astype(int)==j] + offsets[j]

        resps = np.round(np.clip(data[1], 1, 8)).astype(int)
        tep.append(resps)
        print (etam,eval_wrapper(resps,y_test))
        best_paras.append((etam,rawuma,eval_wrapper(resps,y_test)))
