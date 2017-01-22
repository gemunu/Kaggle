# coding: utf-8
get_ipython().magic(u'run preprocess.py')
test
train
test
params["eta"]=0.45
x_train, x_test, y_train, y_test = train_test_split(train,y,test_size=0.33,random_state=42)
xdtest=xgb.DMatrix(x_test)
xdtrain=xgb.DMatrix(x_train,label=y_train)
y_test
get_ipython().magic(u'ls ')
def eval_wrapper(yhat, y):  
        y = np.array(y)
        y = y.astype(int)
        yhat = np.array(yhat)
        yhat = np.clip(np.round(yhat), np.min(y), np.max(y)).astype(int)   
        return quadratic_weighted_kappa(yhat, y)
params = {}
params["objective"] = "reg:linear"     
params["eta"] = 0.3
params["min_child_weight"] = 240
params["subsample"] = 1
params["colsample_bytree"] = 0.67
params["silent"] = 1
params["max_depth"] = 3
plst = list(params.items())
bst=xgb.train(plst,xdtrain,num_boost_round=200)
bst_p=bst.predict(dtest).astype(int)
bst_p
eval_wrapper(bst_pred,y_test)
eval_wrapper(bst_p,y_test)
eval_wrapper(bst_p,y_test)
bst_p=bst.predict(dtest).astype(int)
bst_p1
xdtest
xdtest()
bst_p=bst.predict(xdtest).astype(int)
bst_p
eval_wrapper(bst_p,y_test)
params["eta"]=0.45
params["max_depth"] = 4
params["max_depth"] = 4
plst=list(params.items())
plst
bst=xgb.train(plst,xdtrain,num_boost_round=200)
bst_p=bst.predict(xdtest).astype(int)
eval_wrapper(bst_p,y_test)
bst=xgb.train(plst,xdtrain,num_boost_round=90)
bst_p=bst.predict(xdtest).astype(int)
eval_wrapper(bst_p,y_test)
bst=xgb.train(plst,xdtrain,num_boost_round=350)
bst_p=bst.predict(xdtest).astype(int)
eval_wrapper(bst_p,y_test)
bst
bst()
bst.feature_names
bst_best=bst.copy
bst_best
bst_best=bst.copy()
bst_best.get_fscore
bst_best.get_fscor
bst.get_fscore()
eval_wrapper(bst_p,y_test)
plst
bst=xgb.train(plst,dtrain,num_boost_round=350)
dtrain
dtrain=xgb.DMatrix(train,label=y)
dtrain
stest
bst_p=bst.predict(dtest).astype(int)
y
bst_pt=bst.predict(dtrain).astype(int)
eval_wrapper(bst_pt,y)
ids
resps=bst_p
resps
bst_p=bst.predict(dtest).astype(int)
resps=bst_p
sub = pd.DataFrame(np.column_stack((ids, resps)), columns=['Id', 'Response'])
sub.to_csv('sub.csv',index=False)
pd.read_csv('sub.csv')
get_ipython().magic(u'pwd ')
resps
bst_p
bst_p.max()
bst_p.min()
plst
params["objective"] = 'multi:softmax'
plst=list(params.items())
bst=xgb.train(plst,dtrain,num_boost_round=350)
params["num_class"] = 9
plst=list(params.items())
bst=xgb.train(plst,dtrain,num_boost_round=350)
bst_p=bst.predict(dtest).astype(int)
resps=bst_p
resps
sub = pd.DataFrame(np.column_stack((ids, resps)), columns=['Id', 'Response'])
sub.to_csv('sub.csv',index=False)
pd.read_csv('sub.csv')
bst_pt=bst.predict(dtrain).astype(int)
eval_wrapper(bst_pt,y)
bst=xgb.train(plst,xdtrain,num_boost_round=350)
bst_p=bst.predict(xdtest).astype(int)
bst_p
eval_wrapper(bst_p,y_test)
params["silent"] = 0
params["objective"] = "reg:linear"     
plst=list(params.items())
bst=xgb.train(plst,xdtrain,num_boost_round=350)
get_ipython().magic(u'pinfo params.clear')
params = {}
params["objective"] = "reg:linear"     
params["eta"] = 0.3
params["min_child_weight"] = 240
params["subsample"] = 1
params["colsample_bytree"] = 0.67
params["silent"] = 1
params["max_depth"] = 3
plst = list(params.items())
params["silent"] = 0
params["max_depth"] = 4
params["eta"]=0.45
plst=list(params.items())
bst=xgb.train(plst,xdtrain,num_boost_round=350)
train_preds=bst.predict(xdtrain,ntree_limit=bst.best_iteration)
test_preds=bst.predict(xdtest,ntree_limit=bst.best_iteration)
train_preds = np.clip(train_preds, -0.99, 8.99)
train_preds
test_preds = np.clip(test_preds, -0.99, 8.99)
offsets = np.array([0.1, -1, -2, -1, -0.8, 0.02, 0.8, 1])
data = np.vstack((train_preds, train_preds, y))
y
len(train_preds)
train_preds=bst.predict(dtrain,ntree_limit=bst.best_iteration)
test_preds=bst.predict(dtest,ntree_limit=bst.best_iteration)
test_preds = np.clip(test_preds, -0.99, 8.99)
train_preds = np.clip(train_preds, -0.99, 8.99)
data = np.vstack((train_preds, train_preds, y))
data
for j in range(8):
        data[1, data[0].astype(int)==j] = data[0, data[0].astype(int)==j] + offsets[j]
    
for j in range(num_classes):
        train_offset = lambda x: -apply_offset(data, x, j)
        offsets[j] = fmin_powell(train_offset, offsets[j])
    
for j in range(8):
        train_offset = lambda x: -apply_offset(data, x, j)
        offsets[j] = fmin_powell(train_offset, offsets[j])
    
from scipy.optimize import fmin_powell
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
    
len(test_preds)
type(y)
k=np.array[0]*19765
k=[0]*19765
k
k1=pd.Series(k)
k1
data = np.vstack((test_preds, test_preds, k))
for j in range(8):
        data[1, data[0].astype(int)==j] = data[0, data[0].astype(int)==j] + offsets[j]
    
final_test_preds = np.round(np.clip(data[1], 1, 8)).astype(int)
final_test_preds
final_test_preds.max()
final_test_preds.min
final_test_preds.min()
resps=final_test_preds
resps
sub = pd.DataFrame(np.column_stack((ids, resps)), columns=['Id', 'Response'])
sub.to_csv('sub.csv',index=False)
pd.read_csv('sub.csv')
get_ipython().magic(u'save session-1')
get_ipython().magic(u'save session_1')
get_ipython().magic(u'save session_1.py')
get_ipython().magic(u'save')
get_ipython().magic(u'save session')
get_ipython().magic(u"save 'session'")
