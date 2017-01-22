num_train = df_train.shape[0]

df_train = df_all.iloc[:num_train]
df_test = df_all.iloc[num_train:]
id_test = df_test['id']
y_train = df_train['relevance'].values
#X_train = df_train.drop(['id','relevance'],axis=1).values
#X_test = df_test.drop(['id','relevance'],axis=1).values

df_test.drop('relevance',axis=1,inplace=True)
df_test.drop('id',axis=1,inplace=True)
df_train.drop('relevance',axis=1,inplace=True)
df_train.drop('id',axis=1,inplace=True)


params = {}
params["objective"] ="reg:linear"
params["min_child_weight"] = 120
params["subsample"] = 1
params["colsample_bytree"] = .75
params["silent"] = 0
params["max_depth"] = 4
params["eta"]=0.15
params["gamma"]=1
plst=list(params.items())

x_train, x_test, y_train, y_test = train_test_split(df_train,y_train,test_size=0.33,random_state=0)
for i in [0.001,0.01,0.1,0.3,0.7,0.99,1.5,1.9]:
	params["eta"]=i
	plst=list(params.items())
	xdtest=xgb.DMatrix(x_test,y_test)
	xdtrain=xgb.DMatrix(x_train,label=y_train)
	bst=xgb.train(plst,xdtrain,num_boost_round=400)
	preds=bst.predict(xdtest)
	kk=pd.Series(y_train).unique()
	out=[]
	for i in preds:
	    closest= min(kk, key=lambda x:abs(x-i))
	    out.append(closest)

	print (mean_squared_error(out, y_test)**0.5,mean_squared_error(preds, y_test)**0.5)
