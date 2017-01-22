import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
import snowballstemmer
import xgboost as xgb
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error


df_train = pd.read_csv('train.csv', encoding="ISO-8859-1")
df_test = pd.read_csv('test.csv', encoding="ISO-8859-1")
df_pro_desc = pd.read_csv('product_descriptions.csv')

stemmer = snowballstemmer.EnglishStemmer()

def str_stemmer(s):
	return " ".join([stemmer.stemWord(wrd) for wrd in s.lower().split()])

def str_common_word(str1, str2):
	return sum(int(str2.find(word)>=0) for word in str1.split())

df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)
df_all = pd.merge(df_all, df_pro_desc, how='left', on='product_uid')


df_all['search_term'] = df_all['search_term'].map(lambda x:str_stemmer(x))
df_all['product_title'] = df_all['product_title'].map(lambda x:str_stemmer(x))
df_all['product_description'] = df_all['product_description'].map(lambda x:str_stemmer(x))
df_all['len_of_query'] = df_all['search_term'].map(lambda x:len(x.split())).astype(np.int64)
df_all['product_info'] = df_all['search_term']+"\t"+df_all['product_title']+"\t"+df_all['product_description']
df_all['word_in_title'] = df_all['product_info'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[1]))
df_all['word_in_description'] = df_all['product_info'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[2]))
df_all = df_all.drop(['search_term','product_title','product_description','product_info'],axis=1)

df_all['newc'] = df_all['len_of_query']*df_all['word_in_description']
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
kk=pd.Series(y_train).unique()

params = {}
params["objective"] ="reg:linear"#"count:poisson"#
params["min_child_weight"] = 120
params["subsample"] = 1
params["colsample_bytree"] = .75
params["silent"] = 0
params["max_depth"] = 4
params["eta"]=0.07
params["gamma"]=0
plst=list(params.items())

x_train, x_test, y_train, y_test = train_test_split(df_train,y_train,test_size=0.33,random_state=0)

xdtest=xgb.DMatrix(x_test,y_test)
xdtrain=xgb.DMatrix(x_train,label=y_train)
bst=xgb.train(plst,xdtrain,num_boost_round=350)
preds=bst.predict(xdtest)
out=[]
for i in preds:
	closest= min(kk, key=lambda x:abs(x-i))
	out.append(closest)

print (mean_squared_error(out, y_test)**0.5,mean_squared_error(preds, y_test)**0.5)
