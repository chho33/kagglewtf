import os
os.environ['JOBLIB_TEMP_FOLDER'] = '/tmp'
from trend_util import *
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import pandas as pd
from mlxtend.classifier import StackingClassifier
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.model_selection import StratifiedShuffleSplit,ShuffleSplit
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform, reciprocal as sp_rec

a = time.time()
oversamp = False
n_splits = 20
n_jobs = -1
n_iter_search = 80
cv = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.33, random_state=42)
cv = ShuffleSplit(n_splits=n_splits, test_size=0.33, random_state=42)
version = 7
x,y,test = get_data(version)
train_ids = x.index
test_ids = test.index

#fillna
field = 'QueryTsInterval'
cols = []
for col in x.columns:
    if field in col:
        coll = col.lower()
        if ('std' in coll or 'skew' in coll or 'mad' in coll) and 'range' not in coll:
            cols.append(col)
x[cols] = x[cols].fillna(0)
test[cols] = test[cols].fillna(0)
x=x.fillna(-1)
test=test.fillna(-1)

gmm_file = 'trend_best_gmm_1520501668.6818335.pkl'
x_all = pd.concat([x,test],axis=0)
#x_all = x_all.fillna(-1)
#best_gmm = best_gmm_cluster(x_all)
#joblib.dump(best_gmm,'export/trend_best_gmm.pkl')
#best_gmm = joblib.load('export/'+gmm_file)
#x_all = best_gmm.predict_proba(x_all)
x = x_all[:x.shape[0]]
test = x_all[x.shape[0]:]
print(x.shape,y.shape)
if oversamp:
    x,y = SMOTE().fit_sample(x, y)
    print(x.shape,y.shape)

use_gpu = False
params = {}
params['scale_pos_weight'] = 1
params['learning_rate'] = 0.3
params['reg_lambda'] = 1
cparams = copy.deepcopy(params)
params['reg_alpha'] = 0.2
params['colsample_bytree'] = 0.75
xparams = copy.deepcopy(params)
lparams = copy.deepcopy(params)

xparams['gamma'] = 0.2
lparams['feature_fraction'] = 0.6
lparams['bagging_fraction'] = 0.6
lparams['bagging_freq'] = 6
#lparams['early_stopping_round'] = 20
cparams['n_estimators'] = 120
cparams['max_depth'] = 3
#cparams['l2_leaf_reg'] = 0.001

if use_gpu:
    xparams['tree_method'] = 'gpu_hist'
    xparams['predictor'] = 'gpu_predictor'
    xparams['objective'] = 'gpu:binary:logistic'
    n_jobs = 1
else:
    xparams['objective'] = 'binary:logistic'
    lparams['objective'] = 'binary'

xgbm = XGBClassifier(**xparams)
lgbm = LGBMClassifier(**lparams)
cgbm = CatBoostClassifier(**cparams)
rdf = RandomForestClassifier()
classifiers = [rdf,xgbm,lgbm]
classifiers = [xgbm,lgbm,cgbm]
classifiers = [xgbm,lgbm]
lr = LogisticRegression(C=0.1)
grid = StackingClassifier(classifiers=classifiers,use_probas=True,average_probas=False,meta_classifier=lr)

n_estimators = [100,300]
n_estimators = sp_randint(250,500)
max_depth = [2,3]
subsample = [0.5,0.7]
subsample = sp_rec(0.3,0.8)
C = [0.01,0.2]
C = sp_rec(0.01,0.2)
learning_rate = [0.1,0.4]
learning_rate = sp_rec(0.1,0.4)
reg_lambda = [2,6]
reg_lambda = sp_randint(2,10)
reg_alpha = [0.1,0.3]
reg_alpha = sp_rec(0.1,0.8)
gamma = sp_rec(0.1,0.8)
feature_fraction = sp_rec(0.3,0.8)
bagging_fraction = sp_rec(0.3,0.8)
bagging_freq = sp_randint(3,8)
params = {'xgbclassifier__n_estimators': n_estimators,
          'xgbclassifier__max_depth': max_depth,
          'xgbclassifier__subsample': subsample,
          'xgbclassifier__learning_rate': learning_rate,
          'xgbclassifier__reg_lambda': reg_lambda,
          'xgbclassifier__reg_alpha': reg_alpha,
          'xgbclassifier__gamma':gamma,
          'lgbmclassifier__n_estimators':n_estimators,
          'lgbmclassifier__max_depth': max_depth,
          'lgbmclassifier__subsample': subsample,
          'lgbmclassifier__learning_rate': learning_rate,
          'lgbmclassifier__reg_lambda': reg_lambda,
          'lgbmclassifier__reg_alpha': reg_alpha,
          'lgbmclassifier__feature_fraction': feature_fraction,
          'lgbmclassifier__bagging_fraction': bagging_fraction,
          'lgbmclassifier__bagging_freq':bagging_freq,
          #'catboostclassifier__n_estimators':n_estimators,
          #'catboostclassifier__max_depth': max_depth,
          #'randomforestclassifier__n_estimators':[100],
          #'randomforestclassifier__max_depth': [3],
          'meta-logisticregression__C': C
          }

fit_params={"early_stopping_rounds":100}

grid = RandomizedSearchCV(grid,n_jobs=n_jobs, param_distributions=params,
                          verbose=3, n_iter=n_iter_search,cv=cv)

print('fitting')
grid.fit(x,y)

joblib.dump(grid, 'export/trend_model_random_%s.pkl'%version)


predicted = grid.predict_proba(x)
predicted = list(map(lambda x:x[1],predicted))
print('trian roc: ',roc_auc_score(y,predicted))
print('val roc: ',grid.best_score_)
print('best params: ',grid.best_params_)
if oversamp:
    predicted = pd.Series(grid.predict_proba(test.as_matrix())[:,1])
else:
    predicted = pd.Series(grid.predict_proba(test)[:,1])
predicted.index = test_ids
predicted.to_csv('export/trend_predict_random_%s_%s.csv'%(version,int(time.time())))
print('cost time: ',time.time()-a)
