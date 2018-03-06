import os
os.environ['JOBLIB_TEMP_FOLDER'] = '/tmp'
from trend_util import * 
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import pandas as pd
from mlxtend.classifier import StackingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
x,y,test = get_data()
train_ids = x.index
test_ids = test.index
#x_all = pd.concat([x,test],axis=0) 
#x_all = x_all.fillna(-1)
#best_gmm = best_gmm_cluster(x_all)
#joblib.dump(best_gmm,'export/trend_best_gmm.pkl')
#best_gmm = joblib.load('trend_best_gmm.pkl')
#x_all = best_gmm.predict_proba(x_all)
#x = x_all[:x.shape[0]]
#test = x_all[x.shape[0]:]
use_gpu = False
n_job = 8
params = {}
params['scale_pos_weight'] = 8.31
params['learning_rate'] = 0.3
params['reg_lambda'] = 1
cparams = copy.deepcopy(params) 
params['reg_alpha'] = 0.15
params['colsample_bytree'] = 0.85
xparams = copy.deepcopy(params)
lparams = copy.deepcopy(params)

xparams['gamma'] = 0.3
lparams['feature_fraction'] = 0.9
lparams['bagging_fraction'] = 0.95
lparams['bagging_freq'] = 5
lparams['early_stopping_round'] = 20
cparams['n_estimators'] = 200
cparams['max_depth'] = 3
cparams['n_estimators'] = 20
cparams['max_depth'] = 1500 

if use_gpu:
    xparams['tree_method'] = 'gpu_hist'
    xparams['predictor'] = 'gpu_predictor'
    xparams['objective'] = 'gpu:binary:logistic'
    n_job = 1
else:
    xparams['objective'] = 'binary:logistic'
    lparams['objective'] = 'binary'

xgbm = XGBClassifier(**xparams)
lgbm = LGBMClassifier(**lparams)
cgbm = CatBoostClassifier(**cparams)
rdf = RandomForestClassifier()
classifiers = [rdf,xgbm,lgbm]
classifiers = [xgbm,lgbm,cgbm]
lr = LogisticRegression(C=0.1)
grid = StackingClassifier(classifiers=classifiers,use_probas=True,average_probas=False,meta_classifier=lr)

n_estimators = [1500,20000]
max_depth = [3]
subsample = [0.8,1]
C = [0.1]
params = {'xgbclassifier__n_estimators': n_estimators,
          'xgbclassifier__max_depth': max_depth,
          'xgbclassifier__subsample': subsample,
          'lgbmclassifier__n_estimators':n_estimators,
          'lgbmclassifier__max_depth': max_depth,
          'lgbmclassifier__subsample': subsample,
          #'catboostclassifier__n_estimators':n_estimators,
          #'catboostclassifier__max_depth': max_depth,
          #'randomforestclassifier__n_estimators':[100],
          #'randomforestclassifier__max_depth': [3],
          'meta-logisticregression__C': C
          }

grid = GridSearchCV(estimator=grid,
                    param_grid=params,
                    cv=3,
                    refit=True,
                    verbose=3,
                    n_jobs=n_job,
                    early_stopping_rounds=100,
                    scoring='roc_auc')

print('fitting')
grid.fit(x,y)

joblib.dump(grid, 'export/trend_model.pkl') 


predicted = grid.predict_proba(x)
predicted = list(map(lambda x:x[1],predicted))
print('trian roc: ',roc_auc_score(y,predicted))

predicted = pd.Series(grid.predict_proba(test)[:,1])
predicted.index = test_ids
predicted.to_csv('export/trend_predict_test.csv')
