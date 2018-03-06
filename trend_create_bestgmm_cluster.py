from trend_util import *
import time
file_path = '/data/examples/trend/data/'
rawdata_path = file_path + 'query_log/'
version = 5
x,y,test = get_data(version)
train_ids = x.index
test_ids = test.index
x_all = pd.concat([x,test],axis=0)
x_all = x_all.fillna(-1)
x_all_norm = normit(x_all)
x_norm = x_all_norm[:x.shape[0]]
test_norm = x_all_norm[x.shape[0]:]
best_gmm = best_gmm_cluster(x_all_norm,range(100,102))
joblib.dump(best_gmm,'export/trend_best_gmm_%s.pkl'%time.time())
