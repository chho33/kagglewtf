import pandas as pd
from util import *
import re
from datetime import datetime
import pickle
import os
import glob
logging.basicConfig(filename='log/trend_util.log',level=logging.DEBUG)
file_path = os.environ['file_path'] if 'file_path' in os.environ else '/data/examples/trend/data/' 
rawdata_path = os.environ['rawdata_path'] if 'rawdata_path' in os.environ else '/data/examples/trend/data/query_log/' 
export_file_path = os.environ['export_file_path'] if 'export_file_path' in os.environ else 'export/' 
etl_file_path = os.environ['etl_file_path'] if 'etl_file_path' in os.environ else 'data/trend/' 
files = glob.glob(rawdata_path+'/*.csv')
list.sort(files)
ns = len(files)

############ etl ############
def check_file_empty(path,header=None):
    try:
        df = pd.read_csv(path,header=header)
    except pd.errors.EmptyDataError:
        return False
    return df

def norm_it(df):
    df = df.div(df.sum(axis=1), axis=0)
    return df

def percentile(n):
    def percentile_(x):
        return np.percentile(x, n)
    percentile_.__name__ = 'percentile_%s' % n
    return percentile_

flags = ['size','sum','mean','std','max','min','median','skew','mad',percentile(25),percentile(75)]
flags = ['mean','std','max','min','median','skew','mad',percentile(25),percentile(75)]
def get_aggr(df,flags=flags,deep=False):
    if deep: return get_deep_aggr(df=df,flags=flags)
    df = df.groupby(df.index.get_level_values(0)).agg(flags)
    df = df.fillna(0)
    return df

def get_deep_aggr(df,flags=flags):
    levels = df.index.names
    if len(levels) > 1:
        df = df.groupby(level=levels).agg('sum')
        df = df.groupby(level='FileID').agg(flags)
    elif len(levels) == 1:
        df = df.groupby(df.index.get_level_values(0)).agg(flags)
    df = df.fillna(0)
    return df

def df_add(df,df_past,add_by='int'):
    intersect_ix = df.index.intersection(df_past.index)
    diff_ix = df.index.difference(df_past.index)
    diff_ix_past = df_past.index.difference(df.index)
    if add_by == 'int':
        df0 = df.ix[intersect_ix].add(df_past.ix[intersect_ix])
    elif add_by == 'corpus':
        df0 = df.ix[intersect_ix] + ' ' + df_past.ix[intersect_ix]
    df1 = df.ix[diff_ix]
    df2 = df_past.ix[diff_ix_past]
    df = pd.concat([df0,df1,df2],axis=0)
    return df
    
def clean_df(df):
    df['ProductID'] = df['ProductID'].astype('str')
    df = df.replace(['055649'],['55649'])
    return df

def append_set(x):
    new_x = x.dropna()
    if new_x.shape[0] == 0:
        return
    elif new_x.shape[0] == 1:
        new_x = new_x.values[0]
    else:
        new_x = np.append(np.array(list(new_x[0])),np.array(list(new_x[1])))
    new_x = set(new_x)
    return new_x

def outer_set(x):
    new_x = x.dropna()
    if new_x.shape[0] == 0:
        return
    elif new_x.shape[0] == 1:
        new_x = new_x.values[0]
    else:
        set_tmp0 = set(new_x[0])
        set_tmp = set(new_x[1])
        new_x = set_tmp0 - set_tmp 
        set_tmp = set_tmp - set_tmp0
        new_x.update(set_tmp)
    new_x = set(new_x)
    return new_x

def intersect_set(x):
    new_x = x.dropna()
    if new_x.shape[0] == 0 or new_x.shape[0] == 1:
        return
    else:
        new_x = pd.Index(list(new_x[0])).intersection(pd.Index(list(new_x[1])))
    new_x = set(new_x)
    return new_x

def get_train_test_prod_cus_set(df,df_past=None,return_fields=['prod_cus']):
    if 'FileID' in df.columns:
        df = df.set_index('FileID')
    test_ids = pd.read_csv(file_path+'testing-set.csv',header=None)
    test_ids = pd.Index(test_ids[0])
    train_ids = pd.read_csv(file_path+'training-set.csv',header=None)
    train_ids = pd.Index(train_ids[0])
    df_train = df.ix[train_ids].dropna()
    df_test = df.ix[test_ids].dropna()
    #uniq product
    if 'product' in return_fields:
        train_prod_set = set(df_train['ProductID'])
        test_prod_set = set(df_test['ProductID'])
    #uniq customer
    if 'customer' in return_fields:
        train_cus_set = set(df_train['CustomerID'])
        test_cus_set = set(df_test['CustomerID'])
    #uniq prod cross customer
    if 'prod_cus' in return_fields:
        train_prod_cus_set = df_train.groupby('ProductID')['CustomerID'].unique()
        test_prod_cus_set = df_test.groupby('ProductID')['CustomerID'].unique()
    if df_past is not None:
        #merge prod
        if 'product' in return_fields:
            train_prod_set.update(df_past['train_prod_set'])
            test_prod_set.update(df_past['test_prod_set'])

        #merge customer
        if 'customer' in return_fields:
            train_cus_set.update(df_past['train_cus_set'])
            test_cus_set.update(df_past['test_cus_set'])

        #merge prod cross customer
        if 'prod_cus' in return_fields:
            train_prod_cus_set = df_train.groupby('ProductID')['CustomerID'].unique()
            train_prod_cus_set = pd.concat([train_prod_cus_set,df_past['train_prod_cus_set']],axis=1)
            train_prod_cus_set.columns = [0,1]
            train_prod_cus_set = train_prod_cus_set.apply(append_set,axis=1)

            test_prod_cus_set = df_test.groupby('ProductID')['CustomerID'].unique()
            test_prod_cus_set = pd.concat([test_prod_cus_set,df_past['test_prod_cus_set']],axis=1)
            test_prod_cus_set.columns = [0,1]
            test_prod_cus_set = test_prod_cus_set.apply(append_set,axis=1)
    data = {}
    if 'product' in return_fields:
        data['train_prod_set'] = train_prod_set
        data['test_prod_set'] = test_prod_set
    if 'customer' in return_fields:
        data['train_cus_set'] = train_cus_set
        data['test_cus_set'] = test_cus_set
    if 'prod_cus' in return_fields:
        data['train_prod_cus_set'] = train_prod_cus_set
        data['test_prod_cus_set'] = test_prod_cus_set

    return data
    return {'train_prod_set':train_prod_set,'test_prod_set':test_prod_set,
           'train_cus_set':train_cus_set,'test_cus_set':test_cus_set,
           'train_prod_cus_set':train_prod_cus_set,'test_prod_cus_set':test_prod_cus_set}

def get_prod_cus_intersect(df,dump=False,return_fields=['prod_cus'],reverse=True):
    #merge cus
    if 'customer' in return_fields:
        cus_set = df['train_cus_set']
        test_cus_set = df['test_cus_set']
        if reverse:
            cus_set_tmp = cus_set - test_cus_set  
            cus_set = test_cus_set - cus_set 
            cus_set.update(cus_set_tmp) 
            if dump:
                pickle.dump(cus_set, 'export/trend_uncommon_customers.pkl')
        else:
            cus_set = pd.Index(cus_set)
            test_cus_set = pd.Index(test_cus_set)
            cus_set = cus_set.intersection(test_cus_set)
            cus_set = set(cus_set)
            if dump:
                pickle.dump(cus_set, 'export/trend_common_customers.pkl')
    #merge prod
    if 'product' in return_fields:
        prod_set = df['train_prod_set']
        test_prod_set = df['test_prod_set']
        if reverse:
            prod_set_tmp = prod_set - test_prod_set  
            prod_set = test_prod_set - prod_set 
            prod_set.update(prod_set_tmp) 
            if dump:
                pickle.dump(prod_set, 'export/trend_uncommon_products.pkl')
        else:
            prod_set = pd.Index(prod_set)
            test_prod_set = pd.Index(test_prod_set)
            prod_set = prod_set.intersection(test_prod_set)
            prod_set = set(prod_set)
            if dump:
                pickle.dump(prod_set, 'export/trend_common_products.pkl')
    #merge prod cross cus
    if 'prod_cus' in return_fields:
        train_prod_cus = df['train_prod_cus_set']
        test_prod_cus = df['test_prod_cus_set']
        df_prod_cus = pd.concat([train_prod_cus,test_prod_cus],join='inner',axis=1)
        df_prod_cus.columns = [0,1]
        if reverse:
            df_prod_cus = df_prod_cus.apply(outer_set,axis=1)
            if dump:
                pickle.dump(df_prod_cus, 'export/trend_uncommon_products_customers.pkl')
        else:
            df_prod_cus = df_prod_cus.apply(intersect_set,axis=1)
            if dump:
                pickle.dump(df_prod_cus, 'export/trend_common_products_customers.pkl')
    data = {}
    if 'product' in return_fields:
        data['product'] = prod_set
    if 'customer' in return_fields:
        data['customer'] = cus_set
    if 'prod_cus' in return_fields:
        data['prod_cus'] = df_prod_cus
    return data
    return {'customer':cus_set,'product':prod_set,'prod_cus':df_prod_cus}

def common_prod_cus_filter(df,df_prod_cus):
    prods = df_prod_cus.index
    if len(prods) == 0: return pd.DataFrame({None : []}) 
    if df.index.name == 'FileID': df = df.reset_index()
    df = df.set_index('ProductID')
    df = df.ix[prods]
    df = pd.DataFrame(df)
    dfs = []
    logging.info('======= common_prod_cus_filter ======')
    for i,prod in enumerate(prods):
        #get customers of certain product
        cus = pd.Index(df_prod_cus.ix[prod])
        #get certain product of df
        dft = (df.ix[prod]).dropna()
        if dft.empty: return False
        dft = pd.DataFrame(dft)
        if 'FileID' not in dft.columns:
            dft = dft.T
        dft = dft.reset_index()
        if 'index' not in dft.columns:
            dft = dft.rename({'index':'ProductID'},axis=1)
        logging.info('%s, %s: '%(i,prod))
        logging.info(dft.head(2))
        logging.info('----------------------')
        dft = dft.set_index('CustomerID')
        dft = (dft.ix[cus]).dropna()
        dft = pd.DataFrame(dft)
        if 'CustomerID' not in dft.columns:
            dft = dft.T
        dft = dft.reset_index()
        if 'index' not in dft.columns:
            dft = dft.rename({'index':'CustomerID'},axis=1)
        dfs.append(dft)
    df = pd.concat(dfs,axis=0)
    df['QueryTs'] = df['QueryTs'].astype(int)
    if df.index.name == 'FileID': df = df.reset_index()
    logging.info('======= common_prod_cus_filter end ======')
    return df

def common_filter(df,df_target,typ='tight'):
    # typ = [loose|tight|product|customer]
    # tight means uniq prod+cus ; loose otherwise
    if df.index.name == 'FileID': df = df.reset_index()
    if typ == 'tight':
        df = common_prod_cus_filter(df,df_target['prod_cus'])
    else:
        if typ == 'loose':
            target_set = df_target['product']
            ixs = pd.Index(target_set)
            df = df.set_index('ProductID')
            df = df.ix[ixs]
            dft = pd.DataFrame(dft)
            if 'ProductID' not in dft.columns:
                df = df.T
            df = df.reset_index()
            if 'index' not in dft.columns:
                df = df.rename({'index':'ProductID'},axis=1)
            target_set = df_target['customer']
            ixs = pd.Index(target_set)
            df = df.set_index('CustomerID')
            df = df.ix[ixs]
            dft = pd.DataFrame(dft)
            if 'CustomerID' not in dft.columns:
                df = df.T
            df = df.reset_index()
            if 'index' not in dft.columns:
                df = df.rename({'index':'CustomerID'},axis=1)
        else:
            if typ == 'product':
                target_set = df_target['product']
                ixs = pd.Index(target_set)
                df = df.set_index('ProductID')
            elif typ == 'customer':
                target_set = df_target['customer']
                ixs = pd.Index(target_set)
                df = df.set_index('CustomerID')
            if df.empty: return False
            df = df.ix[ixs]
            df = df.reset_index()
    return df

#FileID被各個ProductID開啟的次數的比例
def get_file_product_count_percentage(df,df_perc=None,normalize=False):
    dft = df[['FileID','ProductID']]
    dft = dft.assign(Count=1)
    dft = dft.groupby(['FileID','ProductID'],as_index = False).sum().pivot('FileID','ProductID').fillna(0)
    if normalize:
        dft = dft.div(dft.sum(axis=1), axis=0)
    cols = [col+'_count_percentage' for col in list(dft.columns.get_level_values(1))]
    dft.columns = cols
    if df_perc is not None:
        intersect_ix = dft.index.intersection(df_perc.index)
        diff_ix = dft.index.difference(df_perc.index)
        diff_ix_perc = df_perc.index.difference(dft.index)
        df0 = dft.ix[intersect_ix].add(df_perc.ix[intersect_ix])
        df1 = dft.ix[diff_ix]
        df2 = df_perc.ix[diff_ix_perc]
        dft = pd.concat([df0,df1,df2],axis=0)
        '''
        rows = set(df_perc.index) - set(dft.index)
        for row in rows:
            dft.ix[row] = 0
        rows = set(dft.index) - set(df_perc.index)
        for row in rows:
            df_perc.ix[row] = 0
        dft = dft.add(df_perc)
        '''
    return dft

def get_file_product_count(df,df_perc=None,normalize=False):
    dft = df[['FileID','ProductID']]
    dft = dft.assign(Count=1)
    dft = dft.groupby(['FileID','ProductID'],as_index = False).sum().pivot('FileID','ProductID').fillna(0)
    if normalize:
        dft = dft.div(dft.sum(axis=1), axis=0)
    cols = [col for col in list(dft.columns.get_level_values(1))]
    dft.columns = cols
    if df_perc is not None:
        dft = df_perc.append(dft)
    return dft

#每次被開啟的間隔時間的mean/std
def get_open_time(df,df_interval=None,max_timestamp=None):
    dft = df[['FileID','QueryTs']]
    if max_timestamp is not None:
        dft = dft.set_index('FileID')
        dft = pd.concat([pd.DataFrame(max_timestamp),dft],axis=0)
        dft = dft.reset_index()
    dft = dft.sort_values(by=['QueryTs'])
    dft['QueryTsInterval'] = dft.groupby('FileID')['QueryTs'].transform(pd.Series.diff)
    dft = dft.dropna()
    if df_interval is not None:
        dft = pd.concat([df_interval,dft],axis=0)
    max_timestamp = dft.groupby('FileID')['QueryTs'].max()
    return dft, max_timestamp

def get_open_time(df,df_past=None):
    df = df[['FileID','QueryTs']]
    if df_past is not None and df_past['max_timestamp'] is not None:
        df = df.set_index('FileID')
        df = pd.concat([pd.DataFrame(df_past['max_timestamp']),df],axis=0)
        df = df.reset_index()
    df = df.sort_values(by=['QueryTs'])
    df['QueryTsInterval'] = df.groupby('FileID')['QueryTs'].transform(pd.Series.diff)
    df = df.dropna()
    if df_past is not None and df_past['df'] is not None:
        df = pd.concat([df_past['df'],df],axis=0)
    max_timestamp = df.groupby('FileID')['QueryTs'].max()
    return {'df':df, 'max_timestamp':max_timestamp}

def get_field_open_time(df,df_past=None,field='CustomerID'):
    if field == 'ProductCustomerID':
        df = df.assign(ProductCustomerID=df['ProductID'].astype('str')+df['CustomerID'].astype('str'))
    df = df[['FileID',field,'QueryTs']]
    if df_past is not None and df_past['max_timestamp'] is not None:
        df = df.set_index('FileID')
        df = pd.concat([pd.DataFrame(df_past['max_timestamp']).reset_index().set_index('FileID'),df],axis=0)
        df = df.reset_index()
    df = df.sort_values(by=['QueryTs'])
    df['QueryTsInterval%s'%field] = df.groupby(['FileID',field])['QueryTs'].transform(pd.Series.diff)
    df = df.dropna()
    if df_past is not None and df_past['df'] is not None:
        df = pd.concat([df_past['df'],df],axis=0)
    max_timestamp = df.groupby(['FileID',field])['QueryTs'].max()
    return {'df':df, 'max_timestamp':max_timestamp}

flags = ['mean','std','count','size','nunique','max','min','median','sum','skew','mad']
flags = ['mean','std','count','nunique','max','min','median','sum','skew','mad',percentile(25),percentile(75)]
def get_open_time_aggr(df,field='',flags=flags):
    #q1 = df.groupby('FileID')['QueryTsInterval'].quantile(0.25)
    #q1.name = 'QueryTsIntervalQ1'
    #q3 = df.groupby('FileID')['QueryTsInterval'].quantile(0.75)
    #q3.name = 'QueryTsIntervalQ3'
    df = df.groupby('FileID')['QueryTsInterval%s'%field].agg(flags)
    cols = df.columns
    cols = [col.capitalize() for col in cols]
    cols = ['QueryTsInterval%s%s'%(col,field) for col in cols]
    df.columns = cols
    #df = pd.concat([df,q1,q3],axis=1)
    df = df.fillna(0)
    return df

#tfidf
def cal_tfidf(df,field):
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(df)
    tfidf = tfidf.toarray()
    fileIds=vectorizer.get_feature_names()
    #tfidf_sum = np.sum(tfidf,axis=0)
    tfidf_mean = np.mean(tfidf,axis=0)
    tfidf_median = np.median(tfidf,axis=0)
    tfidf_std = np.std(tfidf,axis=0)
    tfidf_min = np.amin(tfidf,axis=0)
    tfidf_max = np.amax(tfidf,axis=0)
    tfidf_ptp = np.ptp(tfidf,axis=0)
    tfidf_q1 = np.percentile(tfidf,25,axis=0)
    tfidf_q3 = np.percentile(tfidf,75,axis=0)
    idf = vectorizer.idf_
    df = pd.DataFrame({'FileID':fileIds,'%s_tfidf_mean'%field:tfidf_mean,'%s_tfidf_median'%field:tfidf_median,
                      '%s_tfidf_std'%field:tfidf_std,'%s_tfidf_min'%field:tfidf_min,'%s_tfidf_max'%field:tfidf_max,
                      '%s_tfidf_ptp'%field:tfidf_ptp,'%s_tfidf_q1'%field:tfidf_q1,'%s_tfidf_q3'%field:tfidf_q3,'%s_idf'%field:idf})

    df = df.set_index('FileID')
    return df

def cal_tfidf(df,field):
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(df)
    tfidf = tfidf.toarray()
    fileIds=vectorizer.get_feature_names()
    idf = vectorizer.idf_
    #tfidf_sum = np.sum(tfidf,axis=0)
    tfidf_mean = []
    tfidf_median = [] 
    tfidf_std = [] 
    tfidf_min = []
    tfidf_max = []
    tfidf_ptp = []
    tfidf_q1  = []
    tfidf_q3  = []
    for t in tfidf.T:
        tfidf_mean.append(np.mean(t))
        tfidf_median.append(np.median(t))
        tfidf_std.append(np.std(t))
        tfidf_min.append(np.amin(t))
        tfidf_max.append(np.amax(t))
        tfidf_ptp.append(np.ptp(t))
        tfidf_q1.append(np.percentile(t,25))
        tfidf_q3.append(np.percentile(tf,75))
    
    df = pd.DataFrame({'FileID':fileIds,'%s_tfidf_mean'%field:tfidf_mean,'%s_tfidf_median'%field:tfidf_median,
                      '%s_tfidf_std'%field:tfidf_std,'%s_tfidf_min'%field:tfidf_min,'%s_tfidf_max'%field:tfidf_max,
                      '%s_tfidf_ptp'%field:tfidf_ptp,'%s_tfidf_q1'%field:tfidf_q1,'%s_tfidf_q3'%field:tfidf_q3,'%s_idf'%field:idf})

    df = df.set_index('FileID')
    return df

def get_corpus(df,df_past=None,field='CustomerID'):
    df = df.groupby(field).apply(get_str_FileID)
    if df_past is not None:
        intersect_ix = df.index.intersection(df_past.index)
        diff_ix = df.index.difference(df_past.index)
        diff_ix_past = df_past.index.difference(df.index)
        df0 = df.ix[intersect_ix] + ' ' + df_past.ix[intersect_ix]
        df1 = df.ix[diff_ix]
        df2 = df_past.ix[diff_ix_past]
        df = pd.concat([df0,df1,df2],axis=0)
    return df

#uniq count
def concat_list(x):
    return list(x[0])+list(x[1])

def count_uniq(x):
    return len(x[0])

def get_uniq(df,df_past=None,field='uniqCustomer',deep=True):
    #filed : 'uniqCustomer' , 'uniqProduct'
    if 'Customer' in field:
        if deep:
            df = df.groupby(['FileID','CustomerID'])['CustomerID'].unique()
        else:
            df = df.groupby('FileID')['CustomerID'].unique()
    elif 'Product' in field:
        if deep:
            df = df.groupby(['FileID','ProductID'])['ProductID'].unique()
        else:
            df = df.groupby('FileID')['ProductID'].unique()
    elif 'Product' in field and 'Customer' in field:
        df = df.assign(ProductCustomerID=df['ProductID'].astype('str')+df['CustomerID'].astype('str'))
        df = df.drop(['ProductID','CustomerID'])
        if deep:
            df = df.groupby(['FileID','ProductCustomerID'])['ProductCustomerID'].unique()
        else:
            df = df.groupby('FileID')['ProductCustomerID'].unique()
    df.name = field
    ixs = df.index
    df = pd.DataFrame(df)
    df = get_nuniq(df,field=field)
    df = pd.DataFrame(df)
    if df_past is not None:
        #df  = pd.DataFrame({0:df.values,1:df_past.values})
        #df = df.apply(concat_list,axis=1)
        #df.index = ixs
        df = pd.concat([df_past,df],axis=0)
        #df.name = field
    return df

def get_nuniq(df,field='uniqCustomer'):
    df = df.groupby(df.index.get_level_values(0))[field].apply(count_uniq)
    return df

#count
def get_count(df,df_past=None,field='countCustomer',deep=True):
    #filed : 'countCustomer' , 'countProduct'
    if 'Product' in field and 'Customer' in field:
        print('prod_cus')
        df = df.assign(ProductCustomerID=df['ProductID'].astype('str')+df['CustomerID'].astype('str'))
        df = df.drop(['ProductID','CustomerID'],axis=1)
        df = df.groupby(['FileID','ProductCustomerID'])['ProductCustomerID'].count()
    elif 'Customer' in field:
        print('cus')
        if deep:
            df = df.groupby(['FileID','CustomerID'])['CustomerID'].count()
        else:
            df = df.groupby('FileID')['CustomerID'].count()
    elif 'Product' in field:
        print('prod')
        if deep:
            df = df.groupby(['FileID','ProductID'])['ProductID'].count()
        else:
            df = df.groupby('FileID')['ProductID'].count()

    df.name = field
    df = pd.DataFrame(df)
    if df_past is not None:
        #df = df_add(df,df_past)
        df = df_past.append(df)
    df.name = field
    return df

def get_daily_count(df,df_past=None):
    df = df.groupby('FileID')['FileID'].count()
    df.name = 'FileIDCountDaily'
    df = pd.DataFrame(df)
    if df_past is not None:
        df = df_past.append(df) 
    df = pd.DataFrame(df)
    return df

#datetime
def get_datetime(ts,fields=['hour','weekday']):
    dt = datetime.fromtimestamp(ts)
    dt_dict = {}
    for field in fields:
        if field == 'hour':
            dt_dict[field] = dt.hour
        elif field == 'weekday':
            dt_dict[field] = dt.weekday()
    return dt_dict

def get_hour(ts):
    return get_datetime(ts,['hour'])['hour']

def get_weekday(ts):
    return get_datetime(ts,['weekday'])['weekday']

def get_hour_df(df,df_past=None):
    df = df.set_index('FileID')
    df_hour = df['QueryTs'].apply(get_hour)
    df_hour.name = 'hour'
    df_hour = pd.DataFrame(df_hour)
    df_hour = pd.get_dummies(df_hour.hour)
    df_hour = df_hour.groupby(df_hour.index.get_level_values(0)).sum()
    cols = df_hour.columns
    cols = ['hour%s'%str(c) for c in cols]
    df_hour.columns = cols
    if df_past is not None:
        df_hour = df_past.append(df_hour)
    return df_hour

def get_week_df(df,df_past=None):
    if 'FileID' in df.columns:
        df = df.set_index('FileID')
    df = df['QueryTs'].apply(get_weekday)
    df.name = 'weekday'
    df = pd.DataFrame(df)
    df = pd.get_dummies(df.weekday)
    df = df.groupby(df.index.get_level_values(0)).sum()
    cols = df.columns
    cols = ['week%s'%str(c) for c in cols]
    df.columns = cols
    if df_past is not None:
        df = df_past.append(df)
    return df

def get_norm_df(df,read_file=None): 
    if read_file:
        df = pd.read_csv('export/%s.csv'%read_file)
    if 'FileID' in df.columns:
        df= df.set_index('FileID')
    cols = []
    for k,v in df.iteritems():
        if 'sum' in k:
            cols.append(k)
    if len(cols) == 0:
        cols = []
        for k,v in df.iteritems():
            if 'mean' in k:
                cols.append(k)
        df_norm = norm_it(df[cols])
        cols = df_norm.columns
        cols = [re.sub('mean','percentage',col) for col in cols]
    else:
        df_norm = norm_it(df[cols])
        cols = df_norm.columns
        cols = [re.sub('sum','percentage',col) for col in cols]
    df_norm.columns = cols
    df = pd.concat([df,df_norm],axis=1)
    return df
   
def extend_cols(df):
    i0 = df.columns.get_level_values(0)
    i1 = df.columns.get_level_values(1)
    cols = ['%s_%s'%(x[0],x[1]) for x in zip(i0,i1)]
    df.columns = cols
    return df
  
############ etl ############

############ model ############
def get_data(version=4):
    cols = ['FileID','y']
    df = pd.read_csv('%s/trend_v%s.csv'%(export_file_path,version))
    df = df.set_index('FileID')
    test = pd.read_csv(file_path+'testing-set.csv',header=None)
    train = pd.read_csv(file_path+'training-set.csv',header=None)
    test.columns = cols
    train.columns = cols
    train = train.set_index('FileID')
    test = test.set_index('FileID')
    train_indices = train.index
    test_indices = test.index
    train = pd.concat([df.ix[train_indices],train],axis=1)
    y = train.pop('y')
    test = df.ix[test_indices]
    return train, y, test

get_str_FileID = lambda x:' '.join(list(x.FileID))
def get_list_FileID(x):
    return list(x.FileID)
############ model ############

###### deep learning etl #######
def get_count_nuniq_by_tm_fileid(df,df_past=None):
    df_count = df.groupby(['FileID','QueryTs'])['CustomerID','ProductID'].count()
    df_count.columns = ['%sCount'%col for col in df_count.columns] 
    df_nuniq = df.groupby(['FileID','QueryTs'])['CustomerID','ProductID'].nunique()
    df_nuniq.columns = ['%sUniqCount'%col for col in df_nuniq.columns] 
    df_count = pd.concat([df_count,df_nuniq],axis=1) 
    df_count = df_count.reset_index()
    print(df_count.head())
    if df_past is not None:
        df_count = pd.concat([df_past,df_count],axis=0)
    return df_count

def df_generator(files=files):
    for f in files:
        print(f)
        df = pd.read_csv(f,header=None)
        df.rename(columns={0: 'FileID', 1: 'CustomerID',2:'QueryTs',3:'ProductID'}, inplace=True)
        df = clean_df(df)
        df['QueryTs'] = df['QueryTs'].astype(int)
        yield df
    
def dict_generator(files=files,ns=ns,n=31):
    print('ns :',ns)
    print('n :',n)
    for i in range(ns)[::n]:
        fs = files[i:i+n]
        print('no.%s =====> %s'%(i,fs))
        df = [d for d in df_generator(files=fs)]
        print('len of df =',len(df))
        df = pd.concat(df,axis=0)
        if i == 0:
            df_past = get_count_nuniq_by_tm_fileid(df)
        else:
            df_past = get_count_nuniq_by_tm_fileid(df,df_past)
            df_past = df_past.groupby(['FileID','QueryTs']).sum()
    df_past = df_past.reset_index()
    df_past = df_to_dict(df_past)
    for d in df_past:
        yield d
=======
############ dnn ############
from keras.models import Sequential
from keras.layers.core import Dense,Activation
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.utils import to_categorical
from sklearn import metrics
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from keras.callbacks import Callback
from keras.backend import clear_session
from keras.layers import Dense, Dropout, Flatten, Activation, BatchNormalization, regularizers
from keras import backend as K
import tensorflow as tf

def binary_crossentropy_with_ranking(y_true, y_pred):
    #https://gist.github.com/jerheff/8cf06fe1df0695806456
    #https://github.com/keras-team/keras/issues/1732#issuecomment-358236607
    #http://tflearn.org/objectives/#roc-auc-score
    print(y_true.dtype,y_pred.dtype)
    """ Trying to combine ranking loss with numeric precision"""
    # first get the log loss like normal
    logloss = K.mean(K.binary_crossentropy(y_pred, y_true), axis=-1)
    
    # next, build a rank loss
    
    # clip the probabilities to keep stability
    y_pred_clipped = K.clip(y_pred, K.epsilon(), 1-K.epsilon())

    # translate into the raw scores before the logit
    y_pred_score = K.log(y_pred_clipped / (1 - y_pred_clipped))

    # determine what the maximum score for a zero outcome is
    y_pred_score_zerooutcome_max = K.max(y_pred_score * tf.cast((y_true <1), tf.float32))

    # determine how much each score is above or below it
    rankloss = y_pred_score - y_pred_score_zerooutcome_max

    # only keep losses for positive outcomes
    rankloss = rankloss * y_true

    # only keep losses where the score is below the max
    rankloss = K.square(K.clip(rankloss, -100, 0))

    # average the loss for just the positive outcomes
    rankloss = K.sum(rankloss, axis=-1) / (K.sum(tf.cast((y_true <1), tf.float32)) + 1)

    # return (rankloss + 1) * logloss - an alternative to try
    return rankloss + logloss

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

class roc_callback(Callback):
    def __init__(self,training_data,validation_data):
        self.x = training_data[0]
        self.y = training_data[1]
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.x)
        roc = roc_auc_score(self.y, y_pred)
        y_pred_val = self.model.predict(self.x_val)
        roc_val = roc_auc_score(self.y_val, y_pred_val)
        print('\rroc-auc: %s - roc-auc_val: %s' % (str(round(roc,4)),str(round(roc_val,4))),end=100*' '+'\n')
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return

def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate

@static_vars(stream_vars=None)
def tf_auc_roc(y_true, y_pred):
    value, update_op = tf.contrib.metrics.streaming_auc(
        y_pred, y_true, curve='ROC', name='auc_roc')
    tf_auc_roc.stream_vars = [i for i in tf.local_variables() if i.name.split('/')[0] == 'auc_roc']
    return control_flow_ops.with_dependencies([update_op], value)

def tf_auc_roc(y_true, y_pred):
    # any tensorflow metric
    value, update_op = tf.contrib.metrics.streaming_auc(y_pred, y_true)

    # find all variables created for this metric
    metric_vars = [i for i in tf.local_variables() if 'auc_roc' in i.name.split('/')[1]]

    # Add metric variables to GLOBAL_VARIABLES collection.
    # They will be initialized for new session.
    for v in metric_vars:
        tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, v)

    # force to update metric values
    with tf.control_dependencies([update_op]):
        value = tf.identity(value)
        return value
############ dnn ############
