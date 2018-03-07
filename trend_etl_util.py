from trend_util import *
from functools import partial
import glob
import re
logging.basicConfig(filename='log/trend_etl_util.log',level=logging.DEBUG)
files = glob.glob(rawdata_path+'/*.csv')
list.sort(files)
#files = files[-30:-20]
#files = [rawdata_path+'0325.csv',rawdata_path+'0326.csv']
#files = [rawdata_path+'0520.csv',rawdata_path+'0301.csv']
ns = len(files)

def get_files_every_nday(n,files=files,df_intersect=None):
    print('======= get_files_every_nday =======')
    for i in range(ns)[::n]:
        fs = files[i:i+n]
        print(i,fs)
        dfs = [pd.read_csv(f,header=None) for f in fs]
        df = pd.concat(dfs,axis=0)
        df.rename(columns={0: 'FileID', 1: 'CustomerID',2:'QueryTs',3:'ProductID'}, inplace=True)
        df = clean_df(df)
        if df_intersect is not None:
            print('###### common filter ######')
            df = common_filter(df,df_intersect)
        yield ' '.join(df['FileID'].values)

#count base on hour
def etl_hour(df_hour,suffix=''):
    print('cal hour')
    df_hour = get_aggr(df_hour) 
    df_hour = extend_cols(df_hour)
    #df_hour = get_norm_df(df_hour)
    filename = 'export/trend_hour%s.csv'%suffix
    print('dump %s ...'%filename)
    df_hour.to_csv(filename)

#count base on week
def etl_week(df_week,suffix=''):
    print('cal week')
    df_week = get_aggr(df_week)
    df_week = extend_cols(df_week)
    #df_week = get_norm_df(df_week)
    filename = 'export/trend_week%s.csv'%suffix
    print('dump %s ...'%filename)
    df_week.to_csv(filename)

#get_file_product_count day by day and aggr
def etl_file_prod_count(df_prod_daily,suffix=''):
    print('dump prod daily...')
    flags = ['sum','mean','std','skew','mad']
    df_prod_daily.to_csv('export/trend_prod_daily_raw%s.csv'%suffix)
    df_prod_daily = get_aggr(df_prod_daily,flags=flags)
    df_prod_daily = extend_cols(df_prod_daily)
    #df_prod_daily = get_norm_df(df_prod_daily)
    filename = 'export/trend_prod_daily%s.csv'%suffix
    print('dump %s ...'%filename)
    df_prod_daily.to_csv(filename)

#count customer day by day
def etl_customer(df_cus_count,suffix=''):
    print('dump customer count...')
    flags = ['sum','size','mean','std','max','min','median','skew','mad',percentile(25),percentile(75)]
    df_cus_count= get_aggr(df_cus_count,flags=flags)
    df_cus_count = extend_cols(df_cus_count)
    filename = 'export/trend_cus_count%s.csv'%suffix
    print('dump %s ...'%filename)
    df_cus_count.to_csv(filename)

#count product day by day
def etl_prod(df_prod_count,suffix=''):
    print('dump prod count...')
    flags = ['sum','size','mean','std','max','min','median','skew','mad',percentile(25),percentile(75)]
    df_prod_count= get_aggr(df_prod_count,flags=flags)
    df_prod_count = extend_cols(df_prod_count)
    filename = 'export/trend_prod_count%s.csv'%suffix
    print('dump %s ...'%filename)
    df_prod_count.to_csv(filename)

#count prod+customer day by day
def etl_prod_cus(df_cus_count,suffix=''):
    print('dump prod+customer count...')
    flags = ['sum','size','mean','std','max','min','median','skew','mad',percentile(25),percentile(75)]
    df_cus_count= get_aggr(df_cus_count,flags=flags)
    df_cus_count = extend_cols(df_cus_count)
    filename = 'export/trend_prod_cus_count%s.csv'%suffix
    print('dump %s ...'%filename)
    df_cus_count.to_csv(filename)

#count uniq costomer day by day
def etl_costomer_uniq(df_cus_nuniq,suffix=''):
    print('dump uniq customer count...')
    flags = ['sum','size','mean','std','max','min','median','skew','mad',percentile(25),percentile(75)]
    #df_cus_nuniq = get_nuniq(df_cus_uniq,field='uniqCustomer')
    df_cus_nuniq = get_aggr(df_cus_nuniq,flags=flags)
    df_cus_nuniq = extend_cols(df_cus_nuniq)
    filename = 'export/trend_cus_nuniq%s.csv'%suffix
    print('dump %s ...'%filename)
    df_cus_nuniq.to_csv(filename)

#count uniq product day by day
def etl_prod_uniq(df_prod_nuniq,suffix=''):
    print('dump uniq product count...')
    flags = ['sum','size','mean','std','max','min','median','skew','mad',percentile(25),percentile(75)]
    #df_prod_nuniq = get_nuniq(df_prod_uniq,field='uniqProduct')
    df_prod_nuniq = get_aggr(df_prod_nuniq,flags=flags)
    df_prod_nuniq = extend_cols(df_prod_nuniq)
    filename = 'export/trend_prod_nuniq%s.csv'%suffix
    print('dump %s ...'%filename)
    df_prod_nuniq.to_csv(filename)

#count uniq product day by day
def etl_prod_cus_uniq(df_prod_nuniq,suffix=''):
    print('dump uniq prod+cus count...')
    flags = ['sum','size','mean','std','max','min','median','skew','mad',percentile(25),percentile(75)]
    #df_prod_nuniq = get_nuniq(df_prod_uniq,field='uniqProduct')
    df_prod_nuniq = get_aggr(df_prod_nuniq,flags=flags)
    df_prod_nuniq = extend_cols(df_prod_nuniq)
    filename = 'export/trend_prod_cus_nuniq%s.csv'%suffix
    print('dump %s ...'%filename)
    df_prod_nuniq.to_csv(filename)

#count by fileid daily
def etl_count_by_fileid(df_fileid_count_daliy,suffix=''): 
    print('dump count...')
    flags = ['sum','size','mean','std','max','min','median','skew','mad',percentile(25),percentile(75)]
    df_fileid_count_daliy = get_aggr(df_fileid_count_daliy,flags=flags)
    filename = 'export/trend_fileid_count_daily%s.csv'%suffix
    print('dump %s ...'%filename)
    df_fileid_count_daliy.to_csv(filename)

#open time interval
def etl_interval(df_interval,field='',suffix=''):
    print('dump open time interval...')
    #df_interval.to_csv('export/trend_interval_raw.csv',index=False)
    df_interval = get_open_time_aggr(df_interval,field=field)
    if len(field) >0: field = '_%s'%field.lower()
    filename = 'export/trend_interval%s%s.csv'%(field,suffix)
    print('dump %s ...'%filename)
    df_interval.to_csv(filename)

#product tfidf 
#customer tfidf 
def etl_tfidf(df_tfidf,field=None,suffix=''):
    df_tfidf = cal_tfidf(df_tfidf,field)
    filename = 'export/trend_tfidf_%s%s.csv'%(field,suffix)
    print('dump %s ...'%filename)
    df_tfidf.to_csv(filename)

etl_tfidf_prod = partial(etl_tfidf,field='ProductID')
etl_tfidf_cus = partial(etl_tfidf,field='CustomerID')

#per day tfidf
def etl_tfidf_day(n=1,df_intersect=None,suffix=''):
    field = 'day'
    vecs = get_files_every_nday(n=n,df_intersect=df_intersect)
    df = cal_tfidf(vecs,field=field) 
    filename = 'export/trend_tfidf_%s%s.csv'%(field,suffix)
    print('dump %s ...'%filename)
    df.to_csv(filename)

#get intersection btw train and test: prod set, cus set, prod_cus set 
def etl_intersection_train_test(n=31,ns=ns,dump=False,reverse=False,breakout=None,
                                return_fields=['prod_cus','product','customer'],**kwargs):
    if n > ns: n = ns
    for i in range(ns)[::n]:
        if breakout:
            if i >= breakout: return df_past 
        fs = files[i:i+n]
        print(i,fs)
        dfs = [pd.read_csv(f,header=None) for f in fs]
        df = pd.concat(dfs,axis=0)
        df.rename(columns={0: 'FileID', 1: 'CustomerID',2:'QueryTs',3:'ProductID'}, inplace=True)
        df = clean_df(df)
        if i == 0:
            df_past = get_train_test_prod_cus_set(df,return_fields=return_fields,**kwargs)
        else:
            df_past = get_train_test_prod_cus_set(df,df_past,return_fields=return_fields,**kwargs)
    df = get_prod_cus_intersect(df_past,dump=dump,reverse=reverse,return_fields=return_fields)
    return df

def etl_dump_intersection_data(typ=['tight','loose'],ns=ns,dump=False,reverse=False,breakout=None,
                               return_fields=['prod_cus','product','customer'],**kwargs):
    df_intersect = etl_intersection_train_test(ns=ns,reverse=reverse,return_fields=return_fields,**kwargs)
    if not isinstance(typ,list):
        typs = [typ]
    else: typs = typ
    print('typs: ',typs)
    for i in range(ns):
        if breakout:
            if i >= breakout: return 
        f = files[i]
        print(i,f)
        #df = pd.read_csv(f,header=None)
        df = check_file_empty(f,header=None)
        df.rename(columns={0: 'FileID', 1: 'CustomerID',2:'QueryTs',3:'ProductID'}, inplace=True)
        df = clean_df(df)
        for typ in typs:
            print('typ: ',typ)
            df = common_filter(df,df_intersect,typ=typ)
            if df is False: 
                print('df is empty, skip %s'%f)
                logging.info('df is empty, skip %s ...'%f)
                continue
            if 'FileID' in df.columns:
                df = df.set_index('FileID')
            f = re.search('/(\d+\.csv)',f).group(1)
            f = '%s%s_intersect/%s'%(etl_file_path,typ,f)
            print('dumping %s ...'%f)
            df.to_csv(f,header=None) 
