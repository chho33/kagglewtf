from trend_etl_util import *

n = 31
#for i,f in enumerate(files):
for i in range(ns)[::n]:
    fs = files[i:i+n]
    print(i,fs)
    dfs = [pd.read_csv(f,header=None) for f in fs]
    df = pd.concat(dfs,axis=0)
    #if i == 2: break
    #df = pd.read_csv(f,header=None)
    df.rename(columns={0: 'FileID', 1: 'CustomerID',2:'QueryTs',3:'ProductID'}, inplace=True)
    df = clean_df(df)
    if i == 0:
        df_cus_count = get_count(df,field='countCustomer')
        df_prod_count = get_count(df,field='countProduct')
        df_cus_nuniq = get_uniq(df,field='uniqCustomer')
        df_prod_nuniq = get_uniq(df,field='uniqProduct')
        df_prod_cus_count = get_count(df,field='countProductCustomer')
        df_prod_cus_nuniq = get_uniq(df,field='uniqProductCustomer')
        df_interval_prod_cus = get_field_open_time(df,field='ProductCustomerID')
        df_interval = get_open_time(df)
        df_interval_cus = get_field_open_time(df)
        df_interval_prod = get_field_open_time(df,field='ProductID')
        df_tfidf_prod = get_corpus(df,'ProductID')
        #df_tfidf_cus = get_corpus(df,'CustomerID')
    else:
        df_cus_count = get_count(df,df_cus_count,field='countCustomer')
        df_prod_count = get_count(df,df_prod_count,field='countProduct')
        df_cus_nuniq = get_uniq(df,df_cus_nuniq,field='uniqCustomer')
        df_prod_nuniq = get_uniq(df,df_prod_nuniq,field='uniqProduct')
        df_prod_cus_count = get_count(df,df_prod_cus_count,field='countProductCustomer')
        df_prod_cus_nuniq = get_uniq(df,df_prod_cus_nuniq,field='uniqProductCustomer')
        df_interval_prod_cus = get_field_open_time(df,df_interval_prod_cus,field='ProductCustomerID')
        df_interval = get_open_time(df,df_interval)
        df_tfidf_prod = get_corpus(df,df_tfidf_prod,'ProductID')
        df_interval_cus = get_field_open_time(df,df_interval_cus)
        df_interval_prod = get_field_open_time(df,df_interval_prod,field='ProductID')
        #df_tfidf_cus = get_corpus(df,df_tfidf_cus,'CustomerID')

suffix=''
etl_interval(df_interval['df'],suffix=suffix)
etl_interval(df_interval_cus['df'],field='CustomerID',suffix=suffix)
etl_interval(df_interval_prod['df'],field='ProductID',suffix=suffix)
etl_interval(df_interval_prod_cus['df'],field='ProductCustomerID',suffix=suffix)
etl_tfidf_prod(df_tfidf_prod,suffix=suffix)
#etl_tfidf_cus(df_tfidf_cus,suffix=suffix)

#count
etl_customer(df_cus_count,deep=True)
etl_prod(df_prod_count,deep=True)
etl_prod_cus(df_prod_cus_count,deep=True)

#nuniq
etl_costomer_uniq(df_cus_nuniq,deep=True)
etl_prod_uniq(df_prod_nuniq,deep=True)
etl_prod_cus_uniq(df_prod_cus_nuniq,deep=True)
