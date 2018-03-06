from trend_etl_util import *

n = 1 

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
        #FileID被各個ProductID開啟的次數的比例
        df_prod_daily = get_file_product_count(df)
        df_cus_count = get_count(df,field='countCustomer')
        df_prod_count = get_count(df,field='countProduct')
        df_cus_nuniq = get_uniq(df,field='uniqCustomer')
        df_prod_nuniq = get_uniq(df,field='uniqProduct')
        df_prod_cus_count = get_count(df,field='countProductCustomer')
        df_prod_cus_nuniq = get_uniq(df,field='uniqProductCustomer')
        df_fileid_count_daliy = get_count_daily(df)
        df_hour = get_hour_df(df)
    else:
        df_prod_daily = get_file_product_count(df,df_prod_daily)
        df_cus_count = get_count(df,df_cus_count,field='countCustomer')
        df_prod_count = get_count(df,df_cus_count,field='countProduct')
        df_cus_nuniq = get_uniq(df,df_cus_nuniq,field='uniqCustomer')
        df_prod_nuniq = get_uniq(df,df_prod_nuniq,field='uniqProduct')
        df_prod_cus_count = get_count(df,df_prod_cus_count,field='countProductCustomer')
        df_prod_cus_nuniq = get_uniq(df,df_prod_cus_nuniq,field='uniqProductCustomer')
        df_fileid_count_daliy = get_count_daily(df,df_fileid_count_daliy)
        df_hour = get_hour_df(df,df_hour)


#count
etl_cusromer(df_cus_count)
etl_prod(df_prod_count)
etl_prod_cus(df_prod_cus_count)
etl_count_by_fileid(df_fileid_count_daliy)

#nuniq
etl_costomer_uniq(df_cus_nuniq)
etl_prod_uniq(df_prod_nuniq)
etl_prod_cus_uniq(df_prod_cus_nuniq)

#product detail
etl_file_prod_count(df_prod_daily)

etl_hour(df_hour)

etl_tfidf_day(n=1)
