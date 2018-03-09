from trend_util import *
files = ['trend_tfidf_CustomerID0.csv']
files = ['trend_interval.csv','trend_prod_daily.csv','trend_tfidf_ProductID.csv',
         'trend_week_norm.csv','trend_hour_norm.csv','trend_fileid_count_daily.csv',
         'trend_tfidf_day.csv','trend_interval_productid.csv','trend_interval_customerid.csv']

files_concat = ['trend_prod_count.csv','trend_prod_nuniq.csv']
files_concat += ['trend_cus_count.csv','trend_cus_nuniq.csv']
files_concat += ['trend_prod_cus_nuniq.csv','trend_prod_cus_count.csv']
files_concat = ['trend_prod_count_deep.csv','trend_prod_nuniq_deep.csv']
files_concat += ['trend_cus_count_deep.csv','trend_cus_nuniq_deep.csv']
files_concat += ['trend_prod_cus_nuniq_deep.csv','trend_prod_cus_count_deep.csv']
files_concat_list = [files_concat]

vari = False
cori = True
vari_threshold = 0.25
cori_threshold = 0.999
show_remain = False

def remove_percentage_cols(df):
    cols = [] 
    for col in df.columns:
        if 'percentage' in col:
            cols.append(col)
    df = df.drop(cols,axis=1)
    return df 

def filter_data(df,vari=True,cori=True):
    #remove_cols = find_correlation_to_remove(df,cori_threshold)
    #df = df.drop(remove_cols,axis=1)
    df = get_lowcorr_df(df,threshold=cori_threshold)
    df = remain_high_variance(df,True,vari_threshold,show_remain=show_remain)
    df = remove_percentage_cols(df)
    return df

dfs = []
for f in files:
    print(f)
    df = pd.read_csv('export/'+f)
    df = df.set_index('FileID')
    df = filter_data(df,vari=vari,cori=cori)
    dfs.append(df)
    print('-----')

for files_con in files_concat_list:
    dfc= []
    for f in files_con:
        df = pd.read_csv('export/'+f)
        df = df.set_index('FileID')
        dfc.append(df)
    df = pd.concat(dfc,axis=1)
    df = filter_data(df,vari=vari,cori=cori)
    dfs.append(df)
     
df = pd.concat(dfs,axis=1)
#remove_cols = find_correlation_to_remove(df, threshold=0.95)
#print(remove_cols)
#df = df.drop(remove_cols,axis=1)

#print(df.head())
print(df.shape)
df = df.reset_index()
df.to_csv('export/trend_v5.csv',index=False)
