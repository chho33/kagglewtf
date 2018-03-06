from trend_etl_util import *

n = 7 
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
        df_week = get_week_df(df)
    else:
        df_week = get_week_df(df,df_week)

etl_week(df_week)

