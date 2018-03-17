from trend_etl_dl import *
print(ns,n)
dfs = df_count_generator(files,ns=ns,n=n)
df_count_dump(dfs)

