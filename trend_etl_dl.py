from trend_util import *
from functools import partial
import glob
import re
block_size = 10000
n = 31

def get_count_nuniq_by_tm_fileid(df,df_past=None):
    df_count = df.groupby(['FileID','QueryTs'])['CustomerID','ProductID'].count()
    df_count.columns = ['%sCount'%col for col in df_count.columns]
    #df_nuniq = df.groupby(['FileID','QueryTs'])['CustomerID','ProductID'].nunique()
    dfp = df.groupby(['FileID','QueryTs'])['ProductID'].nunique()
    dfp.name = 'ProductID'
    dfc = df.groupby(['FileID','QueryTs'])['CustomerID'].nunique()
    dfc.name = 'CustomerID'
    df_nuniq = pd.concat([dfc,dfp],axis=1)
    df_nuniq.columns = ['%sUniqCount'%col for col in df_nuniq.columns]
    df_count = pd.concat([df_count,df_nuniq],axis=1)
    df_count = df_count.reset_index()
    if df_past is not None:
        df_count = pd.concat([df_past,df_count],axis=0)
    #df_count = df_count.sort_values('QueryTs')
    return df_count

def df_generator(files=files):
    for f in files:
        print(f)
        df = pd.read_csv(f,header=None)
        df.rename(columns={0: 'FileID', 1: 'CustomerID',2:'QueryTs',3:'ProductID'}, inplace=True)
        df = clean_df(df)
        df['QueryTs'] = df['QueryTs'].astype(int)
        yield df

def df_count_generator(files=files,ns=ns,n=31):
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
    if 'FileID' not in df_past.columns:
        df_past = df_past.reset_index()
    df_past = df_to_dict(df_past)
    for d in df_past:
        yield d

def df_count_generator(files=files,ns=ns,n=31):
    print('======= df_count_generator =======')
    print('files: ',files)
    print('ns :',ns)
    print('n :',n)
    for i in range(ns)[::n]:
        fs = files[i:i+n]
        print('no.%s =====> %s'%(i,fs))
        df = [d for d in df_generator(files=fs)]
        print('len of df =',len(df))
        df = pd.concat(df,axis=0)
        df = get_count_nuniq_by_tm_fileid(df)
        if 'FileID' in df.columns:
            df = df.set_index('FileID')
        yield df
    
#df = df_count_generator(files)      
def df_count_dump(dfs,train_unhealthy=None, test_unhealthy=None):
    logging.basicConfig(filename='log/trend_etl_dl.log',level=logging.DEBUG)
    dfs = [df for df in dfs]
    dfs = pd.concat(dfs,axis=0)
    train_ids = get_data_ids(typ='train')
    test_ids = get_data_ids(typ='test')
    #train_ids = set(dfs.index)
    for i in train_ids:
        path = 'data/train/%s.csv'%i
        if isinstance(train_unhealthy,list):
            if path not in train_unhealthy: continue
        df = dfs.ix[i]
        if isinstance(df,pd.Series):
            df = pd.DataFrame(df).T
        print('train id: ',i)
        logging.info(path)
        #print(df.head())
        df.to_csv(path,index=False)
        print('-------')
        #dfs = dfs[~dfs.index.isin([i])]
    for i in test_ids:
        path = 'data/test/%s.csv'%i
        if isinstance(test_unhealthy,list):
            if path not in test_unhealthy: continue
        df = dfs.ix[i]
        if isinstance(df,pd.Series):
            df = pd.DataFrame(df).T
        print('test id: ',i)
        logging.info(path)
        #print(df.head())
        df.to_csv(path)
        print('-------')
        #dfs = dfs[~dfs.index.isin([i])]

def dump_count(block_size=block_size,files=files,ns=ns,n=n):
    blocks = []
    for i,data in enumerate(dict_generator(files=files,ns=ns,n=n)):
        if len(blocks) >= block_size:
            print('inserting ====> ',i,blocks[:3])
            with  r.connect(host=host,port=port,db=db) as conn:
                r.table(table_count).insert(blocks).run(conn)
            blocks = []
        blocks.append(data)
    print('inserting ====> ',i,blocks[:3])
    with  r.connect(host=host,port=port,db=db) as conn:
        r.table(table_count).insert(blocks).run(conn)

def get_data_ids(typ='train'):
    if typ == 'train':
        ids = pd.read_csv(file_path+'training-set.csv',header=None)
        ids = list(ids[0])
    elif typ == 'test':
        ids = pd.read_csv(file_path+'testing-set.csv',header=None)
        ids = list(ids[0])
    return ids

import glob
fields = ['QueryTs','CustomerIDCount','CustomerIDUniqCount','ProductIDUniqCount']

def gen_matrixs(ids,func,typ='train',**kvargs):
    if not isinstance(ids,list): ids = [ids]
    for ix in ids:
        yield func(ix=ix,typ=typ,**kvargs)

def get_matrix(ix,typ,split_n=6,group_n=1,norm=False):
    interval = 86400 * 6
    df = pd.read_csv('data/%s/%s.csv'%(typ,ix))
    df = df[fields]
    df = df.sort_values('QueryTs')
    if 'FileID' in df.columns:
        df = df.drop(['FileID'],axis=1)
    data_start_tm = int(df.head(1).QueryTs)
    data_end_tm = int(df.tail(1).QueryTs)
    data_interval = data_end_tm - data_start_tm
    df = df.set_index('QueryTs')
    #print('%s shape: '%ix,df.shape)
    tm_ixs = df.index
    end_tm = data_start_tm + interval -1
    tm_arr = [i for i in range(data_start_tm,end_tm+1)]
    df_zero = pd.DataFrame({'QueryTs':tm_arr,'ProductIDUniqCount':0,'CustomerIDUniqCount':0,'CustomerIDCount':0})
    df_zero = df_zero.set_index('QueryTs')
    df_zero = df_zero.ix[~df_zero.index.isin(tm_ixs)]
    df = pd.concat([df,df_zero],axis=0)
    df = df.sort_index()
    df.index = range(0,interval)
    df = df.groupby(df.index // group_n * group_n).sum()
    if norm:
        print('normalized')
        df = normit(df,return_df=True)
    mat = df.as_matrix()
    mat = mat.reshape(split_n,int(interval/(split_n*group_n)),mat.shape[-1])
    return mat

def get_max_ts(ix,typ,group_n=1,norm=False):
    df = pd.read_csv('data/%s/%s.csv'%(typ,ix))
    #print(ix)
    df = df[fields]
    df = df.sort_values('QueryTs')
    if 'FileID' in df.columns:
        df = df.drop(['FileID'],axis=1)
    data_start_tm = int(df.head(1).QueryTs)
    data_end_tm = int(df.tail(1).QueryTs)
    df = df.set_index('QueryTs')
    #print('%s shape: '%ix,df.shape)
    tm_ixs = df.index
    tm_arr = [i for i in range(data_start_tm,data_end_tm+1)]
    df_zero = pd.DataFrame({'QueryTs':tm_arr,'ProductIDUniqCount':0,'CustomerIDUniqCount':0,'CustomerIDCount':0})
    df_zero = df_zero.set_index('QueryTs')
    df_zero = df_zero.ix[~df_zero.index.isin(tm_ixs)]
    df = pd.concat([df,df_zero],axis=0)
    df = df.sort_index()
    df.index = range(0,df.shape[0])
    df = df.groupby(df.index // group_n * group_n).sum()
    if norm:
        print('normalized')
        df = normit(df,return_df=True)
    mat = df.as_matrix()
    return mat

def slide_windows(mat,step=86400,width=86400*2,mode='lookback',threshold=0.8):
    #mode: lookback | padd
    step = int(step)
    width = int(width)
    blocks = []
    for i in range(0,len(mat))[::step]:
        print(i,i+width)
        block = data[0][i:i+width]
        if block.shape[0] < width:
            if block.shape[0] < int(width*threshold): return blocks
            if mode == 'lookback':
                block = mat[-width:]
            elif mode == 'padd':
                padd_num = width - block.shape[0]
                block_zero = np.zeros((padd_num,4), dtype=int)
                block = np.concatenate((block, block_zero), axis=0)
            blocks.append(block)
            print(block.shape)
            return blocks
        blocks.append(block)
        print(block.shape)
    return blocks

def get_train_answers():
    data = pd.read_csv(file_path+'training-set.csv',header=None)
    data = {k:v for k,v in  data.to_dict(orient='split')['data']}
    return data

def data_checker(typ='train'):
    files = glob.glob('data/%s/*.csv'%typ)
    unhealthy = []
    for f in files:
        df = pd.read_csv(f)
        if len(df.columns) <5:
            print(f)
            unhealthy.append(f)
    return unhealthy

