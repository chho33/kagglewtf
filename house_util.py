from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
import pandas as pd
import os
from mlxtend.regressor import StackingRegressor
from sklearn.decomposition import PCA,SparsePCA,TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression,Lasso,Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.cross_validation import train_test_split
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.metrics.pairwise import euclidean_distances,linear_kernel,cosine_similarity
from sklearn.cluster import DBSCAN
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler,Normalizer,MinMaxScaler,LabelEncoder,Imputer
from sklearn.feature_selection import VarianceThreshold
#import matplotlib.pyplot as plt
import copy
import numpy as np
from collections import Counter
import warnings
warnings.filterwarnings('ignore')
file_path = 'data/house/'

# too many na (>90%)
cols_del = ["MiscFeature","Alley"]

# int
cols_del += ["Id","BsmtHalfBath","BsmtFinSF2","FullBath","HalfBath","KitchenAbvGr",
            "Fireplaces","EnclosedPorch","3SsnPorch","ScreenPorch","PoolArea",
            "MiscVal","MoSold","YrSold"
           ]

# category data
cols_del += [
    "Utilities"
]

cat_cols =['MSZoning','Street','Alley','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood','Condition1','Condition2','BldgType','HouseStyle','RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','ExterQual','ExterCond','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','Heating','HeatingQC','CentralAir','Electrical','KitchenQual','Functional','FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive','PoolQC','Fence','MiscFeature','SaleType','SaleCondition']
cat_cols += ["MSSubClass","YearBuilt","YearRemodAdd","BsmtFullBath","BsmtHalfBath","MoSold","YrSold","YMSold"]
#cat_cols += ["MSSubClass","BsmtFullBath","BsmtHalfBath"]

# transform grades to int
cols_grade = ["ExterQual","ExterCond","BsmtQual","BsmtCond","BsmtExposure","BsmtFinType1","BsmtFinType2",
             "HeatingQC",'KitchenQual',"FireplaceQu","GarageQual","GarageCond","PoolQC","Fence"]
dict_grade = {
    "ExterQual":{"Ex":3,"Gd":2,"TA":1,"Fa":0,"Po":-1},
    "ExterCond":{"Ex":3,"Gd":2,"TA":1,"Fa":0,"Po":-1},
    "BsmtQual":{"Ex":3,"Gd":2,"TA":1,"Fa":0,"NA":0,"Po":-1},
    "BsmtCond":{"Ex":3,"Gd":2,"TA":1,"Fa":0,"NA":0,"Po":-1},
    "BsmtExposure":{"Gd":3,"Av":2,"Mn":1,"No":0,"NA":0},
    "BsmtFinType1":{"GLQ":2,"ALQ":1,"BLQ":0,"Rec":1,"LwQ":-1,"Unf":0,"NA":0},
    "BsmtFinType2":{"GLQ":2,"ALQ":1,"BLQ":0,"Rec":1,"LwQ":-1,"Unf":0,"NA":0},
    "HeatingQC":{"Ex":3,"Gd":2,"TA":1,"Fa":0,"Po":-1},
    "KitchenQual":{"Ex":3,"Gd":2,"TA":1,"Fa":0,"Po":-1},
    "FireplaceQu":{"Ex":3,"Gd":2,"TA":1,"Fa":0,"NA":0,"Po":-1},
    "GarageQual":{"Ex":3,"Gd":2,"TA":1,"Fa":0,"NA":0,"Po":-1},
    "GarageCond":{"Ex":3,"Gd":2,"TA":1,"Fa":0,"NA":0,"Po":-1},
    "PoolQC":{"Ex":3,"Gd":2,"TA":1,"Fa":0,"NA":0,"Po":-1},
    "Fence":{"GdPrv":2,"MnPrv":1,"GdWo":2,"MnWw":1,"NA":0}
}

imputer_dict = {
    "GarageCars" : 0, 
    "GarageArea" : 0,
    "BsmtFinSF2" : 0,
    "MSZoning"     : "most_frequent",
    "BsmtFullBath" : "most_frequent", 
    "BsmtHalfBath" : "most_frequent",
    "BsmtFinSF1"   : "median",
    "BsmtUnfSF"    : "median",
    "LotFrontage"  : "median",
    "MasVnrArea"   : "median",
    "TotalBsmtSF"  : "median"
}

def delete_cols(df):
    return df.drop(cols_del,axis=1)

def map_cols(df):
    for col,vals in dict_grade.items():
        df[col] = df[col].map(vals)
        df[col] = df[col].fillna(0)
    return df

# deal with na
def fill_cols_na(df):
    #idxs = pd.isnull(df[["GarageYrBlt"]]).any(1).nonzero()[0]
    #df["GarageYrBlt"].iloc[idxs] = df["YearBuilt"].iloc[idxs]
    df["GarageCars"] = df["GarageCars"].fillna(0)
    df["GarageArea"] = df["GarageArea"].fillna(0)
    if "BsmtFinSF2" in df.columns:
        df["BsmtFinSF2"] = df["BsmtFinSF2"].fillna(0) 
    #most
    df[["BsmtFullBath"]] = df[["BsmtFullBath"]].fillna(df[["BsmtFullBath"]].groupby(["BsmtFullBath"]).size().idxmax())
    if "BsmtHalfBath" in df.columns:
        df[["BsmtHalfBath"]] = df[["BsmtHalfBath"]].fillna(df[["BsmtHalfBath"]].groupby(["BsmtHalfBath"]).size().idxmax())
    #ave: "BsmtFinSF1","BsmtUnfSF","LotFrontage","MasVnrArea","TotalBsmtSF"
    df[["BsmtFinSF1"]] = df[["BsmtFinSF1"]].fillna(df[["BsmtFinSF1"]].mean())
    df[["BsmtUnfSF"]] = df[["BsmtUnfSF"]].fillna(df[["BsmtUnfSF"]].mean())
    df[["LotFrontage"]] = df[["LotFrontage"]].fillna(df[["LotFrontage"]].mean())
    df[["MasVnrArea"]] = df[["MasVnrArea"]].fillna(df[["MasVnrArea"]].mean())
    df[["TotalBsmtSF"]] = df[["TotalBsmtSF"]].fillna(df[["TotalBsmtSF"]].mean())
    return df

def delete_useless(df):
    return df.drop(['Id','GarageYrBlt',"MiscFeature","Alley"],axis=1)

mq_dict = {
    1:1,
    2:1,
    3:1,
    4:2,
    5:2,
    6:2,
    7:3,
    8:3,
    9:3,
    10:4,
    11:4,
    12:4
}    

def m2q(m):
    return mq_dict[m]

def combine_cols(df,q=False):
    if q:
        df = df.assign(QSold=df.MoSold.apply(m2q))
        df = df.assign(YQSold=df.YrSold.astype('str')+df.QSold.astype('str'))
        return df.drop(['YrSold','MoSold','QSold'],axis=1)
    else:
        df = df.assign(YMSold=df.YrSold.astype('str')+df.MoSold.astype('str'))
        return df.drop(['YrSold','MoSold'],axis=1)

def get_data(delete=False):
    train = pd.read_csv(file_path+'train.csv',header=0)
    train = delete_useless(train)
    train = map_cols(train)
    num_train = train.shape[0]
    test = pd.read_csv(file_path+'test.csv',header=0)
    test = delete_useless(test)
    test = map_cols(test)
    if delete:
        train = delete_cols(train)
        test = delete_cols(test)
    #train = create_dummies(train)
    #test = create_dummies(test)
    y_train = train.pop('SalePrice')
    all_data = pd.concat([train,test],axis=0)
    all_data = create_dummies(all_data)
    all_data = fill_cols_na(all_data)
    #for col in set(list(train.columns))-set(list(test.columns)):
    #    all_data[col] = all_data[col].fillna(0)
    train,test = pd.concat([all_data[:num_train],y_train],axis=1),all_data[num_train:]
    return train,test

def get_data(delete=False,combine=True,label=True,dummy=False,fill_na=False,concat=False):
    train = pd.read_csv(file_path+'train.csv',header=0)
    test = pd.read_csv(file_path+'test.csv',header=0)
    ids = test['Id']
    num_train = train.shape[0]
    y = train.pop('SalePrice')
    x_all = pd.concat([train,test],axis=0)
    x_all = delete_useless(x_all)
    if delete:
        x_all = delete_cols(x_all)
    x_all = map_cols(x_all)
    
    if combine:
        x_all = combine_cols(x_all)
    
    if dummy:
        x_all = create_dummies(x_all)
    else:
        if label:
            # cat --> number
            x_all = label_encode(x_all)
        '''
        for col in x_all:
            if x_all[col].dtype.kind == 'O':
                x_all[col] = x_all[col].astype('category')
        cat_columns = x_all.select_dtypes(['category']).columns
        x_all[cat_columns] = x_all[cat_columns].apply(lambda c: c.cat.codes)
        '''
    
    # NA value
    if fill_na:
        x_all = fill_cols_na(x_all)
    
    if concat:
        return x_all, y, ids 
    x,test = x_all[:num_train],x_all[num_train:]
    return x,y,test,ids 

def label_encode(data):
    data = copy.deepcopy(data)
    for col in data:
        if data[col].dtype.kind == 'O' or col in cat_cols:
            le = LabelEncoder()  
            le.fit(data[col].astype('str'))
            data[col] = data[col].apply(lambda x: le.transform([x])[0] if type(x) == str else x)
    return data

def dim_reduc(data,reduc_obj):
    data = copy.deepcopy(data)
    data = reduc_obj.fit_transform(data)
    return data

def create_dummies(data):
    for col in data:
        #if data[col].isnull().values.any() or data[col].dtype.kind == 'O':
        if data[col].dtype.kind == 'O' or col in cat_cols:
            if data[col].isnull().values.any():
                dummies = pd.get_dummies(data[col])
            else:
                dummies = pd.get_dummies(data[col],drop_first=True)
            dcols = dummies.columns
            dcols = list(map(lambda x:'%s_%s'%(col,x),dcols))
            dummies.columns = dcols
            data.drop(col,axis=1,inplace=True)
            data = pd.concat([data,dummies],axis=1)
    return data

def get_all_cats(data):
    cols = []
    for col in data:
        if data[col].dtype.kind == 'O':  
            print(col)
            cols.append(col)
    return cols

# na percentage
def get_na_count(data):
    na_cols = get_na_cols(data)
    return count_na_percent(data,na_cols)

def get_na_cols(data):
    na_cols = []
    for col in data:
        if data[col].isnull().values.any():
            na_cols.append(col)
    return na_cols

def count_na_percent(data,na_cols):
    na_dict = {}
    for col in na_cols:
        c = Counter(data[col].isnull().values)
        na_dict[col] = c[True]/(c[True]+c[False])
    return na_dict

def get_full_na_cols(data):
    na_cols = []
    full_cols = []
    for col in data:
        if data[col].isnull().values.any():
            na_cols.append(col)
        else:
            full_cols.append(col)
    return full_cols,na_cols

# corr btw cols

# cluster to fill missing value
def fill_cols_na_by_cluster(data,gmm_file=True,keep_groups=False):
    full_cols, na_cols = get_full_na_cols(data)
    data_full,data_na = data[full_cols],data[na_cols]
    if gmm_file:
        best_gmm = joblib.load('export/best_gmm.pkl')
    else:
        best_gmm = best_gmm_cluster(data_full)
        joblib.dump(best_gmm,'export/best_gmm.pkl')
    groups = best_gmm.predict(data_full)
    data_na['Id'] = data_full['Id'] = [i for i in range(data.shape[0])]
    data_na['groups'] = groups
    gs = set(groups) 
    data_gs = []
    for g in gs:
        data_g = data_na.query('groups == %s'%g)
        for col in na_cols: 
            if col in imputer_dict:
                strategy = imputer_dict[col]
            else:
                strategy = "most_frequent"
            if strategy == 0:
                data_g[col] = data_g[col].fillna(0)
            else:
                imp = Imputer(missing_values='NaN', strategy=strategy, axis=0)
                if data_g.shape[0] < 5:
                    if strategy=="most_frequent": data_g[[col]] = data_g[[col]].fillna(data[[col]].groupby([col]).size().idxmax())
                    elif strategy=="median": data_g[[col]] = data_g[[col]].fillna(data[[col]].median())
                else:
                    imp = imp.fit(data_g[[col]])
                    data_g[[col]] = imp.transform(data_g[[col]])
        data_gs.append(data_g)
    data_na = pd.concat(data_gs,axis=0)
    data = pd.merge(data_full, data_na.drop('groups',axis=1), on='Id')
    if keep_groups:
        data['groups'] = groups 
    return data.drop('Id',axis=1)

def best_gmm_cluster(data):
    lowest_bic = np.infty
    bic = []
    n_components_range = range(6, 20)
    cv_types = ['spherical', 'tied', 'diag', 'full']
    for cv_type in cv_types:
        for n_components in n_components_range:
            # Fit a mixture of Gaussians with EM
            gmm = GaussianMixture(n_components=n_components,covariance_type=cv_type)
            gmm.fit(data)
            bic.append(gmm.aic(data))
            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
                best_gmm = gmm
    return best_gmm

# variance
def remain_high_variance(data,return_df=False):
    sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
    result = sel.fit_transform(data)
    cols = data.columns
    remain_cols = np.array(cols)[sel.get_support()]
    del_cols = np.array(cols)[np.logical_not(sel.get_support())]
    print ('remain cols: ', remain_cols )
    print ('del cols: ', del_cols )
    if return_df: return data[remain_cols]
    return result 

# clusters --> diff model

# normalization
def normit(data):
    scaler = StandardScaler()
    scaler.fit(data)
    return scaler.transform(data)

def prox_matrix(df, y, features, cluster_dimension,trees = 10):
    #https://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm#prox

    #initialize datframe for independant variables
    independant = pd.DataFrame()

    #Handle Categoricals: This should really be added to RandomForestRegressor
    for column,data_type in df[features].dtypes.iteritems():       
        try:
            independant[column] = pd.to_numeric(df[column],downcast = 'integer')
        except ValueError:
            contains_nulls = df[column].isnull().values.any()
            dummies = pd.get_dummies(df[column],prefix=column,dummy_na=contains_nulls,drop_first=True)
            independant[dummies.columns] = dummies

    if len(independant.index) != len(df.index):
        raise Exception('independant variables not stored properly')

    #train Model    
    clf = RandomForestRegressor(n_estimators=trees, n_jobs=-1)
    clf.fit(independant, y)

    #Final leaf for each tree
    leaves = clf.apply(independant)
    #value in cluster dimension
    labels = df[cluster_dimension].values

    numerator_matrix = {}
    for i,value_i in enumerate(labels):
        for j,value_j in enumerate(labels):
            if i >= j:       
                numerator_matrix[(value_i,value_j)] = numerator_matrix.get((value_i,value_j), 0) + np.count_nonzero(leaves[i]==leaves[j])
                numerator_matrix[(value_j,value_i)] = numerator_matrix[(value_i,value_j)] 

    #normalize by the total number of possible matchnig leaves        
    prox_matrix = {key: 1.0 - float(x)/(trees*np.count_nonzero(labels==key[0])*np.count_nonzero(labels==key[1])) for key, x in numerator_matrix.items()}                                                                  

    #make sorted dataframe                                                                                                                                                                                                                                                                
    levels = np.unique(labels)
    D = pd.DataFrame(data=[[ prox_matrix[(i,j)] for i in levels] for j in levels],index=levels,columns=levels)

    return D

def kMedoids(D, k, tmax=100):
    #https://www.researchgate.net/publication/272351873_NumPy_SciPy_Recipes_for_Data_Science_k-Medoids_Clustering

    # determine dimensions of distance matrix D
    m, n = D.shape

    if m != n:
        raise Exception('matrix not symmetric')

    if sum(D.columns.values != D.index.values):
        raise Exception('rows and columns do not match')

    if k > n:
        raise Exception('too many medoids')

    #Some distance matricies will not have a 0 diagonal    
    Dtemp =D.copy()
    np.fill_diagonal(Dtemp.values,0)

    # randomly initialize an array of k medoid indices
    M = list(Dtemp.sample(k).index.values)

    # initialize a dictionary to represent clusters
    Cnew = {}

    for t in range(tmax):    
        # determine mapping to clusters
        J = Dtemp.loc[M].idxmin(axis='index')
        #Fill dictionary with cluster members
        C = {kappa: J[J==kappa].index.values for kappa in J.unique()}  
        # update cluster medoids
        Cnew = {Dtemp.loc[C[kappa],C[kappa]].mean().idxmin() : C[kappa] for kappa in C.keys()}       
        #Update mediod list
        M = Cnew.keys()

        # check for convergence (ie same clusters)
        if set(C.keys()) == set(Cnew.keys()):
            if not sum(set(C[kappa]) != set(Cnew[kappa]) for kappa in C.keys()): break            
    else:        
        print('did not converge')

    #Calculate silhouette 
    #a(i) is a measure of how dissimilar i is to its own cluster, a small value means it is well matched. Furthermore, a large b(i) implies that i is badly matched to its neighbouring cluster. Thus an s(i) close to one means that the data is appropriately clustered.
    S = {}
    for kappa_same in Cnew.keys():
        a = Dtemp.loc[Cnew[kappa_same],Cnew[kappa_same]].mean().mean()
        b = np.min([Dtemp.loc[Cnew[kappa_other],Cnew[kappa_same]].mean().mean() for kappa_other in Cnew.keys() if kappa_other!=kappa_same])
        S[kappa_same] = (b - a) / max(a, b)

    # return results
    return M, Cnew, S
