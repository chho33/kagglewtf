from lightgbm import LGBMRegressor,LGBMClassifier
from xgboost import XGBRegressor,XGBClassifier
import pandas as pd
import os
import time
from mlxtend.regressor import StackingRegressor
from mlxtend.classifier import StackingClassifier
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
from sklearn.metrics import mean_squared_error,roc_auc_score
from sklearn.svm import SVR
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.preprocessing import StandardScaler,Normalizer,MinMaxScaler,LabelEncoder,Imputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_extraction.text import TfidfVectorizer,TfidfTransformer,HashingVectorizer
#import matplotlib.pyplot as plt
import copy
import numpy as np
from collections import Counter
import logging
import warnings
warnings.filterwarnings('ignore')

def delete_cols(df):
    return df.drop(cols_del,axis=1)

def map_cols(df):
    for col,vals in dict_grade.items():
        df[col] = df[col].map(vals)
        df[col] = df[col].fillna(0)
    return df

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

def best_gmm_cluster(data,n_components_range=range(6, 20)):
    logging.basicConfig(filename='log/gmm.log',level=logging.DEBUG)
    lowest_bic = np.infty
    bic = []
    #n_components_range = range(6, 20)
    cv_types = ['spherical', 'tied', 'diag', 'full']
    for cv_type in cv_types:
        for n_components in n_components_range:
            print('n_components: ',n_components)
            a = time.time()
            # Fit a mixture of Gaussians with EM
            gmm = GaussianMixture(n_components=n_components,covariance_type=cv_type,warm_start=True)
            gmm.fit(data)
            aic_tmp = gmm.aic(data)
            bic.append(aic_tmp)
            print('aic: ',aic_tmp)
            logging.info('cv_type: %s, n_components : %s, aic: %s'%(cv_type, n_components,aic_tmp))
            print('cost time: ', time.time() - a)
            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
                best_gmm = gmm
    return best_gmm

# variance
def remain_high_variance(data,return_df=False,threshold=0.15,show_remain=True):
    sel = VarianceThreshold(threshold=threshold)
    result = sel.fit_transform(normit(data,True))
    cols = data.columns
    remain_cols = np.array(cols)[sel.get_support()]
    del_cols = np.array(cols)[np.logical_not(sel.get_support())]
    if show_remain:
        print ('remain cols: ', remain_cols )
    print ('del cols: ', del_cols )
    if return_df: return data[remain_cols]
    return result 

# normalization
def normit(data,return_df=False):
    if return_df:
        cols = data.columns 
    scaler = StandardScaler()
    scaler.fit(data)
    data = scaler.transform(data)
    if return_df:
        data = pd.DataFrame(data)
        data.columns = cols
    return data

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

def find_correlation_to_remove(data, threshold=0.95):
    """
    Given a numeric pd.DataFrame, this will find highly correlated features,
    and return a list of features to remove.
    Parameters
    -----------
    data : pandas DataFrame
        DataFrame
    threshold : float
        correlation threshold, will remove one of pairs of features with a
        correlation greater than this value
    Returns
    --------
    select_flat : list
        listof column names to be removed
    """
    corr_mat = data.corr()
    corr_mat.loc[:, :] = np.tril(corr_mat, k=-1)
    already_in = set()
    result = []
    for col in corr_mat:
        perfect_corr = corr_mat[col][corr_mat[col] > threshold].index.tolist()
        if perfect_corr and col not in already_in:
            already_in.update(set(perfect_corr))
            perfect_corr.append(col)
            result.append(perfect_corr)
    select_nested = [f[1:] for f in result]
    select_flat = [i for j in select_nested for i in j]
    return select_flat

def get_lowcorr_df(dataset, threshold=0.95):
    col_corr = set() # Set of all the names of deleted columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if corr_matrix.iloc[i, j] >= threshold:
                colname = corr_matrix.columns[i] # getting the name of column
                col_corr.add(colname)
                if colname in dataset.columns:
                    dataset = dataset.drop(colname,axis=1)
    return dataset

def drop_if_all_same_vals(df):
    del_cols = []
    for col, vals in df.items():
        vals = set(vals)
        if len(vals)==1:
            del_cols.append(col)
    print('drop all same val cols: ',del_cols)
    return df.drop(del_cols,axis=1)

def df_to_dict(df,orient='records'):
    return df.to_dict(orient=orient)
