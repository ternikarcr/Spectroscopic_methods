# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 15:08:31 2025

@author: DNK
"""

####FINAL####
#1 #Import packages
import sys
from sys import stdout
import os
import glob
import pandas as pd
import numpy as np
from numpy.linalg import inv
from scipy.stats import chi2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import cohen_kappa_score
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from matplotlib import gridspec
import time
import pylab as pl
import seaborn as sns
import collections
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis()
from sklearn.svm import SVC
svm = SVC()
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
from sklearn.model_selection import LeaveOneOut
loo = LeaveOneOut()
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()


#2 #Data Import
#Set path
#path = 'D:/Academics/PhD/SEM 9/Paper_2_Soil_texture_quantitative/Working_Files/Codes_final'
path = 'C:/Refined/Draft_2/Working_files'
os.chdir(path)
file = 'Working_lab.csv'
#Data Import
df =  pd.read_csv(file)
dict = {' c':'Cl', ' cl':'ClLo', ' ls':'LoSa', ' sc':'SaCl', ' scl':'SaClLo', ' sl':'SaLo'}
df['Texture'] = df['Texture'].map(dict)
#Removed bad bands less than 400 nm wavelength
df = df.drop(df.iloc[:, 5:55], axis = 1)

path = 'C:/Refined/Draft_2/Working_files/Revision'
os.chdir(path)

#3 #Definition of various functions
#VIP scores calculation (adapted from https://github.com/scikit-learn/scikit-learn/issues/7050)
def vip_efficient(model):
    t = model.x_scores_
    w = model.x_weights_ # replace with x_rotations_ if needed
    q = model.y_loadings_ 
    features_, _ = w.shape
    vip = np.zeros(shape=(features_,))
    inner_sum = np.diag(t.T @ t @ q.T @ q)
    SS_total = np.sum(inner_sum)
    vip = np.sqrt(features_*(w**2 @ inner_sum)/ SS_total)
    return vip

def regression_metrics(y0, y0_p, y1, y1_p):
    ssr_c = np.sum(np.square(y0-y0_p))
    sst_c = np.sum(np.square(y0-np.mean(y0)))
    r2_c = 1-(ssr_c/sst_c)
    rmse_c = np.sqrt((np.sum(np.square(y0-y0_p)))/len(y0))
    sd_c = np.std(y0)
    iqr_c = np.percentile(y0,75,interpolation='midpoint') - np.percentile(y0,25,interpolation='midpoint')
    rpd_c = sd_c/rmse_c
    rpiq_c = iqr_c/rmse_c
    bias_c = np.mean(y0_p) - np.mean(y0)
    ssr_v = np.sum(np.square(y1-y1_p))
    sst_v = np.sum(np.square(y1-np.mean(y1)))
    r2_v = 1-(ssr_v/sst_v)
    rmse_v = np.sqrt((np.sum(np.square(y1-y1_p)))/len(y1))
    sd_v = np.std(y1)
    iqr_v = np.percentile(y1,75,interpolation='midpoint') - np.percentile(y1,25,interpolation='midpoint')
    rpd_v = sd_v/rmse_v
    rpiq_v = iqr_v/rmse_v
    bias_v = np.mean(y1_p) - np.mean(y1)
    metrics_final = np.array([r2_c, rmse_c, rpd_c, rpiq_c, bias_c, r2_v, rmse_v, rpd_v, rpiq_v, bias_v])
    return metrics_final

def optimise_pls2_cv(X0, y0, X1, y1, n_comp):
    mse = []
    n_comp_final = []
    component = np.arange(1, n_comp)
    for i in component:
        pls = PLSRegression(n_components=i, scale=False, copy=False)
        y0_p_cv = cross_val_predict(pls, X0, y0, cv=10)
        mse.append(mean_squared_error(y0[:,0], y0_p_cv[:,0]) + mean_squared_error(y0[:,1], y0_p_cv[:,1]) + mean_squared_error(y0[:,2], y0_p_cv[:,2]))
        comp = 100*(i+1)/n_comp
        stdout.write("\r%d%% completed" % comp)
        stdout.flush()
    stdout.write("\n")
    #Calculate and print the position of minimum in MSE
    msemin = np.argmin(mse)
    n_comp_final = msemin+1
    print("Suggested number of components: ", n_comp_final)
    #Define PLS object with optimal number of components
    pls_opt = PLSRegression(n_components=n_comp_final, scale=False)
    #Fit to the entire calibration dataset
    pls_opt.fit(X0, y0)
    vip_scores = vip_efficient(pls_opt)
    y0_p = pls_opt.predict(X0)
    y0_p = np.clip(y0_p, a_min = 0, a_max = 100)
    y0_foo = y0[:,0]
    y0_p_foo = y0_p[:,0]
    #Fit to the entire validation dataset
    y1_p = pls_opt.predict(X1)
    y1_p = np.clip(y1_p, a_min = 0, a_max = 100)
    y1_foo = y1[:,0] 
    y1_p_foo = y1_p[:,0] 
    #Calculate metrics
    metrics_pls2_clay = regression_metrics(y0_foo, y0_p_foo, y1_foo, y1_p_foo)
    y0_foo = y0[:,1]
    y0_p_foo = y0_p[:,1]
    y1_foo = y1[:,1] 
    y1_p_foo = y1_p[:,1] 
    metrics_pls2_silt = regression_metrics(y0_foo, y0_p_foo, y1_foo, y1_p_foo)
    y0_foo = y0[:,2]
    y0_p_foo = y0_p[:,2]
    y1_foo = y1[:,2] 
    y1_p_foo = y1_p[:,2] 
    metrics_pls2_sand = regression_metrics(y0_foo, y0_p_foo, y1_foo, y1_p_foo)
    return metrics_pls2_clay, metrics_pls2_silt, metrics_pls2_sand, y1_p[:,0], y1_p[:,1], y1_p[:,2], y1[:,0], y1[:,1], y1[:,2], vip_scores

#Classification
def classification_metrics(Y_train2, Y_train_pred, Y_test2, Y_test_pred):
    tr_oa, t_oa, tr_aa, t_aa, tr_k, t_k, tr_na, t_na, tr_ana, t_ana = [[] for _ in range(10)]
    seq_texture = ['Sa', 'LoSa', 'SaLo', 'SaClLo', 'SaCl', 'Cl', 'ClLo', 'Lo', 'SiCl', 'SiClLo', 'SiLo', 'Si']
    seq = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    neigh1 = np.array([[0, 1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
    [1,	0,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0],
    [0,	1,	0,	1,	0,	0,	0,	1,	0,	0,	1,	0],
    [0,	0,	1,	0,	1,	0,	1,	1,	0,	0,	0,	0],
    [0,	0,	0,	1,	0,	1,	1,	0,	0,	0,	0,	0],
    [0,	0,	0,	0,	1,	0,	1,	0,	1,	1,	0,	0],
    [0,	0,	0,	1,	1,	1,	0,	1,	1,	1,	1,	0],
    [0,	0,	1,	1,	0,	0,	1,	0,	0,	0,	1,	0],
    [0,	0,	0,	0,	0,	1,	1,	0,	0,	1,	0,	0],
    [0,	0,	0,	0,	0,	1,	1,	0,	1,	0,	1,	0],
    [0,	0,	1,	0,	0,	0,	1,	1,	0,	1,	0,	1],
    [0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0]])
    neigh2 = np.array([[1, 1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
    [1, 1,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0],
    [0,	1,	1,	1,	0,	0,	0,	1,	0,	0,	1,	0],
    [0,	0,	1,	1,	1,	0,	1,	1,	0,	0,	0,	0],
    [0,	0,	0,	1,	1,	1,	1,	0,	0,	0,	0,	0],
    [0,	0,	0,	0,	1,	1,	1,	0,	1,	1,	0,	0],
    [0,	0,	0,	1,	1,	1,	1,	1,	1,	1,	1,	0],
    [0,	0,	1,	1,	0,	0,	1,	1,	0,	0,	1,	0],
    [0,	0,	0,	0,	0,	1,	1,	0,	1,	1,	0,	0],
    [0,	0,	0,	0,	0,	1,	1,	0,	1,	1,	1,	0],
    [0,	0,	1,	0,	0,	0,	1,	1,	0,	1,	1,	1],
    [0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	1]])
    tr_oa = accuracy_score(Y_train2, Y_train_pred)
    tr_aa = balanced_accuracy_score(Y_train2, Y_train_pred)
    tr_k = cohen_kappa_score(Y_train2, Y_train_pred)
    tr_con = confusion_matrix(Y_train2, Y_train_pred, labels = seq_texture)
    tr_na = (np.sum(tr_con * neigh1))/np.sum(tr_con)
    tr_ana = (np.sum(tr_con * neigh2))/np.sum(tr_con)
    t_oa = accuracy_score(Y_test2, Y_test_pred)
    t_aa = balanced_accuracy_score(Y_test2, Y_test_pred)
    t_k = cohen_kappa_score(Y_test2, Y_test_pred)
    t_con = confusion_matrix(Y_test2, Y_test_pred, labels = seq_texture)
    t_na = (np.sum(t_con * neigh1))/np.sum(t_con)
    t_ana = (np.sum(t_con * neigh2))/np.sum(t_con)
    metrics = []
    metrics = np.array([tr_oa, tr_aa, tr_k, tr_na, tr_ana, t_oa, t_aa, t_k, t_na, t_ana])
    return metrics, tr_con, t_con

# def optimise_pls_cv2(X0, y0, n_comp):
#     ms_tr_aa = []
#     component = np.arange(1, n_comp)
#     for i in component:
#         pls = PLSRegression(n_components=i, scale=False, copy=False)
#         y0_p_cv = cross_val_predict(pls, X0, y0, cv=10)
#         dict = {0:'Cl', 1:'ClLo', 2:'LoSa', 3:'SaCl', 4:'SaClLo', 5:'SaLo'}
#         y01 = pd.DataFrame(np.argmax(y0.to_numpy(dtype='float64'), axis = 1))[0].map(dict)
#         y0_p_cv1 = pd.DataFrame(np.argmax(y0_p_cv, axis = 1))[0].map(dict)
#         ms_tr_aa.append(balanced_accuracy_score(y01, y0_p_cv1))
#         comp = 100*(i+1)/n_comp
#         stdout.write("\r%d%% completed" % comp)
#         stdout.flush()
#     stdout.write("\n")
#     #Calculate and print the position of minimum in MSE
#     msemax = np.argmax(ms_tr_aa)
#     n_comp_final = msemax+1
#     return n_comp_final

#Classification functions
#PLS-DA
def c_plsda(X_train2, Y_train2, X_test2, Y_test2, n_comp):
    ce = []
    n_comp_final = []
    component = np.arange(1, n_comp)
    Y_train2 = pd.get_dummies(Y_train2)
    dict = {0:'Cl', 1:'ClLo', 2:'LoSa', 3:'SaCl', 4:'SaClLo', 5:'SaLo'}
    for i in component:
        pls_opt = PLSRegression(n_components=i, scale=False)
        #Predicting and using discriminant analysis i.e. argmax function for assigning the class
        Y_train2_pred_cv = cross_val_predict(pls_opt, X_train2, Y_train2, cv=50)
        Y_train2_pred  = pd.DataFrame(np.argmax(Y_train2_pred_cv, axis = 1))[0].map(dict)
        Y_train2_act  = pd.DataFrame(np.argmax(Y_train2, axis = 1))[0].map(dict)
        #Y_train_act = [dict[key] for key in Y_train_act_foo]
        tr_oa = accuracy_score(Y_train2_pred, Y_train2_act)
        ce.append(tr_oa)
        comp = 100*(i+1)/n_comp
        stdout.write("\r%d%% completed" % comp)
        stdout.flush()
    stdout.write("\n")
    #Calculate and print the position of minimum in MSE
    cemax = np.argmax(ce)
    n_comp_final = cemax+1
    print("Suggested number of components: ", n_comp_final)
    #Define PLS object with optimal number of components
    pls_opt = PLSRegression(n_components=n_comp_final, scale=False)
    pls_opt.fit(X_train2, Y_train2)
    #VIP score calculation
    vip_scores = vip_efficient(pls_opt)
    Y_train_pred = pd.DataFrame(np.argmax(pls_opt.predict(X_train2), axis = 1))[0].map(dict)
    Y_test_pred = pd.DataFrame(np.argmax(pls_opt.predict(X_test2), axis = 1))[0].map(dict)
    Y_train2 = pd.DataFrame(np.argmax(Y_train2, axis = 1))[0].map(dict)
    Y_test2 = Y_test2.map(dict)
    metrics, tr_con, t_con = classification_metrics(Y_train2, Y_train_pred, Y_test2, Y_test_pred)
    return vip_scores, metrics, tr_con, t_con


# #PLS-DA
# def c_plsda(X_train2, Y_train2, X_test2, Y_test2, n_comp):
#     ce = []
#     n_comp_final = []
#     component = np.arange(1, n_comp)
#     Y_train3 = pd.get_dummies(Y_train2)
#     dict = {0:'Cl', 1:'ClLo', 2:'LoSa', 3:'SaCl', 4:'SaClLo', 5:'SaLo'}
#     for i in component:
#         pls_opt = PLSRegression(n_components=i, scale=False)
#         Y_train_pred = []
#         Y_train_act = []
#         Y_train_act_foo = []
#         #y0_p_cv = pls_opt.fit(X_train2, Y_train3)
#         #Predicting and using discriminant analysis i.e. argmax function for assigning the class
#         for train_index, test_index in loo.split(X_train2):
#             # Separate data for training and testing in each iteration
#             X_train11, X_test11 = X_train2[train_index], X_train2[test_index]
#             Y_train11, Y_test11 = Y_train3.iloc[train_index], Y_train3.iloc[test_index]
#             pls_opt.fit(X_train11, Y_train11)
#             Y_train_pred_foo = pd.DataFrame(np.argmax(pls_opt.predict(X_test11), axis = 1))[0].map(dict)
#             Y_train_pred.append(Y_train_pred_foo[0])
#             Y_train_act_foo.append(np.argmax(Y_test11))
#             Y_train_act = [dict[key] for key in Y_train_act_foo]
#         tr_oa = accuracy_score(Y_train_pred, Y_train_act)
#         ce.append(tr_oa)
#         comp = 100*(i+1)/n_comp
#         stdout.write("\r%d%% completed" % comp)
#         stdout.flush()
#     stdout.write("\n")
#     #Calculate and print the position of minimum in MSE
#     cemax = np.argmax(ce)
#     n_comp_final = cemax+1
#     print("Suggested number of components: ", n_comp_final)
#     #Define PLS object with optimal number of components
#     pls_opt = PLSRegression(n_components=n_comp_final, scale=False)
#     pls_opt.fit(X_train2, Y_train3)
#     #VIP score calculation
#     vip_scores = vip_efficient(pls_opt)
#     Y_train_pred = pd.DataFrame(np.argmax(pls_opt.predict(X_train2), axis = 1))[0].map(dict)
#     Y_test_pred = pd.DataFrame(np.argmax(pls_opt.predict(X_test2), axis = 1))[0].map(dict)
#     metrics, tr_con, t_con = classification_metrics(Y_train2, Y_train_pred, Y_test2, Y_test_pred)
#     return vip_scores, metrics, tr_con, t_con


#Initializing Empty variables
metrics_pls2_clay = np.empty([10,100])
metrics_pls2_silt = np.empty([10,100])
metrics_pls2_sand = np.empty([10,100])
Y_p_iter_pls2_clay = np.empty([69,100])
Y_p_iter_pls2_silt = np.empty([69,100])
Y_p_iter_pls2_sand = np.empty([69,100])
Y_iter_pls2_clay = np.empty([69,100])
Y_iter_pls2_silt = np.empty([69,100])
Y_iter_pls2_sand = np.empty([69,100])
vip_pls2_final = np.empty([2050,100])

vip_clay_final = np.empty([2050,100])
vip_silt_final = np.empty([2050,100])
vip_sand_final = np.empty([2050,100])

vip_plsda_final = np.empty([2050,100])
metrics_classification_plsda = np.empty([10,100])
tr_con_final_plsda = []
t_con_final_plsda = []

#Iterations of Main code
X = -np.log10(df.drop(['Sample_Code', 'Sand', 'Clay', 'Silt', 'Texture'], axis = 1))
Y = pd.DataFrame(df, columns= ['Clay', 'Silt', 'Sand', 'Texture'])
Y['Texture'] = label_encoder.fit_transform(Y['Texture'])
start_time = time.time()
for i in range(100):
    ##Regression
    X_train, X_test, Y_train_original, Y_test_original = train_test_split(X, Y, stratify=Y['Texture'])
    Y_train = Y_train_original[['Clay', 'Silt', 'Sand']].to_numpy()
    Y_test = Y_test_original[['Clay', 'Silt', 'Sand']].to_numpy()
    Y_train_class = Y_train_original['Texture']
    Y_test_class = Y_test_original['Texture']
    #Data preprocessing, Outlier detection
    pca = PCA(n_components=3)
    S = pca.fit_transform(X_train)
    cov = np.cov(S.T)
    i_cov = inv(cov)
    left = np.dot(S, i_cov)
    dist_mat = np.dot(left, S.T)
    mahal_dist = pd.DataFrame(np.sqrt(dist_mat.diagonal()))
    mahal_dist.columns = ['MD']
    #Setting index same as X_train
    mahal_dist.set_index(X_train.index, inplace=True)
    X_train1 = X_train[mahal_dist['MD']<3.5]
    Y_train1 = Y_train[mahal_dist['MD']<3.5]
    Y_train1_class = Y_train_class[mahal_dist['MD']<3.5]
    #PLS2R regression
    scaler = StandardScaler()
    X_train2 = scaler.fit_transform(X_train1)
    X_test2 = scaler.fit_transform(X_test)
    Y_train2 = Y_train1
    Y_test2 = Y_test
    metrics_pls2_clay[:,i], metrics_pls2_silt[:,i], metrics_pls2_sand[:,i], Y_p_iter_pls2_clay[:,i], Y_p_iter_pls2_silt[:,i], Y_p_iter_pls2_sand[:,i], Y_iter_pls2_clay[:,i], Y_iter_pls2_silt[:,i], Y_iter_pls2_sand[:,i], vip_pls2_final[:,i] = optimise_pls2_cv(X_train2, Y_train2, X_test2, Y_test2, n_comp = 40)    
    # #Classification
    # # Resampling: Apply SMOTE and Tomek Links
    # smote_tomek = SMOTETomek(sampling_strategy="auto", smote=SMOTE(k_neighbors=2))
    # X_train_resampled, Y_train_resampled_class = smote_tomek.fit_resample(X_train1, Y_train1_class)
    # # Check class distribution after resampling
    # print(f"Class distribution after resampling: {Counter(Y_train_resampled_class)}")
    # scaler = StandardScaler()
    # X_train2 = scaler.fit_transform(X_train_resampled)
    # X_test2 = scaler.fit_transform(X_test)
    # Y_train2_class = Y_train_resampled_class
    # Y_test2_class = Y_test_class
    # #PLSDA Classification
    # vip_plsda_final[:,i], metrics_classification_plsda[:,i], tr_con, t_con = c_plsda(X_train2, Y_train2_class, X_test2, Y_test2_class, n_comp = 40)
    # tr_con_final_plsda.append(tr_con)    
    # t_con_final_plsda.append(t_con)    
    print(i)
end_time = time.time()
#total time taken
req_time = (end_time - start_time) / 60
print("Time required was {} minutes".format(req_time))

pd.DataFrame(Y_p_iter_pls2_clay).to_csv('Y_p_iter_pls2_clay.csv', index = False)
pd.DataFrame(Y_p_iter_pls2_silt).to_csv('Y_p_iter_pls2_silt.csv', index = False)
pd.DataFrame(Y_p_iter_pls2_sand).to_csv('Y_p_iter_pls2_sand.csv', index = False)
pd.DataFrame(Y_iter_pls2_clay).to_csv('Y_iter_pls2_clay.csv', index = False)
pd.DataFrame(Y_iter_pls2_silt).to_csv('Y_iter_pls2_silt.csv', index = False)
pd.DataFrame(Y_iter_pls2_sand).to_csv('Y_iter_pls2_sand.csv', index = False)

####Outputs from R soiltexture classification####
list_iter = []
for i in range(100):
    name = 'iter_tt_' + str(i+1) + '.csv'
    list_iter.append(name)

#Metrics sequence (OA,AA,K,NA,ANA)
metrics_classification = np.empty([5,100])
tr_con_final = []
t_con_final = []

for i in range(100):
    tr_oa, t_oa, tr_aa, t_aa, tr_k, t_k, tr_na, t_na, tr_ana, t_ana = [[] for _ in range(10)]
    df =  pd.read_csv(list_iter[i])
    #Testing
    act = []
    pred = []
    act = df['actual']
    pred = df['pred']
    t_oa = accuracy_score(act, pred)
    t_aa = balanced_accuracy_score(act, pred)
    t_k = cohen_kappa_score(act, pred)
    t_con = confusion_matrix(act, pred, labels = seq)
    t_na = (np.sum(t_con * neigh1))/np.sum(t_con)
    t_ana = (np.sum(t_con * neigh2))/np.sum(t_con)
    t_con_final.append(t_con)
    metrics = []
    metrics = np.array([t_oa, t_aa, t_k, t_na, t_ana])
    metrics_classification[:,i] = metrics
    print(i)

np.mean(metrics_classification, axis=1)
pd.DataFrame(metrics_classification).to_csv('metrics_pls2r_classification .csv', index = False)


pd.DataFrame(metrics_classification_plsda).to_csv('metrics_classification_plsda.csv', index = False)
pd.DataFrame(vip_plsda_final).to_csv('vip_plsda_final.csv', index = False)

#Writing out the confusion matrix
seq_texture = ['Sa', 'LoSa', 'SaLo', 'SaClLo', 'SaCl', 'Cl', 'ClLo', 'Lo', 'SiCl', 'SiClLo', 'SiLo', 'Si']
seq = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
# tr_con_list = [tr_con_final_plsda]
# t_con_list = [t_con_final_plsda]
# classifier_seq = ['PLS-DA']
tr_con_list = [tr_con_final]
t_con_list = [t_con_final]
classifier_seq = ['PLS2R']
stdout_fileinfo = sys.stdout
sys.stdout = open('Confusion_Matrices.txt','a')
print('Sequence-Mean, Std deviation')
##Mean Confusion metrics
for i in range(1):
    print(classifier_seq[0])
    print('Training')
    print(seq_texture)
    print(seq)
    #Training
    con_tr_mean = np.around(np.mean(tr_con_list[i], axis=0))
    con_tr_std = np.around(np.std(tr_con_list[i], axis=0))
    print(con_tr_mean)
    print(con_tr_std)
    #Testing
    print('Testing')
    con_t_mean = np.around(np.mean(t_con_list[i], axis=0))
    con_t_std = np.around(np.std(t_con_list[i], axis=0))
    print(con_t_mean)
    print(con_t_std)
sys.stdout.close()
sys.stdout = stdout_fileinfo
i


##Rough
seq = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
neigh1 = np.array([[0, 1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
[1,	0,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0],
[0,	1,	0,	1,	0,	0,	0,	1,	0,	0,	1,	0],
[0,	0,	1,	0,	1,	0,	1,	1,	0,	0,	0,	0],
[0,	0,	0,	1,	0,	1,	1,	0,	0,	0,	0,	0],
[0,	0,	0,	0,	1,	0,	1,	0,	1,	1,	0,	0],
[0,	0,	0,	1,	1,	1,	0,	1,	1,	1,	1,	0],
[0,	0,	1,	1,	0,	0,	1,	0,	0,	0,	1,	0],
[0,	0,	0,	0,	0,	1,	1,	0,	0,	1,	0,	0],
[0,	0,	0,	0,	0,	1,	1,	0,	1,	0,	1,	0],
[0,	0,	1,	0,	0,	0,	1,	1,	0,	1,	0,	1],
[0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0]])
neigh2 = np.array([[1, 1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
[1, 1,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0],
[0,	1,	1,	1,	0,	0,	0,	1,	0,	0,	1,	0],
[0,	0,	1,	1,	1,	0,	1,	1,	0,	0,	0,	0],
[0,	0,	0,	1,	1,	1,	1,	0,	0,	0,	0,	0],
[0,	0,	0,	0,	1,	1,	1,	0,	1,	1,	0,	0],
[0,	0,	0,	1,	1,	1,	1,	1,	1,	1,	1,	0],
[0,	0,	1,	1,	0,	0,	1,	1,	0,	0,	1,	0],
[0,	0,	0,	0,	0,	1,	1,	0,	1,	1,	0,	0],
[0,	0,	0,	0,	0,	1,	1,	0,	1,	1,	1,	0],
[0,	0,	1,	0,	0,	0,	1,	1,	0,	1,	1,	1],
[0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	1]])