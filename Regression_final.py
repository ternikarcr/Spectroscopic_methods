#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 18:09:00 2023

@author: chirag
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

#2 #Data Import
#Set path
#path = 'D:/Academics/PhD/SEM 9/Paper_2_Soil_texture_quantitative/Working_Files/Codes_final'
path = 'C:/Users/DNK/Downloads/Foo/Draft_2'
os.chdir(path)
file = 'Working_lab.csv'
#Data Import
df =  pd.read_csv(file)
dict = {' c':'Cl', ' cl':'ClLo', ' ls':'LoSa', ' sc':'SaCl', ' scl':'SaClLo', ' sl':'SaLo'}
df['Texture'] = df['Texture'].map(dict)
#Removed bad bands less than 400 nm wavelength
df = df.drop(df.iloc[:, 5:55], axis = 1)


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

#PLS function
def optimise_pls_cv(X0, y0, X1, y1, n_comp, plot_components):
    mse = []
    n_comp_final = []
    component = np.arange(1, n_comp)
    loocv = y0.shape[0] - 1
    for i in component:
        pls = PLSRegression(n_components=i, scale=False, copy=False)
        y0_p_cv = cross_val_predict(pls, X0, y0, cv=loocv)
        mse.append(mean_squared_error(y0, y0_p_cv))
        comp = 100*(i+1)/n_comp
        stdout.write("\r%d%% completed" % comp)
        stdout.flush()
    stdout.write("\n")
    #Calculate and print the position of minimum in MSE
    msemin = np.argmin(mse)
    n_comp_final = msemin+1
    #print("Suggested number of components: ", n_comp_final)
    stdout.write("\n")
    if plot_components is True:
        with plt.style.context(('ggplot')):
            plt.plot(component, np.array(mse), '-v', color = 'blue', mfc='blue')
            plt.plot(component[msemin], np.array(mse)[msemin], 'P', ms=10, mfc='red')
            plt.xlabel('Number of PLS components')
            plt.ylabel('MSE')
            plt.title('PLS')
            plt.xlim(left=-1)
        plt.show()
    #Define PLS object with optimal number of components
    pls_opt = PLSRegression(n_components=n_comp_final, scale=False)
    #Fit to the entire calibration dataset
    pls_opt.fit(X0, y0)
    y0_p = pls_opt.predict(X0)
    y0_p = np.clip(y0_p, a_min = 0, a_max = 100)
    # Cross-validation
    y0_p_cv = cross_val_predict(pls_opt, X0, y0, cv=loocv)
    y0_p_cv = np.clip(y0_p_cv, a_min = 0, a_max = 100)
    #Formatting for easy calculations
    y0_p = pd.Series(y0_p[:,0], index=y0.index)
    y0_p_cv = pd.Series(y0_p_cv[:,0], index=y0.index)
    #Calculate scores for calibration and cross-validation
    score0_p = r2_score(y0, y0_p)
    score0_p_cv = r2_score(y0, y0_p_cv)
    #VIP score calculation
    vip_scores = vip_efficient(pls_opt)
    #Plot regression and figures of merit
    rangey = max(y0) - min(y0)
    rangex = max(y0_p) - min(y0_p)
    #Fit a line to the CV vs response
    if plot_components is True:
        z = np.polyfit(y0, y0_p, 1)
        with plt.style.context(('ggplot')):
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.scatter(y0_p, y0, c='red', edgecolors='k')
            #Plot the best fit line
            ax.plot(np.polyval(z,y0), y0, c='blue', linewidth=1)
            #Plot the ideal 1:1 line
            ax.plot(y0, y0, color='green', linewidth=1)
            plt.title('$R^{2}$ (CV): '+str(score0_p_cv))
            plt.xlabel('Predicted')
            plt.ylabel('Measured')
            plt.show()
    #Calculate root mean squared error for calibration and cross validation
    ssr_cv = np.sum(np.square(y0-y0_p_cv))
    sst_cv = np.sum(np.square(y0-np.mean(y0)))
    r2_cv = 1-(ssr_cv/sst_cv)
    rmse_cv = np.sqrt((np.sum(np.square(y0-y0_p_cv)))/len(y0))
    ssr_c = np.sum(np.square(y0-y0_p))
    sst_c = np.sum(np.square(y0-np.mean(y0)))
    r2_c = 1-(ssr_c/sst_c)
    rmse_c = np.sqrt((np.sum(np.square(y0-y0_p)))/len(y0))
    #Calculate rpd, rpiq, bias for calibration 
    sd_c = np.std(y0)
    iqr_c = np.percentile(y0,75,interpolation='midpoint') - np.percentile(y0,25,interpolation='midpoint')
    rpd_c = sd_c/rmse_c
    rpiq_c = iqr_c/rmse_c
    bias_c = np.mean(y0_p) - np.mean(y0)
    #Fit to the entire validation dataset
    y1_p = pls_opt.predict(X1)
    #Formatting for easy calculations
    y1_p = pd.Series(y1_p[:,0], index=y1.index)
    #Calculate scores for Validation 
    score1_p = r2_score(y1, y1_p)
    #Calculate root mean squared error for calibration and cross validation
    ssr_v = np.sum(np.square(y1-y1_p))
    sst_v = np.sum(np.square(y1-np.mean(y1)))
    r2_v = 1-(ssr_v/sst_v)
    rmse_v = np.sqrt((np.sum(np.square(y1-y1_p)))/len(y1))
    #Calculate rpd, rpiq, bias for validation
    sd_v = np.std(y1)
    iqr_v = np.percentile(y1,75,interpolation='midpoint') - np.percentile(y1,25,interpolation='midpoint')
    rpd_v = sd_v/rmse_v
    rpiq_v = iqr_v/rmse_v
    bias_v = np.mean(y1_p) - np.mean(y1)
    metrics_final = np.array([n_comp_final, r2_cv, rmse_cv, r2_c, rmse_c, rpd_c, rpiq_c, bias_c, r2_v, rmse_v, rpd_v, rpiq_v, bias_v])
    return vip_scores, metrics_final, y0_p, y1_p

def optimise_pls_cv1(X0, y0, X1, y1, n_comp):
    mse = []
    n_comp_final = []
    component = np.arange(1, n_comp)
    loocv = y0.shape[0] - 1
    for i in component:
        pls = PLSRegression(n_components=i, scale=False, copy=False)
        y0_p_cv = cross_val_predict(pls, X0, y0, cv=loocv)
        mse.append(mean_squared_error(y0, y0_p_cv))
        comp = 100*(i+1)/n_comp
        stdout.write("\r%d%% completed" % comp)
        stdout.flush()
    stdout.write("\n")
    #Calculate and print the position of minimum in MSE
    msemin = np.argmin(mse)
    n_comp_final = msemin+1
    #print("Suggested number of components: ", n_comp_final)
    stdout.write("\n")
    #Define PLS object with optimal number of components
    pls_opt = PLSRegression(n_components=n_comp_final, scale=False)
    #Fit to the entire calibration dataset
    pls_opt.fit(X0, y0)
    y0_p = pls_opt.predict(X0)
    #Formatting for easy calculations
    y0_p = pd.Series(y0_p[:,0], index=y0.index)
    #VIP score calculation
    vip_scores = vip_efficient(pls_opt)
    #Fit to the entire validation dataset
    y1_p = pls_opt.predict(X1)
    #Formatting for easy calculations
    y1_p = pd.Series(y1_p[:,0], index=y1.index)
    return vip_scores, y0_p, y1_p

def log_ratio_pred(y_c_si_cl, y_c_sa_cl):
    y_silt_pred = 100 * np.exp(y_c_si_cl)/(np.exp(y_c_si_cl) + np.exp(y_c_sa_cl) +1)
    y_sand_pred = 100 * np.exp(y_c_sa_cl)/(np.exp(y_c_si_cl) + np.exp(y_c_sa_cl) +1)
    y_clay_pred = 100 * 1/(np.exp(y_c_si_cl) + np.exp(y_c_sa_cl) +1)
    return y_clay_pred, y_silt_pred, y_sand_pred

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

def optimise_pls_cv2(X0, y0, n_comp):
    ms_tr_aa = []
    component = np.arange(1, n_comp)
    loocv = y0.shape[0] - 1
    for i in component:
        pls = PLSRegression(n_components=i, scale=False, copy=False)
        y0_p_cv = cross_val_predict(pls, X0, y0, cv=loocv)
        dict = {0:'Cl', 1:'ClLo', 2:'LoSa', 3:'SaCl', 4:'SaClLo', 5:'SaLo'}
        y01 = pd.DataFrame(np.argmax(y0.to_numpy(dtype='float64'), axis = 1))[0].map(dict)
        y0_p_cv1 = pd.DataFrame(np.argmax(y0_p_cv, axis = 1))[0].map(dict)
        ms_tr_aa.append(balanced_accuracy_score(y01, y0_p_cv1))
        comp = 100*(i+1)/n_comp
        stdout.write("\r%d%% completed" % comp)
        stdout.flush()
    stdout.write("\n")
    #Calculate and print the position of minimum in MSE
    msemax = np.argmax(ms_tr_aa)
    n_comp_final = msemax+1
    return n_comp_final

#Classification functions
#PLS-DA
def c_plsda(X_train2, Y_train2, X_test2, Y_test2, n_comp, plot_components):
    tr_oa, t_oa, tr_aa, t_aa, tr_k, t_k, tr_na, t_na, tr_ana, t_ana = [[] for _ in range(10)]
    #PLS-DA
    #One-hot encoding for texture data
    ce = []
    n_comp_final = []
    component = np.arange(1, n_comp)
    Y_train3 = pd.get_dummies(Y_train2)
    for i in component:
        pls_opt = PLSRegression(n_components=i, scale=False)
        Y_train_pred = []
        Y_train_act = []
        Y_train_act_foo = []
        #y0_p_cv = pls_opt.fit(X_train2, Y_train3)
        #Predicting and using discriminant analysis i.e. argmax function for assigning the class
        dict = {0:'Cl', 1:'ClLo', 2:'LoSa', 3:'SaCl', 4:'SaClLo', 5:'SaLo'}
        for train_index, test_index in loo.split(X_train2):
            # Separate data for training and testing in each iteration
            X_train11, X_test11 = X_train2[train_index], X_train2[test_index]
            Y_train11, Y_test11 = Y_train3.iloc[train_index], Y_train3.iloc[test_index]
            pls_opt.fit(X_train11, Y_train11)
            Y_train_pred_foo = pd.DataFrame(np.argmax(pls_opt.predict(X_test11), axis = 1))[0].map(dict)
            Y_train_pred.append(Y_train_pred_foo[0])
            Y_train_act_foo.append(np.argmax(Y_test11))
            Y_train_act = [dict[key] for key in Y_train_act_foo]
        tr_oa = accuracy_score(Y_train_pred, Y_train_act)
        ce.append(tr_oa)
        comp = 100*(i+1)/n_comp
        stdout.write("\r%d%% completed" % comp)
        stdout.flush()
    stdout.write("\n")
    #Calculate and print the position of minimum in MSE
    cemax = np.argmax(ce)
    n_comp_final = cemax+1
    print("Suggested number of components: ", n_comp_final)
    if plot_components is True:
        with plt.style.context(('ggplot')):
            plt.plot(component, np.array(ce), '-v', color = 'blue', mfc='blue')
            plt.plot(component[cemax], np.array(ce)[cemax], 'P', ms=10, mfc='red')
            plt.xlabel('Number of PLS components')
            plt.ylabel('CE')
            plt.title('PLS')
            plt.xlim(left=-1)
        plt.show()
    #Define PLS object with optimal number of components
    pls_opt = PLSRegression(n_components=n_comp_final, scale=False)
    pls_opt.fit(X_train2, Y_train3)
    #VIP score calculation
    vip_scores = vip_efficient(pls_opt)
    #Predicting and using discriminant analysis i.e. argmax function for assigning the class
    dict = {0:'Cl', 1:'ClLo', 2:'LoSa', 3:'SaCl', 4:'SaClLo', 5:'SaLo'}
    Y_train_pred = pd.DataFrame(np.argmax(pls_opt.predict(X_train2), axis = 1))[0].map(dict)
    tr_oa = accuracy_score(Y_train2, Y_train_pred)
    tr_aa = balanced_accuracy_score(Y_train2, Y_train_pred)
    tr_k = cohen_kappa_score(Y_train2, Y_train_pred)
    tr_con = confusion_matrix(Y_train2, Y_train_pred, labels = seq_texture)
    tr_na = (np.sum(tr_con * neigh1))/np.sum(tr_con)
    tr_ana = (np.sum(tr_con * neigh2))/np.sum(tr_con)
    Y_test_pred = pd.DataFrame(np.argmax(pls_opt.predict(X_test2), axis = 1))[0].map(dict)
    t_oa = accuracy_score(Y_test2, Y_test_pred)
    t_aa = balanced_accuracy_score(Y_test2, Y_test_pred)
    t_k = cohen_kappa_score(Y_test2, Y_test_pred)
    t_con = confusion_matrix(Y_test2, Y_test_pred, labels = seq_texture)
    t_na = (np.sum(t_con * neigh1))/np.sum(t_con)
    t_ana = (np.sum(t_con * neigh2))/np.sum(t_con)
    metrics = []
    metrics = np.array([tr_oa, tr_aa, tr_k, tr_na, tr_ana, t_oa, t_aa, t_k, t_na, t_ana])
    return vip_scores, metrics, tr_con, t_con


#4 #Initializing Empty variables
vip_clay_final = np.empty([2050,100])
metrics_clay_final = np.empty([13,100])
Y_iter_clay = []
Y_p_iter_clay = []
vip_silt_final = np.empty([2050,100])
metrics_silt_final = np.empty([13,100])
Y_iter_silt = []
Y_p_iter_silt = []
vip_sand_final = np.empty([2050,100])
metrics_sand_final = np.empty([13,100])
Y_iter_sand = []
Y_p_iter_sand = []
train_nos = []
test_nos = []
vip_fraction_1_final = np.empty([2050,100])
vip_fraction_2_final = np.empty([2050,100])
metrics_clay_final_log_ratio = np.empty([10,100])
Y_iter_clay_log_ratio = []
Y_p_iter_clay_log_ratio = []
metrics_silt_final_log_ratio = np.empty([10,100])
Y_iter_silt_log_ratio = []
Y_p_iter_silt_log_ratio = []
metrics_sand_final_log_ratio = np.empty([10,100])
Y_iter_sand_log_ratio = []
Y_p_iter_sand_log_ratio = []

vip_plsda_final = np.empty([2050,100])
metrics_classification_plsda = np.empty([10,100])
tr_con_final_plsda = []
t_con_final_plsda = []

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


#5 #Iterations of Main code
X = -np.log10(df.drop(['Sample_Code', 'Sand', 'Clay', 'Silt', 'Texture'], axis = 1))
Y = pd.DataFrame(df, columns= ['Clay', 'Silt', 'Sand', 'Texture'])
#Log-ratio trick
Y['si_cl'] = np.log(Y['Silt']/Y['Clay'])
Y['sa_cl'] = np.log(Y['Sand']/Y['Clay'])

start_time = time.time()
for i in range(100):
    ##Regression
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, stratify=Y['Texture'])
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
    scaler = StandardScaler()
    X_train2 = scaler.fit_transform(X_train1)
    X_test2 = scaler.fit_transform(X_test)
    #Clay
    a = 0
    Y_train2 = Y_train1.iloc[:,a]
    Y_test2 = Y_test.iloc[:,a]
    vip_clay, metrics_clay, y_c_clay, y_v_clay = optimise_pls_cv(X0=X_train2, y0 = Y_train2, X1 = X_test2, y1 = Y_test2, n_comp = 40, plot_components=False)
    vip_clay_final[:,i] = vip_clay
    metrics_clay_final[:,i] = metrics_clay
    Y_iter_clay.append(pd.concat([Y_train2,Y_test2]))
    Y_p_iter_clay.append(pd.concat([y_c_clay,y_v_clay]))
    #Silt
    a = 1
    Y_train2 = Y_train1.iloc[:,a]
    Y_test2 = Y_test.iloc[:,a]
    vip_silt, metrics_silt, y_c_silt, y_v_silt = optimise_pls_cv(X0=X_train2, y0 = Y_train2, X1 = X_test2, y1 = Y_test2, n_comp = 40, plot_components=False)
    vip_silt_final[:,i] = vip_silt
    metrics_silt_final[:,i] = metrics_silt
    Y_iter_silt.append(pd.concat([Y_train2,Y_test2]))
    Y_p_iter_silt.append(pd.concat([y_c_silt,y_v_silt]))
    #Sand
    a = 2
    Y_train2 = Y_train1.iloc[:,a]
    Y_test2 = Y_test.iloc[:,a]
    vip_sand, metrics_sand, y_c_sand, y_v_sand = optimise_pls_cv(X0=X_train2, y0 = Y_train2, X1 = X_test2, y1 = Y_test2, n_comp = 40, plot_components=False)
    vip_sand_final[:,i] = vip_sand
    metrics_sand_final[:,i] = metrics_sand
    Y_iter_sand.append(pd.concat([Y_train2,Y_test2]))
    Y_p_iter_sand.append(pd.concat([y_c_sand,y_v_sand]))
    train_nos.append(Y_train2.size)
    test_nos.append(Y_test2.size)
    ##Log-ratio trick
    a = 4
    Y_train2 = Y_train1.iloc[:,a]
    Y_test2 = Y_test.iloc[:,a]
    vip_fraction_1, y_c_si_cl, y_v_si_cl = optimise_pls_cv1(X0=X_train2, y0 = Y_train2, X1 = X_test2, y1 = Y_test2, n_comp = 30)
    vip_fraction_1_final[:,i] = vip_fraction_1
    a = 5
    Y_train2 = Y_train1.iloc[:,a]
    Y_test2 = Y_test.iloc[:,a]
    vip_fraction_2, y_c_sa_cl, y_v_sa_cl = optimise_pls_cv1(X0=X_train2, y0 = Y_train2, X1 = X_test2, y1 = Y_test2, n_comp = 30)
    vip_fraction_2_final[:,i] = vip_fraction_2
    y_c_clay, y_c_silt, y_c_sand = log_ratio_pred(y_c_si_cl, y_c_sa_cl)
    y_v_clay, y_v_silt, y_v_sand = log_ratio_pred(y_v_si_cl, y_v_sa_cl)
    metrics_clay = regression_metrics(Y_train1.iloc[:,0], y_c_clay, Y_test.iloc[:,0], y_v_clay)
    metrics_clay_final_log_ratio[:,i] = metrics_clay
    metrics_silt = regression_metrics(Y_train1.iloc[:,1], y_c_silt, Y_test.iloc[:,1], y_v_silt)
    metrics_silt_final_log_ratio[:,i] = metrics_silt
    metrics_sand = regression_metrics(Y_train1.iloc[:,2], y_c_sand, Y_test.iloc[:,2], y_v_sand)
    metrics_sand_final_log_ratio[:,i] = metrics_sand
    Y_iter_clay_log_ratio.append(pd.concat([Y_train1.iloc[:,0],Y_test.iloc[:,0]]))
    Y_p_iter_clay_log_ratio.append(pd.concat([y_c_clay,y_v_clay]))
    Y_iter_silt_log_ratio.append(pd.concat([Y_train1.iloc[:,1],Y_test.iloc[:,1]]))
    Y_p_iter_silt_log_ratio.append(pd.concat([y_c_silt,y_v_silt]))
    Y_iter_sand_log_ratio.append(pd.concat([Y_train1.iloc[:,2],Y_test.iloc[:,2]]))
    Y_p_iter_sand_log_ratio.append(pd.concat([y_c_sand,y_v_sand]))    
    #Classification
    a = 3
    Y_train2 = Y_train1.iloc[:,a]
    Y_test2 = Y_test.iloc[:,a]
    #PLSDA
    vip_plsda, metrics, tr_con, t_con = c_plsda(X_train2, Y_train2, X_test2, Y_test2, n_comp = 40, plot_components=False)
    vip_plsda_final[:,i] = vip_plsda
    metrics_classification_plsda[:,i] = metrics
    tr_con_final_plsda.append(tr_con)    
    t_con_final_plsda.append(t_con)    
    print(i)
end_time = time.time()
#total time taken
req_time = (end_time - start_time) / 60
print("Time required was {} minutes".format(req_time))


#6
#Writing outputs
#Check
#foo = np.sum(metrics_silt_final[8,:]>0)
####Saving raw npy files####
np.save("metrics_clay_final",metrics_clay_final)
np.save("metrics_silt_final",metrics_silt_final)
np.save("metrics_sand_final",metrics_sand_final)

array_dict = {f'array_{idx}': arr for idx, arr in enumerate(Y_iter_clay)}
np.savez_compressed('Y_iter_clay.npz', **array_dict)
array_dict = {f'array_{idx}': arr for idx, arr in enumerate(Y_iter_silt)}
np.savez_compressed('Y_iter_silt.npz', **array_dict)
array_dict = {f'array_{idx}': arr for idx, arr in enumerate(Y_iter_sand)}
np.savez_compressed('Y_iter_sand.npz', **array_dict)
array_dict = {f'array_{idx}': arr for idx, arr in enumerate(Y_p_iter_clay)}
np.savez_compressed('Y_p_iter_clay.npz', **array_dict)
array_dict = {f'array_{idx}': arr for idx, arr in enumerate(Y_p_iter_silt)}
np.savez_compressed('Y_p_iter_silt.npz', **array_dict)
array_dict = {f'array_{idx}': arr for idx, arr in enumerate(Y_p_iter_sand)}
np.savez_compressed('Y_p_iter_sand.npz', **array_dict)
np.save("train_nos",train_nos)
np.save("test_nos",test_nos)
np.save("metrics_clay_final_log_ratio",metrics_clay_final_log_ratio)
np.save("metrics_silt_final_log_ratio",metrics_silt_final_log_ratio)
np.save("metrics_sand_final_log_ratio",metrics_sand_final_log_ratio)
np.save("Y_p_iter_log_ratio_clay",Y_p_iter_clay_log_ratio)
np.save("Y_p_iter_log_ratio_silt",Y_p_iter_silt_log_ratio)
np.save("Y_p_iter_log_ratio_sand",Y_p_iter_sand_log_ratio)
array_dict = {f'array_{idx}': arr for idx, arr in enumerate(Y_p_iter_clay_log_ratio)}
np.savez_compressed('Y_p_iter_clay_log_ratio.npz', **array_dict)
array_dict = {f'array_{idx}': arr for idx, arr in enumerate(Y_p_iter_silt_log_ratio)}
np.savez_compressed('Y_p_iter_silt_log_ratio.npz', **array_dict)
array_dict = {f'array_{idx}': arr for idx, arr in enumerate(Y_p_iter_sand_log_ratio)}
np.savez_compressed('Y_p_iter_sand_log_ratio.npz', **array_dict)
np.save("vip_clay_final",vip_clay_final)
np.save("vip_silt_final",vip_silt_final)
np.save("vip_sand_final",vip_sand_final)
np.save("vip_fraction_1_final",vip_fraction_1_final)
np.save("vip_fraction_2_final",vip_fraction_2_final)
np.save("vip_plsda_final",vip_plsda_final)
del array_dict


pd.DataFrame(metrics_clay_final).to_csv('metrics_clay_final.csv', index = False)
pd.DataFrame(metrics_silt_final).to_csv('metrics_silt_final.csv', index = False)
pd.DataFrame(metrics_sand_final).to_csv('metrics_sand_final.csv', index = False)
pd.DataFrame(metrics_clay_final_log_ratio).to_csv('metrics_clay_final_log_ratio.csv', index = False)
pd.DataFrame(metrics_silt_final_log_ratio).to_csv('metrics_silt_final_log_ratio.csv', index = False)
pd.DataFrame(metrics_sand_final_log_ratio).to_csv('metrics_sand_final_log_ratio.csv', index = False)
pd.DataFrame(vip_clay_final).to_csv('vip_clay_final.csv', index = False)
pd.DataFrame(vip_silt_final).to_csv('vip_silt_final.csv', index = False)
pd.DataFrame(vip_sand_final).to_csv('vip_sand_final.csv', index = False)
pd.DataFrame(vip_fraction_1_final).to_csv('vip_fraction_1_final.csv', index = False)
pd.DataFrame(vip_fraction_2_final).to_csv('vip_fraction_2_final.csv', index = False)
pd.DataFrame(vip_plsda_final).to_csv('vip_plsda_final.csv', index = False)

metrics_clay_final = np.load('metrics_clay_final.npy', allow_pickle = True)
metrics_silt_final = np.load('metrics_silt_final.npy', allow_pickle = True)
metrics_sand_final = np.load('metrics_sand_final.npy', allow_pickle = True)
metrics_clay_final = pd.DataFrame(metrics_clay_final)
metrics_silt_final = pd.DataFrame(metrics_silt_final)
metrics_sand_final = pd.DataFrame(metrics_sand_final)

#Classification Outputs
np.mean(metrics_classification_plsda, axis=1)
pd.DataFrame(metrics_classification_plsda).to_csv('metrics_classification_plsda.csv', index = False)
np.save("metrics_classification_plsda", metrics_classification_plsda)
np.save("tr_con_final_plsda", tr_con_final_plsda)
np.save("t_con_final_plsda", t_con_final_plsda)

#Writing out the confusion matrix
seq_texture = ['Sa', 'LoSa', 'SaLo', 'SaClLo', 'SaCl', 'Cl', 'ClLo', 'Lo', 'SiCl', 'SiClLo', 'SiLo', 'Si']
seq = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
tr_con_list = [tr_con_final_plsda]
t_con_list = [t_con_final_plsda]
classifier_seq = ['PLS-DA']
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

p.mean(metrics_clay_final, axis=1)
foo = np.mean(metrics_silt_final, axis=1)
foo = np.mean(metrics_sand_final, axis=1)

#Saving files to csv
#Predictions
list_iter = []
for i in range(100):
    name = 'iter_p_' + str(i+1)
    list_iter.append(name)

for i in range(100):
    iter_i = pd.DataFrame([Y_p_iter_clay[i],Y_p_iter_silt[i],Y_p_iter_sand[i]]).T
    iter_i[iter_i<0] = 0
    iter_i[iter_i>100] = 100
    iter_i.to_csv(list_iter[i], index = False)

#Predictions log ratio
list_iter = []
for i in range(100):
    name = 'iter_p_log_ratio_' + str(i+1)
    list_iter.append(name)

for i in range(100):
    iter_i = pd.DataFrame([Y_p_iter_clay_log_ratio[i],Y_p_iter_silt_log_ratio[i],Y_p_iter_sand_log_ratio[i]]).T
    iter_i[iter_i<0] = 0
    iter_i[iter_i>100] = 100
    iter_i.to_csv(list_iter[i], index = False)

#Actual
list_iter = []
for i in range(100):
    name = 'iter_' + str(i+1)
    list_iter.append(name)

for i in range(100):
    iter_i = pd.DataFrame([Y_iter_clay[i],Y_iter_silt[i],Y_iter_sand[i]]).T
    iter_i.to_csv(list_iter[i], index = False)


#7
####Check####
path = '/home/chirag/Documents/HSI/Soil/Paper_2_Quantitative/M1'
os.chdir(path)
Y_iter_clay = list(np.load('Y_iter_clay.npy', allow_pickle = True))
Y_iter_silt = list(np.load('Y_iter_silt.npy', allow_pickle = True))
Y_iter_sand = list(np.load('Y_iter_sand.npy', allow_pickle = True))

Y_p_iter_clay = list(np.load('Y_p_iter_clay.npy', allow_pickle = True))
Y_p_iter_silt = list(np.load('Y_p_iter_silt.npy', allow_pickle = True))
Y_p_iter_sand = list(np.load('Y_p_iter_sand.npy', allow_pickle = True))
 
path = '/home/chirag/Documents/HSI/Soil/Paper_2_Quantitative/M2'
os.chdir(path)
Y_pp_iter_silt = list(np.load('Y_pp_iter_silt_clip.npy', allow_pickle = True))

path = '/home/chirag/Documents/HSI/Soil/Paper_2_Quantitative/M3'
os.chdir(path)
Y_p_iter_log_ratio_clay = list(np.load('Y_p_iter_log_ratio_clay.npy', allow_pickle = True))
Y_p_iter_log_ratio_silt = list(np.load('Y_p_iter_log_ratio_silt.npy', allow_pickle = True))
Y_p_iter_log_ratio_sand = list(np.load('Y_p_iter_log_ratio_sand.npy', allow_pickle = True))

#Clipping the predictions of clay and sand to the range of 0 to 100
foo = Y_p_iter_clay
Y_p_iter_clay_clip = []
for i in range(100):
    foo1 = foo[i]
    Y_p_iter_clay_clip.append(np.clip(np.asarray(foo1), a_min = 0, a_max = 100))

foo = Y_p_iter_sand
Y_p_iter_sand_clip = []
for i in range(100):
    foo1 = foo[i]
    Y_p_iter_sand_clip.append(np.clip(np.asarray(foo1), a_min = 0, a_max = 100))

foo = Y_p_iter_silt
Y_p_iter_silt_clip = []
for i in range(100):
    foo1 = foo[i]
    Y_p_iter_silt_clip.append(np.clip(np.asarray(foo1), a_min = 0, a_max = 100))


array_dict = {f'array_{idx}': arr for idx, arr in enumerate(Y_p_iter_clay_clip)}
np.savez_compressed('Y_p_iter_clay_clip.npz', **array_dict)
array_dict = {f'array_{idx}': arr for idx, arr in enumerate(Y_p_iter_silt_clip)}
np.savez_compressed('Y_p_iter_silt_clip.npz', **array_dict)
array_dict = {f'array_{idx}': arr for idx, arr in enumerate(Y_p_iter_sand_clip)}
np.savez_compressed('Y_p_iter_sand_clip.npz', **array_dict)
#np.save("Y_p_iter_clay_clip",Y_p_iter_clay_clip)
#np.save("Y_p_iter_sand_clip",Y_p_iter_sand_clip)
#np.save("Y_p_iter_silt_clip",Y_p_iter_silt_clip)

#Clipped all the files to range of 0 to 100
foo3 = []
foo = list(np.load('Y_p_iter_silt.npy', allow_pickle = True))
for i in range(100):
    foo1 = foo[i]
    foo2 = np.clip(foo1, a_min = 0, a_max = 100)
    foo3.append(foo2)
#np.save("Y_p_iter_clay_clip",foo3)
#np.save("Y_p_iter_silt_clip",foo3)
#np.save("Y_p_iter_sand_clip",foo3)
Y_p_iter_clay = list(np.load('Y_p_iter_clay.npy', allow_pickle = True))
Y_p_iter_silt = list(np.load('Y_p_iter_silt.npy', allow_pickle = True))
Y_p_iter_sand = list(np.load('Y_p_iter_sand.npy', allow_pickle = True))

Y_p_iter_clay_clip = list(np.load('Y_p_iter_clay_clip.npy', allow_pickle = True))
Y_p_iter_silt = list(np.load('Y_p_iter_silt.npy', allow_pickle = True))
Y_p_iter_sand_clip = list(np.load('Y_p_iter_sand_clip.npy', allow_pickle = True))

foo = Y_p_iter_clay_clip
foo = Y_p_iter_silt
foo = Y_p_iter_sand_clip
foo3 = []
for i in range(100):
    foo1 = foo[i]
    foo3.append(np.sum(foo1<0))
np.sum(foo3)

foo2 = np.clip(np.asarray(foo1), a_min = 0, a_max = 100)
np.sum(foo1<0)
np.sum(foo4<0)
train_nos = np.load("train_nos.npy")
test_nos = np.load("test_nos.npy")
Y_iter_silt = list(np.load("Y_iter_silt.npy",allow_pickle=True))
####Check_End###


#8
##M2##
train_nos = np.load("train_nos.npy")
test_nos = np.load("test_nos.npy")
Y_iter_silt = list(np.load('Y_iter_silt.npy', allow_pickle = True))

#Calculating silt as residue for M2
Y_pp_iter_silt = []
for i in range(100):
    Y_pp_iter_silt_foo = 100 - Y_p_iter_clay_clip[i] - Y_p_iter_sand_clip[i]
    Y_pp_iter_silt_foo[Y_pp_iter_silt_foo<0] = 0
    Y_pp_iter_silt_foo[Y_pp_iter_silt_foo>100] = 100
    Y_pp_iter_silt.append(Y_pp_iter_silt_foo)
array_dict = {f'array_{idx}': arr for idx, arr in enumerate(Y_pp_iter_silt)}
np.savez_compressed('Y_pp_iter_silt_clip.npz', **array_dict)
#np.save("Y_pp_iter_silt_clip",Y_pp_iter_silt)
#pd.DataFrame(Y_pp_iter_silt).to_csv('Y_pp_iter_silt_clip.csv', index = False)

metrics_silt_final_m2 = np.empty([10,100])
for i in range(100):
    foo = train_nos[i]
    y0 = Y_iter_silt[i][0:foo]
    y0_p = Y_pp_iter_silt[i][0:foo]
    y1 = Y_iter_silt[i][foo:]
    y1_p = Y_pp_iter_silt[i][foo:]
    metrics_silt = regression_metrics(y0, y0_p, y1, y1_p)
    metrics_silt_final_m2[:,i] = metrics_silt
np.save("metrics_silt_final_m2",metrics_silt_final_m2)
pd.DataFrame(metrics_silt_final_m2).to_csv('metrics_silt_final_m2.csv', index = False)

np.mean(metrics_silt_final_m2, axis=1)
np.min(metrics_silt_final_m2, axis=1)
np.max(metrics_silt_final_m2, axis=1)
np.sum(metrics_silt_final_m2[0,:]>0)

metrics_silt_final_m2 = np.load('metrics_silt_final_m2.npy')
foo = np.mean(metrics_silt_final_m2, axis=1)

#Saving files to csv
#Predictions

list_iter = []
for i in range(100):
    name = 'iter_p_' + str(i+1)
    list_iter.append(name)

for i in range(100):
    iter_i = pd.DataFrame([Y_p_iter_clay_clip[i],Y_pp_iter_silt[i],Y_p_iter_sand_clip[i]]).T
    iter_i.to_csv(list_iter[i], index = False)

#Actual
#Same as M1. So copied directly from the corresponding folders


#9
####Outputs from R soiltexture classification####
train_nos = np.load("train_nos.npy")
test_nos = np.load("test_nos.npy")

#path = '/home/chirag/Documents/HSI/Soil/Paper_2_Quantitative/M1/OP'
path = '/home/chirag/Documents/HSI/Soil/Paper_2_Quantitative/M2/OP'
os.chdir(path)

list_iter = []
for i in range(100):
    name = 'iter_tt_' + str(i+1) + '.csv'
    list_iter.append(name)

#Metrics sequence (OA,AA,K,NA,ANA)
metrics_classification = np.empty([10,100])
tr_con_final = []
t_con_final = []

for i in range(100):
    tr_oa, t_oa, tr_aa, t_aa, tr_k, t_k, tr_na, t_na, tr_ana, t_ana = [[] for _ in range(10)]
    a = train_nos[i]
    b = test_nos[i]
    df =  pd.read_csv(list_iter[i])
    #Training
    act = []
    pred = []
    act = df['actual'].iloc[:a]
    pred = df['pred'].iloc[:a]
    tr_oa = accuracy_score(act, pred)
    tr_aa = balanced_accuracy_score(act, pred)
    tr_k = cohen_kappa_score(act, pred)
    tr_con = confusion_matrix(act, pred, labels = seq)
    tr_na = (np.sum(tr_con * neigh1))/np.sum(tr_con)
    tr_ana = (np.sum(tr_con * neigh2))/np.sum(tr_con)
    tr_con_final.append(tr_con)
    #Testing
    act = []
    pred = []
    act = df['actual'].iloc[a:a+b]
    pred = df['pred'].iloc[a:a+b]
    t_oa = accuracy_score(act, pred)
    t_aa = balanced_accuracy_score(act, pred)
    t_k = cohen_kappa_score(act, pred)
    t_con = confusion_matrix(act, pred, labels = seq)
    t_na = (np.sum(t_con * neigh1))/np.sum(t_con)
    t_ana = (np.sum(t_con * neigh2))/np.sum(t_con)
    t_con_final.append(t_con)
    metrics = []
    metrics = np.array([tr_oa, tr_aa, tr_k, tr_na, tr_ana, t_oa, t_aa, t_k, t_na, t_ana])
    metrics_classification[:,i] = metrics

np.mean(metrics_classification, axis=1)
metrics_classification = pd.DataFrame(metrics_classification)
metrics_classification.to_csv('metrics_classification.csv', index = False)

np.save("metrics_classification", metrics_classification)
np.save("tr_con_final", tr_con_final)
np.save("t_con_final", t_con_final)

#Writing out the confusion matrix
stdout_fileinfo = sys.stdout
sys.stdout = open('Confusion_Matrices.txt','a')
print('Sequence-Mean, Std deviation')
print('Training')
print(seq_texture)
print(seq)
##Mean Confusion metrics
#Training
con_tr_mean = np.around(np.mean(tr_con_final, axis=0))
con_tr_std = np.around(np.std(tr_con_final, axis=0))
print(con_tr_mean)
print(con_tr_std)
#Testing
print('Testing')
con_t_mean = np.around(np.mean(t_con_final, axis=0))
con_t_std = np.around(np.std(t_con_final, axis=0))
print(con_t_mean)
print(con_t_std)
sys.stdout.close()
sys.stdout = stdout_fileinfo
i    


#10
####Other classification Functions####
#LR
def c_lr(X_train2, Y_train2, X_test2, Y_test2):
    tr_oa, t_oa, tr_aa, t_aa, tr_k, t_k, tr_na, t_na, tr_ana, t_ana = [[] for _ in range(10)]
    #LR
    logreg.fit(X_train2, Y_train2)
    Y_train_pred = logreg.predict(X_train2)
    tr_oa = accuracy_score(Y_train2, Y_train_pred)
    tr_aa = balanced_accuracy_score(Y_train2, Y_train_pred)
    tr_k = cohen_kappa_score(Y_train2, Y_train_pred)
    tr_con = confusion_matrix(Y_train2, Y_train_pred, labels = seq_texture)
    tr_na = (np.sum(tr_con * neigh1))/np.sum(tr_con)
    tr_ana = (np.sum(tr_con * neigh2))/np.sum(tr_con)
    Y_test_pred = logreg.predict(X_test2)
    t_oa = accuracy_score(Y_test2, Y_test_pred)
    t_aa = balanced_accuracy_score(Y_test2, Y_test_pred)
    t_k = cohen_kappa_score(Y_test2, Y_test_pred)
    t_con = confusion_matrix(Y_test2, Y_test_pred, labels = seq_texture)
    t_na = (np.sum(t_con * neigh1))/np.sum(t_con)
    t_ana = (np.sum(t_con * neigh2))/np.sum(t_con)
    metrics = []
    metrics = np.array([tr_oa, tr_aa, tr_k, tr_na, tr_ana, t_oa, t_aa, t_k, t_na, t_ana])
    return metrics, tr_con, t_con

#LDA
def c_lda(X_train2, Y_train2, X_test2, Y_test2):
    tr_oa, t_oa, tr_aa, t_aa, tr_k, t_k, tr_na, t_na, tr_ana, t_ana = [[] for _ in range(10)]
    #LDA
    lda.fit(X_train2, Y_train2)
    Y_train_pred = lda.predict(X_train2)
    tr_oa = accuracy_score(Y_train2, Y_train_pred)
    tr_aa = balanced_accuracy_score(Y_train2, Y_train_pred)
    tr_k = cohen_kappa_score(Y_train2, Y_train_pred)
    tr_con = confusion_matrix(Y_train2, Y_train_pred, labels = seq_texture)
    tr_na = (np.sum(tr_con * neigh1))/np.sum(tr_con)
    tr_ana = (np.sum(tr_con * neigh2))/np.sum(tr_con)
    Y_test_pred = lda.predict(X_test2)
    t_oa = accuracy_score(Y_test2, Y_test_pred)
    t_aa = balanced_accuracy_score(Y_test2, Y_test_pred)
    t_k = cohen_kappa_score(Y_test2, Y_test_pred)
    t_con = confusion_matrix(Y_test2, Y_test_pred, labels = seq_texture)
    t_na = (np.sum(t_con * neigh1))/np.sum(t_con)
    t_ana = (np.sum(t_con * neigh2))/np.sum(t_con)
    metrics = []
    metrics = np.array([tr_oa, tr_aa, tr_k, tr_na, tr_ana, t_oa, t_aa, t_k, t_na, t_ana])
    return metrics, tr_con, t_con

##SVM
def c_svm(X_train2, Y_train2, X_test2, Y_test2):
    tr_oa, t_oa, tr_aa, t_aa, tr_k, t_k, tr_na, t_na, tr_ana, t_ana = [[] for _ in range(10)]
    #SVM
    svm.fit(X_train2, Y_train2)
    Y_train_pred = svm.predict(X_train2)
    tr_oa = accuracy_score(Y_train2, Y_train_pred)
    tr_aa = balanced_accuracy_score(Y_train2, Y_train_pred)
    tr_k = cohen_kappa_score(Y_train2, Y_train_pred)
    tr_con = confusion_matrix(Y_train2, Y_train_pred, labels = seq_texture)
    tr_na = (np.sum(tr_con * neigh1))/np.sum(tr_con)
    tr_ana = (np.sum(tr_con * neigh2))/np.sum(tr_con)
    Y_test_pred = svm.predict(X_test2)
    t_oa = accuracy_score(Y_test2, Y_test_pred)
    t_aa = balanced_accuracy_score(Y_test2, Y_test_pred)
    t_k = cohen_kappa_score(Y_test2, Y_test_pred)
    t_con = confusion_matrix(Y_test2, Y_test_pred, labels = seq_texture)
    t_na = (np.sum(t_con * neigh1))/np.sum(t_con)
    t_ana = (np.sum(t_con * neigh2))/np.sum(t_con)
    metrics = []
    metrics = np.array([tr_oa, tr_aa, tr_k, tr_na, tr_ana, t_oa, t_aa, t_k, t_na, t_ana])
    return metrics, tr_con, t_con

#RF
def c_rf(X_train2, Y_train2, X_test2, Y_test2):
    tr_oa, t_oa, tr_aa, t_aa, tr_k, t_k, tr_na, t_na, tr_ana, t_ana = [[] for _ in range(10)]
    #RF
    rf.fit(X_train2, Y_train2)
    Y_train_pred = rf.predict(X_train2)
    tr_oa = accuracy_score(Y_train2, Y_train_pred)
    tr_aa = balanced_accuracy_score(Y_train2, Y_train_pred)
    tr_k = cohen_kappa_score(Y_train2, Y_train_pred)
    tr_con = confusion_matrix(Y_train2, Y_train_pred, labels = seq_texture)
    tr_na = (np.sum(tr_con * neigh1))/np.sum(tr_con)
    tr_ana = (np.sum(tr_con * neigh2))/np.sum(tr_con)
    Y_test_pred = rf.predict(X_test2)
    t_oa = accuracy_score(Y_test2, Y_test_pred)
    t_aa = balanced_accuracy_score(Y_test2, Y_test_pred)
    t_k = cohen_kappa_score(Y_test2, Y_test_pred)
    t_con = confusion_matrix(Y_test2, Y_test_pred, labels = seq_texture)
    t_na = (np.sum(t_con * neigh1))/np.sum(t_con)
    t_ana = (np.sum(t_con * neigh2))/np.sum(t_con)
    metrics = []
    metrics = np.array([tr_oa, tr_aa, tr_k, tr_na, tr_ana, t_oa, t_aa, t_k, t_na, t_ana])
    return metrics, tr_con, t_con

metrics_classification_lr = np.empty([10,100])
tr_con_final_lr = []
t_con_final_lr = []
metrics_classification_lda = np.empty([10,100])
tr_con_final_lda = []
t_con_final_lda = []
metrics_classification_svm = np.empty([10,100])
tr_con_final_svm = []
t_con_final_svm = []
metrics_classification_rf = np.empty([10,100])
tr_con_final_rf = []
t_con_final_rf = []

#Classification Outputs
np.mean(metrics_classification_lr, axis=1)
np.mean(metrics_classification_lda, axis=1)
np.mean(metrics_classification_svm, axis=1)
np.mean(metrics_classification_rf, axis=1)

pd.DataFrame(metrics_classification_lr).to_csv('metrics_classification_lr.csv', index = False)
pd.DataFrame(metrics_classification_lda).to_csv('metrics_classification_lda.csv', index = False)
pd.DataFrame(metrics_classification_svm).to_csv('metrics_classification_svm.csv', index = False)
pd.DataFrame(metrics_classification_rf).to_csv('metrics_classification_rf.csv', index = False)

np.save("metrics_classification_plsda", metrics_classification_plsda)
np.save("tr_con_final_plsda", tr_con_final_plsda)
np.save("t_con_final_plsda", t_con_final_plsda)

#Writing out the confusion matrix
seq_texture = ['Sa', 'LoSa', 'SaLo', 'SaClLo', 'SaCl', 'Cl', 'ClLo', 'Lo', 'SiCl', 'SiClLo', 'SiLo', 'Si']
seq = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
tr_con_list = [tr_con_final_lr, tr_con_final_lda, tr_con_final_svm, tr_con_final_rf]
t_con_list = [t_con_final_lr, t_con_final_lda, t_con_final_svm, t_con_final_rf]
classifier_seq = ['LR','LDA','SVM','RF']
stdout_fileinfo = sys.stdout
sys.stdout = open('Confusion_Matrices.txt','a')
print('Sequence-Mean, Std deviation')
##Mean Confusion metrics
for i in range(4):
    print(classifier_seq[i])
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

p.mean(metrics_clay_final, axis=1)
foo = np.mean(metrics_silt_final, axis=1)
foo = np.mean(metrics_sand_final, axis=1)

#Saving files to csv
#Predictions
list_iter = []
for i in range(100):
    name = 'iter_p_' + str(i+1)
    list_iter.append(name)

for i in range(100):
    iter_i = pd.DataFrame([Y_p_iter_clay[i],Y_p_iter_silt[i],Y_p_iter_sand[i]]).T
    iter_i[iter_i<0] = 0
    iter_i[iter_i>100] = 100
    iter_i.to_csv(list_iter[i], index = False)

#Actual
list_iter = []
for i in range(100):
    name = 'iter_' + str(i+1)
    list_iter.append(name)

for i in range(100):
    iter_i = pd.DataFrame([Y_iter_clay[i],Y_iter_silt[i],Y_iter_sand[i]]).T
    iter_i.to_csv(list_iter[i], index = False)


#Other Classification functions with Regression values
#LR
def c_lr(X_train2, Y_train2, X_test2, Y_test2):
    tr_oa, t_oa, tr_aa, t_aa, tr_k, t_k, tr_na, t_na, tr_ana, t_ana = [[] for _ in range(10)]
    #LR
    logreg.fit(X_train2, Y_train2)
    Y_train_pred = logreg.predict(X_train2)
    tr_oa = accuracy_score(Y_train2, Y_train_pred)
    tr_aa = balanced_accuracy_score(Y_train2, Y_train_pred)
    tr_k = cohen_kappa_score(Y_train2, Y_train_pred)
    tr_con = confusion_matrix(Y_train2, Y_train_pred, labels = seq_texture)
    tr_na = (np.sum(tr_con * neigh1))/np.sum(tr_con)
    tr_ana = (np.sum(tr_con * neigh2))/np.sum(tr_con)
    Y_test_pred = logreg.predict(X_test2)
    t_oa = accuracy_score(Y_test2, Y_test_pred)
    t_aa = balanced_accuracy_score(Y_test2, Y_test_pred)
    t_k = cohen_kappa_score(Y_test2, Y_test_pred)
    t_con = confusion_matrix(Y_test2, Y_test_pred, labels = seq_texture)
    t_na = (np.sum(t_con * neigh1))/np.sum(t_con)
    t_ana = (np.sum(t_con * neigh2))/np.sum(t_con)
    metrics = []
    metrics = np.array([tr_oa, tr_aa, tr_k, tr_na, tr_ana, t_oa, t_aa, t_k, t_na, t_ana])
    #Regression
    #dict_clay = {0:'Cl', 1:'ClLo', 2:'LoSa', 3:'SaCl', 4:'SaClLo', 5:'SaLo', 6:'Sa'}
    dict_clay = {'Cl':58, 'ClLo':33.5, 'LoSa':6, 'SaCl':41.6, 'SaClLo':26, 'SaLo':12, 'Sa':3.3}
    metrics_clay = regression_metrics(Y_train1.iloc[:,0],pd.DataFrame(Y_train_pred)[0].map(dict_clay),Y_test.iloc[:,0],pd.DataFrame(Y_test_pred)[0].map(dict_clay))    
    dict_silt = {'Cl':20, 'ClLo':34, 'LoSa':12, 'SaCl':6.4, 'SaClLo':14, 'SaLo':22, 'Sa':5}
    metrics_silt = regression_metrics(Y_train1.iloc[:,1],pd.DataFrame(Y_train_pred)[0].map(dict_silt),Y_test.iloc[:,1],pd.DataFrame(Y_test_pred)[0].map(dict_silt))    
    dict_sand = {'Cl':22, 'ClLo':32.5, 'LoSa':82, 'SaCl':52, 'SaClLo':60, 'SaLo':66, 'Sa':91.7}
    metrics_sand = regression_metrics(Y_train1.iloc[:,2],pd.DataFrame(Y_train_pred)[0].map(dict_sand),Y_test.iloc[:,2],pd.DataFrame(Y_test_pred)[0].map(dict_sand))    
    return metrics, tr_con, t_con, metrics_clay, metrics_silt, metrics_sand

#LDA
def c_lda(X_train2, Y_train2, X_test2, Y_test2):
    tr_oa, t_oa, tr_aa, t_aa, tr_k, t_k, tr_na, t_na, tr_ana, t_ana = [[] for _ in range(10)]
    #LDA
    lda.fit(X_train2, Y_train2)
    Y_train_pred = lda.predict(X_train2)
    tr_oa = accuracy_score(Y_train2, Y_train_pred)
    tr_aa = balanced_accuracy_score(Y_train2, Y_train_pred)
    tr_k = cohen_kappa_score(Y_train2, Y_train_pred)
    tr_con = confusion_matrix(Y_train2, Y_train_pred, labels = seq_texture)
    tr_na = (np.sum(tr_con * neigh1))/np.sum(tr_con)
    tr_ana = (np.sum(tr_con * neigh2))/np.sum(tr_con)
    Y_test_pred = lda.predict(X_test2)
    t_oa = accuracy_score(Y_test2, Y_test_pred)
    t_aa = balanced_accuracy_score(Y_test2, Y_test_pred)
    t_k = cohen_kappa_score(Y_test2, Y_test_pred)
    t_con = confusion_matrix(Y_test2, Y_test_pred, labels = seq_texture)
    t_na = (np.sum(t_con * neigh1))/np.sum(t_con)
    t_ana = (np.sum(t_con * neigh2))/np.sum(t_con)
    metrics = []
    metrics = np.array([tr_oa, tr_aa, tr_k, tr_na, tr_ana, t_oa, t_aa, t_k, t_na, t_ana])
    #Regression
    #dict_clay = {0:'Cl', 1:'ClLo', 2:'LoSa', 3:'SaCl', 4:'SaClLo', 5:'SaLo', 6:'Sa'}
    dict_clay = {'Cl':58, 'ClLo':33.5, 'LoSa':6, 'SaCl':41.6, 'SaClLo':26, 'SaLo':12, 'Sa':3.3}
    metrics_clay = regression_metrics(Y_train1.iloc[:,0],pd.DataFrame(Y_train_pred)[0].map(dict_clay),Y_test.iloc[:,0],pd.DataFrame(Y_test_pred)[0].map(dict_clay))    
    dict_silt = {'Cl':20, 'ClLo':34, 'LoSa':12, 'SaCl':6.4, 'SaClLo':14, 'SaLo':22, 'Sa':5}
    metrics_silt = regression_metrics(Y_train1.iloc[:,1],pd.DataFrame(Y_train_pred)[0].map(dict_silt),Y_test.iloc[:,1],pd.DataFrame(Y_test_pred)[0].map(dict_silt))    
    dict_sand = {'Cl':22, 'ClLo':32.5, 'LoSa':82, 'SaCl':52, 'SaClLo':60, 'SaLo':66, 'Sa':91.7}
    metrics_sand = regression_metrics(Y_train1.iloc[:,2],pd.DataFrame(Y_train_pred)[0].map(dict_sand),Y_test.iloc[:,2],pd.DataFrame(Y_test_pred)[0].map(dict_sand))    
    return metrics, tr_con, t_con, metrics_clay, metrics_silt, metrics_sand

##SVM
def c_svm(X_train2, Y_train2, X_test2, Y_test2):
    tr_oa, t_oa, tr_aa, t_aa, tr_k, t_k, tr_na, t_na, tr_ana, t_ana = [[] for _ in range(10)]
    #SVM
    svm.fit(X_train2, Y_train2)
    Y_train_pred = svm.predict(X_train2)
    tr_oa = accuracy_score(Y_train2, Y_train_pred)
    tr_aa = balanced_accuracy_score(Y_train2, Y_train_pred)
    tr_k = cohen_kappa_score(Y_train2, Y_train_pred)
    tr_con = confusion_matrix(Y_train2, Y_train_pred, labels = seq_texture)
    tr_na = (np.sum(tr_con * neigh1))/np.sum(tr_con)
    tr_ana = (np.sum(tr_con * neigh2))/np.sum(tr_con)
    Y_test_pred = svm.predict(X_test2)
    t_oa = accuracy_score(Y_test2, Y_test_pred)
    t_aa = balanced_accuracy_score(Y_test2, Y_test_pred)
    t_k = cohen_kappa_score(Y_test2, Y_test_pred)
    t_con = confusion_matrix(Y_test2, Y_test_pred, labels = seq_texture)
    t_na = (np.sum(t_con * neigh1))/np.sum(t_con)
    t_ana = (np.sum(t_con * neigh2))/np.sum(t_con)
    metrics = []
    metrics = np.array([tr_oa, tr_aa, tr_k, tr_na, tr_ana, t_oa, t_aa, t_k, t_na, t_ana])
    #Regression
    #dict_clay = {0:'Cl', 1:'ClLo', 2:'LoSa', 3:'SaCl', 4:'SaClLo', 5:'SaLo', 6:'Sa'}
    dict_clay = {'Cl':58, 'ClLo':33.5, 'LoSa':6, 'SaCl':41.6, 'SaClLo':26, 'SaLo':12, 'Sa':3.3}
    metrics_clay = regression_metrics(Y_train1.iloc[:,0],pd.DataFrame(Y_train_pred)[0].map(dict_clay),Y_test.iloc[:,0],pd.DataFrame(Y_test_pred)[0].map(dict_clay))    
    dict_silt = {'Cl':20, 'ClLo':34, 'LoSa':12, 'SaCl':6.4, 'SaClLo':14, 'SaLo':22, 'Sa':5}
    metrics_silt = regression_metrics(Y_train1.iloc[:,1],pd.DataFrame(Y_train_pred)[0].map(dict_silt),Y_test.iloc[:,1],pd.DataFrame(Y_test_pred)[0].map(dict_silt))    
    dict_sand = {'Cl':22, 'ClLo':32.5, 'LoSa':82, 'SaCl':52, 'SaClLo':60, 'SaLo':66, 'Sa':91.7}
    metrics_sand = regression_metrics(Y_train1.iloc[:,2],pd.DataFrame(Y_train_pred)[0].map(dict_sand),Y_test.iloc[:,2],pd.DataFrame(Y_test_pred)[0].map(dict_sand))    
    return metrics, tr_con, t_con, metrics_clay, metrics_silt, metrics_sand

#RF
def c_rf(X_train2, Y_train2, X_test2, Y_test2):
    tr_oa, t_oa, tr_aa, t_aa, tr_k, t_k, tr_na, t_na, tr_ana, t_ana = [[] for _ in range(10)]
    #RF
    rf.fit(X_train2, Y_train2)
    Y_train_pred = rf.predict(X_train2)
    tr_oa = accuracy_score(Y_train2, Y_train_pred)
    tr_aa = balanced_accuracy_score(Y_train2, Y_train_pred)
    tr_k = cohen_kappa_score(Y_train2, Y_train_pred)
    tr_con = confusion_matrix(Y_train2, Y_train_pred, labels = seq_texture)
    tr_na = (np.sum(tr_con * neigh1))/np.sum(tr_con)
    tr_ana = (np.sum(tr_con * neigh2))/np.sum(tr_con)
    Y_test_pred = rf.predict(X_test2)
    t_oa = accuracy_score(Y_test2, Y_test_pred)
    t_aa = balanced_accuracy_score(Y_test2, Y_test_pred)
    t_k = cohen_kappa_score(Y_test2, Y_test_pred)
    t_con = confusion_matrix(Y_test2, Y_test_pred, labels = seq_texture)
    t_na = (np.sum(t_con * neigh1))/np.sum(t_con)
    t_ana = (np.sum(t_con * neigh2))/np.sum(t_con)
    metrics = []
    metrics = np.array([tr_oa, tr_aa, tr_k, tr_na, tr_ana, t_oa, t_aa, t_k, t_na, t_ana])
    #Regression
    #dict_clay = {0:'Cl', 1:'ClLo', 2:'LoSa', 3:'SaCl', 4:'SaClLo', 5:'SaLo', 6:'Sa'}
    dict_clay = {'Cl':58, 'ClLo':33.5, 'LoSa':6, 'SaCl':41.6, 'SaClLo':26, 'SaLo':12, 'Sa':3.3}
    metrics_clay = regression_metrics(Y_train1.iloc[:,0],pd.DataFrame(Y_train_pred)[0].map(dict_clay),Y_test.iloc[:,0],pd.DataFrame(Y_test_pred)[0].map(dict_clay))    
    dict_silt = {'Cl':20, 'ClLo':34, 'LoSa':12, 'SaCl':6.4, 'SaClLo':14, 'SaLo':22, 'Sa':5}
    metrics_silt = regression_metrics(Y_train1.iloc[:,1],pd.DataFrame(Y_train_pred)[0].map(dict_silt),Y_test.iloc[:,1],pd.DataFrame(Y_test_pred)[0].map(dict_silt))    
    dict_sand = {'Cl':22, 'ClLo':32.5, 'LoSa':82, 'SaCl':52, 'SaClLo':60, 'SaLo':66, 'Sa':91.7}
    metrics_sand = regression_metrics(Y_train1.iloc[:,2],pd.DataFrame(Y_train_pred)[0].map(dict_sand),Y_test.iloc[:,2],pd.DataFrame(Y_test_pred)[0].map(dict_sand))    
    return metrics, tr_con, t_con, metrics_clay, metrics_silt, metrics_sand

#PLS-DA
def c_plsda(X_train2, Y_train2, X_test2, Y_test2):
    tr_oa, t_oa, tr_aa, t_aa, tr_k, t_k, tr_na, t_na, tr_ana, t_ana = [[] for _ in range(10)]
    #PLS-DA
    #One-hot encoding for texture data
    Y_train3 = pd.get_dummies(Y_train2)
    #Define PLS object with optimal number of components
    n_comp_opt = optimise_pls_cv2(X_train2, Y_train3, n_comp=30)
    pls_opt = PLSRegression(n_components=n_comp_opt, scale=False)
    pls_opt.fit(X_train2, Y_train3)
    #Predicting and using discriminant analysis i.e. argmax function for assigning the class
    dict = {0:'Cl', 1:'ClLo', 2:'LoSa', 3:'SaCl', 4:'SaClLo', 5:'SaLo'}
    Y_train_pred = pd.DataFrame(np.argmax(pls_opt.predict(X_train2), axis = 1))[0].map(dict)
    tr_oa = accuracy_score(Y_train2, Y_train_pred)
    tr_aa = balanced_accuracy_score(Y_train2, Y_train_pred)
    tr_k = cohen_kappa_score(Y_train2, Y_train_pred)
    tr_con = confusion_matrix(Y_train2, Y_train_pred, labels = seq_texture)
    tr_na = (np.sum(tr_con * neigh1))/np.sum(tr_con)
    tr_ana = (np.sum(tr_con * neigh2))/np.sum(tr_con)
    Y_test_pred = pd.DataFrame(np.argmax(pls_opt.predict(X_test2), axis = 1))[0].map(dict)
    t_oa = accuracy_score(Y_test2, Y_test_pred)
    t_aa = balanced_accuracy_score(Y_test2, Y_test_pred)
    t_k = cohen_kappa_score(Y_test2, Y_test_pred)
    t_con = confusion_matrix(Y_test2, Y_test_pred, labels = seq_texture)
    t_na = (np.sum(t_con * neigh1))/np.sum(t_con)
    t_ana = (np.sum(t_con * neigh2))/np.sum(t_con)
    metrics = []
    metrics = np.array([tr_oa, tr_aa, tr_k, tr_na, tr_ana, t_oa, t_aa, t_k, t_na, t_ana])
    #Regression
    #dict_clay = {0:'Cl', 1:'ClLo', 2:'LoSa', 3:'SaCl', 4:'SaClLo', 5:'SaLo', 6:'Sa'}
    dict_clay = {'Cl':58, 'ClLo':33.5, 'LoSa':6, 'SaCl':41.6, 'SaClLo':26, 'SaLo':12, 'Sa':3.3}
    metrics_clay = regression_metrics(Y_train1.iloc[:,0],pd.DataFrame(Y_train_pred)[0].map(dict_clay),Y_test.iloc[:,0],pd.DataFrame(Y_test_pred)[0].map(dict_clay))    
    dict_silt = {'Cl':20, 'ClLo':34, 'LoSa':12, 'SaCl':6.4, 'SaClLo':14, 'SaLo':22, 'Sa':5}
    metrics_silt = regression_metrics(Y_train1.iloc[:,1],pd.DataFrame(Y_train_pred)[0].map(dict_silt),Y_test.iloc[:,1],pd.DataFrame(Y_test_pred)[0].map(dict_silt))    
    dict_sand = {'Cl':22, 'ClLo':32.5, 'LoSa':82, 'SaCl':52, 'SaClLo':60, 'SaLo':66, 'Sa':91.7}
    metrics_sand = regression_metrics(Y_train1.iloc[:,2],pd.DataFrame(Y_train_pred)[0].map(dict_sand),Y_test.iloc[:,2],pd.DataFrame(Y_test_pred)[0].map(dict_sand))    
    return metrics, tr_con, t_con, metrics_clay, metrics_silt, metrics_sand


#11
####Plots####
##VIP Plots
def plot_vip_with_std(vip_anything, fraction_name, approach_name):
    mean_vip = np.mean(vip_anything, axis=1)
    std_vip = np.std(vip_anything, axis=1)
    x_range = np.arange(401, 2451, step=1)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x_range, mean_vip, linestyle='-', color='b', linewidth=3)
    # Add shading for standard deviation
    ax.fill_between(x_range, mean_vip - std_vip, mean_vip + std_vip, color='b', alpha=0.5)
    plt.axhline(y=1.0, color='r', linestyle='--')  # VIP threshold line
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylim(0, 4)
    plt.grid(True)
    plt.show()
    image_format = 'png'
    image_name = str('VIP_'+ fraction_name + '_' + approach_name +'.png')
    print(image_name)
    fig.savefig(image_name, format=image_format, dpi=600)

plot_vip_with_std(vip_clay_final, 'clay', 'A1')
plot_vip_with_std(vip_silt_final, 'silt', 'A1')
plot_vip_with_std(vip_sand_final, 'sand', 'A1')
plot_vip_with_std(vip_fraction_1_final, 'fraction1', 'A3')
plot_vip_with_std(vip_fraction_2_final, 'fraction2', 'A3')
plot_vip_with_std(vip_plsda_final, 'plsda', 'A4')


#Data Import
#Set path
path = '/home/chirag/Documents/HSI/Soil/Paper_2_Quantitative'
os.chdir(path)
file = 'Working_lab.csv'
#Data Import
df =  pd.read_csv(file)
dict = {' c':'Cl', ' cl':'ClLo', ' ls':'LoSa', ' sc':'SaCl', ' scl':'SaClLo', ' sl':'SaLo'}
df['Texture'] = df['Texture'].map(dict)
#Removed bad bands less than 400 nm wavelength
df = df.drop(df.iloc[:, 5:55], axis = 1)

#Mean spectra for each texture
df_tex_ref = df.groupby('Texture').mean()
df_tex_ref.to_csv('Mean_reflectance_texture.csv')
X = -np.log10(df.drop(['Sample_Code', 'Sand', 'Clay', 'Silt', 'Texture'], axis = 1))
Y = pd.DataFrame(df, columns= ['Sample_Code', 'Clay', 'Silt', 'Sand', 'Texture'])
df1 = pd.concat([Y,X], axis=1)
df_tex_abs = df1.groupby('Texture').mean()
df_tex_abs.to_csv('Mean_absorbance_texture.csv')

#PCA from spectra
pca = PCA(n_components=3)
X = -np.log10(df.drop(['Sample_Code', 'Sand', 'Clay', 'Silt', 'Texture'], axis = 1))
S1 = pca.fit_transform(X)
explained_variance_ratio = pca.explained_variance_ratio_
df1 = pd.concat([pd.DataFrame(S1),Y['Texture']], axis=1)
df1.to_csv('PC_absorbance.csv')
X = df.drop(['Sample_Code', 'Sand', 'Clay', 'Silt', 'Texture'], axis = 1)
S2 = pca.fit_transform(X)
df2 = pd.concat([pd.DataFrame(S2),Y['Texture']], axis=1)
df2.to_csv('PC_reflectance.csv')

#Histogram of soil fractions 
foo = pd.DataFrame(df, columns= ['Clay', 'Silt', 'Sand'])
# Iterate through the five airlines
for i,name in enumerate(['Clay', 'Silt', 'Sand']):
    # Subset to the airline
    subset = foo.iloc[:, i]
    # Draw the density plot
    sns.distplot(subset, hist = False, kde = True, bins = 50,
                 kde_kws = {'shade': True, 'linewidth': 3}, norm_hist=True,
                 label = name)
# Plot formatting
plt.xlim(0,100)
plt.xlabel('Percentage fraction')
plt.ylabel('Density')

#Boxplots
#Regression based analysis
path = 'D:/Academics/PhD/SEM 9/Paper_2_Soil_texture_quantitative/Working_Files/Plot_files'
os.chdir(path)
file = 'RPD.xlsx'
df =  pd.read_excel(file)
labels = ['Cl (A1&A2)','Cl (A3)','Si (A1)','Si_r (A2)','Si (A3)','Sa (A1)','Sa (A3)']
colors = ['Red','Red','Green','Green','Green','Blue','Blue']
fig, ax = plt.subplots()
#ax.set_title('Perfromance evaluation of regression based approaches')
ax.set_ylabel('RPD')
#ax.set_ylim(-0.1,1)
ax.set_xlim(0.5,  7.5)
ax.set_xticklabels(labels, rotation=0, fontsize=8)
bplot = ax.boxplot(df, 1,patch_artist=True) 
# fill with colors
for patch, color in zip(bplot['boxes'], colors):
    patch.set_facecolor(color)
plt.show()

image_format = 'png' # e.g .png, .svg, etc.
image_name = '3RPD.png'
fig.savefig(image_name, format=image_format, dpi=600)

#Classification based analysis
path = 'D:/Academics/PhD/SEM 9/Paper_2_Soil_texture_quantitative/Working_Files/Plot_files'
os.chdir(path)
file = 'AA.xlsx'
df =  pd.read_excel(file)
labels = ['A1','A2','A3','A4']
colors = ['Red','Green','Blue','Yellow']
fig, ax = plt.subplots()
#ax.set_title('Perfromance evaluation of classification based approaches')
ax.set_ylabel('AA')
#ax.set_ylim(-0.1,1)
ax.set_xlim(0.5,  4.5)
ax.set_xticklabels(labels, rotation=0, fontsize=8)
bplot = ax.boxplot(df, 1,patch_artist=True) 
# fill with colors
for patch, color in zip(bplot['boxes'], colors):
    patch.set_facecolor(color)
plt.show()

image_format = 'png' # e.g .png, .svg, etc.
image_name = '6AA.png'
fig.savefig(image_name, format=image_format, dpi=600)

##Scatterplots
#M1
#Set path
path = 'D:/Academics/PhD/SEM 9/Paper_2_Soil_texture_quantitative/Working_Files/M1'
os.chdir(path)

metrics_clay_final = np.load("metrics_clay_final.npy")
metrics_silt_final = np.load("metrics_silt_final.npy")
metrics_sand_final = np.load("metrics_sand_final.npy")
Y_iter_clay = np.load("Y_iter_clay.npy",allow_pickle=True)
Y_iter_silt = np.load("Y_iter_silt.npy",allow_pickle=True)
Y_iter_sand = np.load("Y_iter_sand.npy",allow_pickle=True)
Y_p_iter_clay = np.load("Y_p_iter_clay.npy",allow_pickle=True)
Y_p_iter_silt = np.load("Y_p_iter_silt.npy",allow_pickle=True)
Y_p_iter_sand = np.load("Y_p_iter_sand.npy",allow_pickle=True)
train_nos = np.load("train_nos.npy")
test_nos = np.load("test_nos.npy")

i = 0 #iteration no
j = 2 #clay, silt, sand
m_0 = 0 #Approach number
k1 = [Y_iter_clay,Y_iter_silt,Y_iter_sand]
k2 = [Y_p_iter_clay,Y_p_iter_silt,Y_p_iter_sand]
l = ['clay', 'silt', 'sand']
m = ['A1', 'A2', 'A3']
metrics = metrics_sand_final

foo = k1[j]
foo1 = k2[j]
y_tr = foo[i][i:train_nos[i]]
y_p_tr = foo1[i][i:train_nos[i]]
y_test = foo[i][train_nos[i]:(train_nos[i]+test_nos[i])]
y_p_test = foo1[i][train_nos[i]:(train_nos[i]+test_nos[i])]

x = y_tr
y = y_p_tr
text = "$R^{2}$ = " +str(round(metrics[3,i],2)) + "\nRMSE = " +str(round(metrics[4,i],1)) + "\nRPD = " +str(round(metrics[5,i],2)) 
z = np.polyfit(x, y, 1)
with plt.style.context(('ggplot')):
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.scatter(x, y, c='red', edgecolors='k')
    #Plot the best fit line
    ax.plot(np.polyval(z, x), x, c='blue', linewidth=1)
    #Plot the ideal 1:1 line
    ax.plot(x, x, color='green', linewidth=1)
    plt.text( 30, 10, text, fontsize=20)
    #plt.xlabel('Measured')
    #plt.ylabel('Predicted')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.show()
image_format = 'png' # e.g .png, .svg, etc.
image_name = str(l[j]+ '_tr_' +m[m_0]+'.png')
print(image_name)
fig.savefig(image_name, format=image_format, dpi=600)

x = y_test
y = y_p_test
text = "$R^{2}$ = " +str(round(metrics[8,i],2)) + "\nRMSE = " +str(round(metrics[9,i],1)) + "\nRPD = " +str(round(metrics[10,i],2)) 
z = np.polyfit(x, y, 1)
with plt.style.context(('ggplot')):
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.scatter(x, y, c='red', edgecolors='k')
    #Plot the best fit line
    ax.plot(np.polyval(z, x), x, c='blue', linewidth=1)
    #Plot the ideal 1:1 line
    ax.plot(x, x, color='green', linewidth=1)
    plt.text( 30, 10, text,fontsize=20)
    #plt.xlabel('Measured')
    #plt.ylabel('Predicted')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.show()
image_format = 'png' # e.g .png, .svg, etc.
image_name = str(l[j]+ '_test_' +m[m_0]+'.png')
print(image_name)
fig.savefig(image_name, format=image_format, dpi=600)


#M2
path = 'D:/Academics/PhD/SEM 9/Paper_2_Soil_texture_quantitative/Working_Files/M2'
os.chdir(path)
metrics_silt_final_m2 = np.load("metrics_silt_final_m2.npy")
Y_pp_iter_silt = np.load("Y_pp_iter_silt_clip.npy",allow_pickle=True)

i = 0 #iteration no
j = 1 #clay, silt, sand
m_0 = 1 #Approach number
k1 = [Y_iter_clay,Y_iter_silt,Y_iter_sand]
#k2 = [Y_p_iter_clay,Y_p_iter_silt,Y_p_iter_sand]
l = ['clay', 'silt', 'sand']
m = ['A1', 'A2', 'A3']
metrics = metrics_silt_final_m2

foo = k1[1]
foo1 = Y_pp_iter_silt
y_tr = foo[i][i:train_nos[i]]
y_p_tr = foo1[i][i:train_nos[i]]
y_test = foo[i][train_nos[i]:(train_nos[i]+test_nos[i])]
y_p_test = foo1[i][train_nos[i]:(train_nos[i]+test_nos[i])]

x = y_tr
y = y_p_tr
text = "$R^{2}$ = " +str(round(metrics[0,i],2)) + "\nRMSE = " +str(round(metrics[1,i],1)) + "\nRPD = " +str(round(metrics[2,i],2)) 
z = np.polyfit(x, y, 1)
with plt.style.context(('ggplot')):
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.scatter(x, y, c='red', edgecolors='k')
    #Plot the best fit line
    ax.plot(np.polyval(z, x), x, c='blue', linewidth=1)
    #Plot the ideal 1:1 line
    ax.plot(x, x, color='green', linewidth=1)
    plt.text( 30, 10, text,fontsize=20)
    #plt.xlabel('Measured')
    #plt.ylabel('Predicted')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.show()
image_format = 'png' # e.g .png, .svg, etc.
image_name = str(l[j]+ '_tr_' +m[m_0]+'.png')
print(image_name)
fig.savefig(image_name, format=image_format, dpi=600)

x = y_test
y = y_p_test
text = "$R^{2}$ = " +str(round(metrics[5,i],2)) + "\nRMSE = " +str(round(metrics[6,i],1)) + "\nRPD = " +str(round(metrics[7,i],2)) 
z = np.polyfit(x, y, 1)
with plt.style.context(('ggplot')):
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.scatter(x, y, c='red', edgecolors='k')
    #Plot the best fit line
    ax.plot(np.polyval(z, x), x, c='blue', linewidth=1)
    #Plot the ideal 1:1 line
    ax.plot(x, x, color='green', linewidth=1)
    plt.text( 30, 10, text,fontsize=20)
    #plt.xlabel('Measured')
    #plt.ylabel('Predicted')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.show()
image_format = 'png' # e.g .png, .svg, etc.
image_name = str(l[j]+ '_test_' +m[m_0]+'.png')
print(image_name)
fig.savefig(image_name, format=image_format, dpi=600)


#M3
path = 'D:/Academics/PhD/SEM 9/Paper_2_Soil_texture_quantitative/Working_Files/M3'
os.chdir(path)
metrics_clay_final_m3 = np.load("metrics_clay_final_log_ratio.npy")
metrics_silt_final_m3 = np.load("metrics_silt_final_log_ratio.npy")
metrics_sand_final_m3 = np.load("metrics_sand_final_log_ratio.npy")
Y_p_iter_log_ratio_clay = np.load("Y_p_iter_log_ratio_clay.npy",allow_pickle=True)
Y_p_iter_log_ratio_silt = np.load("Y_p_iter_log_ratio_silt.npy",allow_pickle=True)
Y_p_iter_log_ratio_sand = np.load("Y_p_iter_log_ratio_sand.npy",allow_pickle=True)

k1 = [Y_iter_clay,Y_iter_silt,Y_iter_sand]
k2 = [Y_p_iter_log_ratio_clay,Y_p_iter_log_ratio_silt,Y_p_iter_log_ratio_sand]
i = 0 #iteration no
j = 2 #clay, silt, sand
m_0 = 2 #Approach number
l = ['clay', 'silt', 'sand']
m = ['A1', 'A2', 'A3']
metrics = metrics_sand_final_m3 

foo = k1[j]
foo1 = k2[j]
y_tr = foo[i][i:train_nos[i]]
y_p_tr = foo1[i][i:train_nos[i]]
y_test = foo[i][train_nos[i]:(train_nos[i]+test_nos[i])]
y_p_test = foo1[i][train_nos[i]:(train_nos[i]+test_nos[i])]

x = y_tr
y = y_p_tr
text = "$R^{2}$ = " +str(round(metrics[0,i],2)) + "\nRMSE = " +str(round(metrics[1,i],1)) + "\nRPD = " +str(round(metrics[2,i],2)) 
z = np.polyfit(x, y, 1)
with plt.style.context(('ggplot')):
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.scatter(x, y, c='red', edgecolors='k')
    #Plot the best fit line
    ax.plot(np.polyval(z, x), x, c='blue', linewidth=1)
    #Plot the ideal 1:1 line
    ax.plot(x, x, color='green', linewidth=1)
    plt.text( 30, 10, text,fontsize=20)
    #plt.xlabel('Measured')
    #plt.ylabel('Predicted')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.show()
image_format = 'png' # e.g .png, .svg, etc.
image_name = str(l[j]+ '_tr_' +m[m_0]+'.png')
print(image_name)
fig.savefig(image_name, format=image_format, dpi=600)

x = y_test
y = y_p_test
text = "$R^{2}$ = " +str(round(metrics[5,i],2)) + "\nRMSE = " +str(round(metrics[6,i],1)) + "\nRPD = " +str(round(metrics[7,i],2)) 
z = np.polyfit(x, y, 1)
with plt.style.context(('ggplot')):
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.scatter(x, y, c='red', edgecolors='k')
    #Plot the best fit line
    ax.plot(np.polyval(z, x), x, c='blue', linewidth=1)
    #Plot the ideal 1:1 line
    ax.plot(x, x, color='green', linewidth=1)
    plt.text( 30, 10, text,fontsize=20)
    #plt.xlabel('Measured')
    #plt.ylabel('Predicted')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.show()
image_format = 'png' # e.g .png, .svg, etc.
image_name = str(l[j]+ '_test_' +m[m_0]+'.png')
print(image_name)
fig.savefig(image_name, format=image_format, dpi=600)
