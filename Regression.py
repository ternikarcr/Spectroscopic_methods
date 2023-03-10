#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 18:03:33 2022

@author: chirag
"""

#DID NOT WORK
#For rpy2 usage
#conda install -c conda-forge tzlocal
#conda install -c conda-forge rpy2
#Open R base from terminal and install soil texture package
#Then add R Home to the environment path
#To check R Home, ues the following in command line:
#R RHOME
#This gave the following result:  /home/chirag/anaconda3/lib/R
#os.getenv('HOME')
#os.environ['R_HOME'] = "/home/chirag/anaconda3/lib/R"

#import tzlocal
#import rpy2
#print(rpy2.__version__)
#from rpy2.robjects.packages import importr
#foo = rpy_(robjects).packages.importr('soiltexture')
#import rpy2.robjects as robj
#from rpy2.robjects.packages import importr
#foo = importr('soiltexture')
#from rpy2.robjects import pandas2ri

import sys
from sys import stdout
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv
from scipy.stats import chi2
#from kennard_stone import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
#from sklearn.metrics import precision_recall_fscore_support as prfs
from spectres import spectres
import matplotlib.pyplot as plt
from matplotlib import gridspec
import time
import pylab as pl
import seaborn as sns
import collections

path = '/home/chirag/Documents/HSI/Soil/Paper_2_Quantitative'
os.chdir(path)

file = 'Working_lab.csv'

#PLS function
def optimise_pls_cv(X0, y0, X1, y1, n_comp, plot_components=True):
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
    print("Suggested number of components: ", n_comp_final)
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
    # Cross-validation
    y0_p_cv = cross_val_predict(pls_opt, X0, y0, cv=loocv)
    #Formatting for easy calculations
    y0_p = pd.Series(y0_p[:,0], index=y0.index)
    y0_p_cv = pd.Series(y0_p_cv[:,0], index=y0.index)
    #Calculate scores for calibration and cross-validation
    score0_p = r2_score(y0, y0_p)
    score0_p_cv = r2_score(y0, y0_p_cv)
    #Plot regression and figures of merit
    rangey = max(y0) - min(y0)
    rangex = max(y0_p) - min(y0_p)
    #Fit a line to the CV vs response
g    z = np.polyfit(y0, y0_p, 1)
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
    print('R2 calib calculation{:.3f}'.format(r2_c))
    print('RMSE calib{:.3f}'.format(rmse_c))
    print('R2 CV calculation{:.3f}'.format(r2_cv))    
    print('RMSE CV{:.3f}'.format(rmse_cv))
    
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
    print('R2 valid calculation{:.3f}'.format(r2_v))
    print('RMSE valid{:.3f}'.format(rmse_v))
    metrics_final = np.array([n_comp_final, r2_c, rmse_c, rpd_c, rpiq_c, bias_c, r2_v, rmse_v, rpd_v, rpiq_v, bias_v])
    return metrics_final, y0_p, y1_p


#Data Import
df =  pd.read_csv(file)

####DATA WRANGLING and Duplicate checking####
#No duplicates, so commented out the following lines
#df2 = df.copy()
#df2 = df.drop(['Sample_Code', 'Sand', 'Clay', 'Silt', 'Texture'], axis=1)
#df2.drop_duplicates(keep=False, inplace=True)
#del df2


##Regression
X = df.drop(['Sample_Code', 'Sand', 'Clay', 'Silt', 'Texture'], axis = 1)
Y = pd.DataFrame(df, columns= ['Clay', 'Silt', 'Sand', 'Texture'])

#Histogram of textures
#Texture_class = Y['Texture']
#Texture_class.unique()
#len(Texture_class.unique())
#fullset = set(Texture_class)
#print(len(fullset))
#Texture_class1 = collections.Counter(sorted(Texture_class))
#df1 = pd.DataFrame.from_dict(Texture_class1, orient='index')
#df1.plot(kind='bar')
#df1.to_csv('Histogram_Texture')


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, stratify=Y['Texture'])
#X_train.describe()
#X_test.describe()
#Y_train.describe()
#Y_test.describe()

#Data preprocessing, Outlier detection
pca = PCA(n_components=3)
S = pca.fit_transform(X_train)
cov = np.cov(S.T)
i_cov = inv(cov)
left = np.dot(S, i_cov)
dist_mat = np.dot(left, S.T)
mahal_dist = pd.DataFrame(np.sqrt(dist_mat.diagonal()))
mahal_dist.columns = ['MD']
#np.sum(mahal_dist['MD']>3.5)
#From geeks for geeks mahal distance calculation
#mahal_dist['p'] = 1 - chi2.cdf(mahal_dist['MD'],2)
#Setting index same as X_train
mahal_dist.set_index(X_train.index, inplace=True)
#np.sum(mahal_dist['p']<0.001)

#X_train1 = X_train[mahal_dist['p']>0.001]
#Y_train1 = Y_train[mahal_dist['p']>0.001]

scaler = StandardScaler()
X_train2 = scaler.fit_transform(X_train1)
X_test2 = scaler.fit_transform(X_test)

#Clay
a = 0
Y_train2 = Y_train1.iloc[:,a]
Y_test2 = Y_test.iloc[:,a]
metrics_clay, y_c_clay, y_v_clay = optimise_pls_cv(X0=X_train2, y0 = Y_train2, X1 = X_test2, y1 = Y_test2, n_comp = 40, plot_components=True)
metrics_clay_final = metrics_clay
Y_iter_clay = pd.concat([Y_train2,Y_test2])
Y_p_iter_clay = pd.concat([y_c_clay,y_v_clay])

#Silt
a = 1
Y_train2 = Y_train1.iloc[:,a]
Y_test2 = Y_test.iloc[:,a]
metrics_silt, y_c_silt, y_v_silt = optimise_pls_cv(X0=X_train2, y0 = Y_train2, X1 = X_test2, y1 = Y_test2, n_comp = 40, plot_components=True)


#Sand
a = 2
Y_train2 = Y_train1.iloc[:,a]
Y_test2 = Y_test.iloc[:,a]
metrics_sand, y_c_sand, y_v_sand = optimise_pls_cv(X0=X_train2, y0 = Y_train2, X1 = X_test2, y1 = Y_test2, n_comp = 40, plot_components=True)


####FINAL####
##M1##
#Import packages
import sys
from sys import stdout
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
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
from spectres import spectres
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
    return metrics_final, y0_p, y1_p


#Classification functions
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


####Iterations
metrics_clay_final = np.empty([13,100])
Y_iter_clay = []
Y_p_iter_clay = []
metrics_silt_final = np.empty([13,100])
Y_iter_silt = []
Y_p_iter_silt = []
metrics_sand_final = np.empty([13,100])
Y_iter_sand = []
Y_p_iter_sand = []
train_nos = []
test_nos = []

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

X = -np.log10(df.drop(['Sample_Code', 'Sand', 'Clay', 'Silt', 'Texture'], axis = 1))
Y = pd.DataFrame(df, columns= ['Clay', 'Silt', 'Sand', 'Texture'])

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
    #From geeks for geeks mahal distance calculation
    #mahal_dist['p'] = 1 - chi2.cdf(mahal_dist['MD'],2)
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
    metrics_clay, y_c_clay, y_v_clay = optimise_pls_cv(X0=X_train2, y0 = Y_train2, X1 = X_test2, y1 = Y_test2, n_comp = 40, plot_components=False)
    metrics_clay_final[:,i] = metrics_clay
    Y_iter_clay.append(pd.concat([Y_train2,Y_test2]))
    Y_p_iter_clay.append(pd.concat([y_c_clay,y_v_clay]))
    #Silt
    a = 1
    Y_train2 = Y_train1.iloc[:,a]
    Y_test2 = Y_test.iloc[:,a]
    metrics_silt, y_c_silt, y_v_silt = optimise_pls_cv(X0=X_train2, y0 = Y_train2, X1 = X_test2, y1 = Y_test2, n_comp = 40, plot_components=False)
    metrics_silt_final[:,i] = metrics_silt
    Y_iter_silt.append(pd.concat([Y_train2,Y_test2]))
    Y_p_iter_silt.append(pd.concat([y_c_silt,y_v_silt]))
    #Sand
    a = 2
    Y_train2 = Y_train1.iloc[:,a]
    Y_test2 = Y_test.iloc[:,a]
    metrics_sand, y_c_sand, y_v_sand = optimise_pls_cv(X0=X_train2, y0 = Y_train2, X1 = X_test2, y1 = Y_test2, n_comp = 40, plot_components=False)
    metrics_sand_final[:,i] = metrics_sand
    Y_iter_sand.append(pd.concat([Y_train2,Y_test2]))
    Y_p_iter_sand.append(pd.concat([y_c_sand,y_v_sand]))
    train_nos.append(Y_train2.size)
    test_nos.append(Y_test2.size)
    #Classification
    a = 3
    Y_train2 = Y_train1.iloc[:,a]
    Y_test2 = Y_test.iloc[:,a]
    #LR
    metrics, tr_con, t_con = c_lr(X_train2, Y_train2, X_test2, Y_test2)
    metrics_classification_lr[:,i] = metrics
    tr_con_final_lr.append(tr_con)
    t_con_final_lr.append(t_con)    
    #LDA
    metrics, tr_con, t_con = c_lda(X_train2, Y_train2, X_test2, Y_test2)
    metrics_classification_lda[:,i] = metrics
    t_con_final_lda.append(t_con)    
    tr_con_final_lda.append(tr_con)    
    #SVM
    metrics, tr_con, t_con = c_svm(X_train2, Y_train2, X_test2, Y_test2)
    tr_con_final_svm.append(tr_con)    
    t_con_final_svm.append(t_con)    
    metrics_classification_svm[:,i] = metrics
    #RF
    metrics, tr_con, t_con = c_rf(X_train2, Y_train2, X_test2, Y_test2)
    tr_con_final_rf.append(tr_con)    
    t_con_final_rf.append(t_con)    
    metrics_classification_rf[:,i] = metrics
    print(i)
end_time = time.time()
#total time taken
req_time = (end_time - start_time) / 60
print("Time required was {} minutes".format(req_time))

    
#Check
#foo = np.sum(metrics_silt_final[8,:]>0)

####Saving raw npy files####
np.save("metrics_clay_final",metrics_clay_final)
np.save("metrics_silt_final",metrics_silt_final)
np.save("metrics_sand_final",metrics_sand_final)
np.save("Y_iter_clay",Y_iter_clay)
np.save("Y_iter_silt",Y_iter_silt)
np.save("Y_iter_sand",Y_iter_sand)
np.save("Y_p_iter_clay",Y_p_iter_clay)
np.save("Y_p_iter_silt",Y_p_iter_silt)
np.save("Y_p_iter_sand",Y_p_iter_sand)
np.save("train_nos",train_nos)
np.save("test_nos",test_nos)

pd.DataFrame(metrics_clay_final).to_csv('metrics_clay_final.csv', index = False)
pd.DataFrame(metrics_silt_final).to_csv('metrics_silt_final.csv', index = False)
pd.DataFrame(metrics_sand_final).to_csv('metrics_sand_final.csv', index = False)

#Set path
path = '/home/chirag/Documents/HSI/Soil/Paper_2_Quantitative'
os.chdir(path)
metrics_clay_final = np.load('metrics_clay_final.npy', allow_pickle = True)
metrics_silt_final = np.load('metrics_silt_final.npy', allow_pickle = True)
metrics_sand_final = np.load('metrics_sand_final.npy', allow_pickle = True)
metrics_clay_final = pd.DataFrame(metrics_clay_final)
metrics_silt_final = pd.DataFrame(metrics_silt_final)
metrics_sand_final = pd.DataFrame(metrics_sand_final)

#Classification Outputs
np.mean(metrics_classification_lr, axis=1)
np.mean(metrics_classification_lda, axis=1)
np.mean(metrics_classification_svm, axis=1)
np.mean(metrics_classification_rf, axis=1)

pd.DataFrame(metrics_classification_lr).to_csv('metrics_classification_lr.csv', index = False)
pd.DataFrame(metrics_classification_lda).to_csv('metrics_classification_lda.csv', index = False)
pd.DataFrame(metrics_classification_svm).to_csv('metrics_classification_svm.csv', index = False)
pd.DataFrame(metrics_classification_rf).to_csv('metrics_classification_rf.csv', index = False)

np.save("metrics_classification_lr", metrics_classification_lr)
np.save("tr_con_final_lr", tr_con_final_lr)
np.save("t_con_final_lr", t_con_final_lr)
np.save("metrics_classification_lda", metrics_classification_lda)
np.save("tr_con_final_lda", tr_con_final_lda)
np.save("t_con_final_lda", t_con_final_lda)
np.save("metrics_classification_svm", metrics_classification_svm)
np.save("tr_con_final_svm", tr_con_final_svm)
np.save("t_con_final_svm", t_con_final_svm)
np.save("metrics_classification_rf", metrics_classification_rf)
np.save("tr_con_final_rf", tr_con_final_rf)
np.save("t_con_final_rf", t_con_final_rf)

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




np.mean(metrics_clay_final, axis=1)
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


####Check####
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

foo = Y_p_iter_clay
foo = Y_p_iter_silt
foo = Y_p_iter_sand
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


##M2##
Y_pp_iter_silt = []
for i in range(100):
    y_m2_c_silt = 100 - Y_p_iter_clay[i] - Y_p_iter_sand[i]
    Y_pp_iter_silt.append(y_m2_c_silt)
np.save("Y_pp_iter_silt",Y_pp_iter_silt)

metrics_silt_final_m2 = np.empty([10,100])
for i in range(100):
    foo = train_nos[i]
    y0 = Y_iter_silt[i][0:foo]
    y0_p = Y_pp_iter_silt[i][0:foo]
    y1 = Y_iter_silt[i][foo:]
    y1_p = Y_pp_iter_silt[i][foo:]
    #Calculate metrics for calibration
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
    #Calculate metrics for validation
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
    metrics_final = np.array([r2_c, rmse_c, rpd_c, rpiq_c, bias_c, r2_v, rmse_v, rpd_v, rpiq_v, bias_v])
    metrics_silt_final_m2[:,i] = metrics_final

np.save("metrics_silt_final_m2",metrics_silt_final_m2)
pd.DataFrame(metrics_silt_final_m2).to_csv('metrics_silt_final_m2.csv', index = False)

np.mean(metrics_silt_final_m2, axis=1)
np.min(metrics_silt_final_m2, axis=1)
np.max(metrics_silt_final_m2, axis=1)
np.sum(metrics_silt_final_m2[0,:]>0)


#Set path
path = '/home/chirag/Documents/HSI/Soil/Paper_2_Quantitative/M2'
os.chdir(path)
metrics_silt_final_m2 = np.load('metrics_silt_final_m2.npy')
foo = np.mean(metrics_silt_final_m2, axis=1)

#Saving files to csv
#Predictions
list_iter = []
for i in range(100):
    name = 'iter_p_' + str(i+1)
    list_iter.append(name)

for i in range(100):
    iter_i = pd.DataFrame([Y_p_iter_clay[i],Y_pp_iter_silt[i],Y_p_iter_sand[i]]).T
    iter_i[iter_i<0] = 0
    iter_i[iter_i>100] = 100
    iter_i.to_csv(list_iter[i], index = False)

#Actual
#Same as M1. So copied directly from the corresponding folders



####Outputs from R soiltexture classification####
path = '/home/chirag/Documents/HSI/Soil/Paper_2_Quantitative'
os.chdir(path)
train_nos = np.load("train_nos.npy")
test_nos = np.load("test_nos.npy")

#path = '/home/chirag/Documents/HSI/Soil/Paper_2_Quantitative/M1/OP'
path = '/home/chirag/Documents/HSI/Soil/Paper_2_Quantitative/M2/OP'
os.chdir(path)

#seq = ['Sa', 'LoSa', 'SaLo', 'SaClLo', 'SaCl', 'Cl', 'ClLo', 'Lo', 'SiCl', 'SiClLo', 'SiLo', 'Si']
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

seq_texture = ['Sa', 'LoSa', 'SaLo', 'SaClLo', 'SaCl', 'Cl', 'ClLo', 'Lo', 'SiCl', 'SiClLo', 'SiLo', 'Si']
seq = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

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




####Direct Classification####
X = -np.log10(df.drop(['Sample_Code', 'Sand', 'Clay', 'Silt', 'Texture'], axis = 1))
Y = pd.DataFrame(df, columns= ['Texture'])

path = '/home/chirag/Documents/HSI/Soil/Paper_2_Quantitative/M3'
os.chdir(path)

seq_texture = ['Sa', 'LoSa', 'SaLo', 'SaClLo', 'SaCl', 'Cl', 'ClLo', 'Lo', 'SiCl', 'SiClLo', 'SiLo', 'Si']
seq = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]


#Metrics sequence (OA,AA,K,NA,ANA)
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

start_time = time.time()
for i in range(100):
i = 0
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
    Y_train2 = Y_train1
    Y_test2 = Y_test
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
    tr_con_final_lr.append(tr_con)
    Y_test_pred = logreg.predict(X_test2)
    t_oa = accuracy_score(Y_test2, Y_test_pred)
    t_aa = balanced_accuracy_score(Y_test2, Y_test_pred)
    t_k = cohen_kappa_score(Y_test2, Y_test_pred)
    t_con = confusion_matrix(Y_test2, Y_test_pred, labels = seq_texture)
    t_na = (np.sum(t_con * neigh1))/np.sum(t_con)
    t_ana = (np.sum(t_con * neigh2))/np.sum(t_con)
    t_con_final_lr.append(t_con)    
    metrics = []
    metrics = np.array([tr_oa, tr_aa, tr_k, tr_na, tr_ana, t_oa, t_aa, t_k, t_na, t_ana])
    metrics_classification_lr[:,i] = metrics
    #LDA
    lda.fit(X_train2, Y_train2)
    Y_train_pred = lda.predict(X_train2)
    tr_oa = accuracy_score(Y_train2, Y_train_pred)
    tr_aa = balanced_accuracy_score(Y_train2, Y_train_pred)
    tr_k = cohen_kappa_score(Y_train2, Y_train_pred)
    tr_con = confusion_matrix(Y_train2, Y_train_pred, labels = seq_texture)
    tr_na = (np.sum(tr_con * neigh1))/np.sum(tr_con)
    tr_ana = (np.sum(tr_con * neigh2))/np.sum(tr_con)
    tr_con_final_lda.append(tr_con)    
    Y_test_pred = lda.predict(X_test2)
    t_oa = accuracy_score(Y_test2, Y_test_pred)
    t_aa = balanced_accuracy_score(Y_test2, Y_test_pred)
    t_k = cohen_kappa_score(Y_test2, Y_test_pred)
    t_con = confusion_matrix(Y_test2, Y_test_pred, labels = seq_texture)
    t_na = (np.sum(t_con * neigh1))/np.sum(t_con)
    t_ana = (np.sum(t_con * neigh2))/np.sum(t_con)
    t_con_final_lda.append(t_con)    
    metrics = []
    metrics = np.array([tr_oa, tr_aa, tr_k, tr_na, tr_ana, t_oa, t_aa, t_k, t_na, t_ana])
    metrics_classification_lda[:,i] = metrics
    #SVM
    svm.fit(X_train2, Y_train2)
    Y_train_pred = svm.predict(X_train2)
    tr_oa = accuracy_score(Y_train2, Y_train_pred)
    tr_aa = balanced_accuracy_score(Y_train2, Y_train_pred)
    tr_k = cohen_kappa_score(Y_train2, Y_train_pred)
    tr_con = confusion_matrix(Y_train2, Y_train_pred, labels = seq_texture)
    tr_na = (np.sum(tr_con * neigh1))/np.sum(tr_con)
    tr_ana = (np.sum(tr_con * neigh2))/np.sum(tr_con)
    tr_con_final_svm.append(tr_con)    
    Y_test_pred = svm.predict(X_test2)
    t_oa = accuracy_score(Y_test2, Y_test_pred)
    t_aa = balanced_accuracy_score(Y_test2, Y_test_pred)
    t_k = cohen_kappa_score(Y_test2, Y_test_pred)
    t_con = confusion_matrix(Y_test2, Y_test_pred, labels = seq_texture)
    t_na = (np.sum(t_con * neigh1))/np.sum(t_con)
    t_ana = (np.sum(t_con * neigh2))/np.sum(t_con)
    t_con_final_svm.append(t_con)    
    metrics = []
    metrics = np.array([tr_oa, tr_aa, tr_k, tr_na, tr_ana, t_oa, t_aa, t_k, t_na, t_ana])
    metrics_classification_svm[:,i] = metrics
    #RF
    rf.fit(X_train2, Y_train2)
    Y_train_pred = rf.predict(X_train2)
    tr_oa = accuracy_score(Y_train2, Y_train_pred)
    tr_aa = balanced_accuracy_score(Y_train2, Y_train_pred)
    tr_k = cohen_kappa_score(Y_train2, Y_train_pred)
    tr_con = confusion_matrix(Y_train2, Y_train_pred, labels = seq_texture)
    tr_na = (np.sum(tr_con * neigh1))/np.sum(tr_con)
    tr_ana = (np.sum(tr_con * neigh2))/np.sum(tr_con)
    tr_con_final_rf.append(tr_con)    
    Y_test_pred = rf.predict(X_test2)
    t_oa = accuracy_score(Y_test2, Y_test_pred)
    t_aa = balanced_accuracy_score(Y_test2, Y_test_pred)
    t_k = cohen_kappa_score(Y_test2, Y_test_pred)
    t_con = confusion_matrix(Y_test2, Y_test_pred, labels = seq_texture)
    t_na = (np.sum(t_con * neigh1))/np.sum(t_con)
    t_ana = (np.sum(t_con * neigh2))/np.sum(t_con)
    t_con_final_rf.append(t_con)    
    metrics = []
    metrics = np.array([tr_oa, tr_aa, tr_k, tr_na, tr_ana, t_oa, t_aa, t_k, t_na, t_ana])
    metrics_classification_rf[:,i] = metrics
    print(i)
end_time = time.time()
#total time taken
req_time = (end_time - start_time) / 60
print("Time required was {} minutes".format(req_time))

np.mean(metrics_classification_lr, axis=1)
np.mean(metrics_classification_lda, axis=1)
np.mean(metrics_classification_svm, axis=1)
np.mean(metrics_classification_rf, axis=1)

pd.DataFrame(metrics_classification_lr).to_csv('metrics_classification_lr.csv', index = False)
pd.DataFrame(metrics_classification_lda).to_csv('metrics_classification_lda.csv', index = False)
pd.DataFrame(metrics_classification_svm).to_csv('metrics_classification_svm.csv', index = False)
pd.DataFrame(metrics_classification_rf).to_csv('metrics_classification_rf.csv', index = False)

np.save("metrics_classification_lr", metrics_classification_lr)
np.save("tr_con_final_lr", tr_con_final_lr)
np.save("t_con_final_lr", t_con_final_lr)
np.save("metrics_classification_lda", metrics_classification_lda)
np.save("tr_con_final_lda", tr_con_final_lda)
np.save("t_con_final_lda", t_con_final_lda)
np.save("metrics_classification_svm", metrics_classification_svm)
np.save("tr_con_final_svm", tr_con_final_svm)
np.save("t_con_final_svm", t_con_final_svm)
np.save("metrics_classification_rf", metrics_classification_rf)
np.save("tr_con_final_rf", tr_con_final_rf)
np.save("t_con_final_rf", t_con_final_rf)


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


