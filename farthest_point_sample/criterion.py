import pandas as pd
import random
import scipy
from scipy import stats
import numpy as np
import os
from scipy.optimize import curve_fit

def logistic_func(X, bayta1, bayta2, bayta3, bayta4):
    logisticPart = 1 + np.exp(np.negative(np.divide(X - bayta3, np.abs(bayta4))))
    yhat = bayta2 + np.divide(bayta1 - bayta2, logisticPart)
    return yhat

def fit_function(y_label, y_output):
    beta = [np.max(y_label), np.min(y_label), np.mean(y_output), 0.5]
    popt, _ = curve_fit(logistic_func, y_output, \
        y_label, p0=beta, maxfev=100000000)
    y_output_logistic = logistic_func(y_output, *popt)
    
    return y_output_logistic


def compute(database,criteria):
    if database == "wpc":
        mos_dir = r"csvfiles\wpc_data_info"
        fr_dir = "wpc"
        ex_id = 5
    elif database == "sjtu":
        mos_dir = r"csvfiles\sjtu_data_info"
        fr_dir = "sjtu"
        ex_id = 9
    plcc = []
    srcc =[]
    rmse = []
    krcc = []
    cnt = 0
    for i in range(ex_id):
        # record the result
        predict_score = pd.read_csv(os.path.join(fr_dir,'test_'+str(i+1)+'_fr_score.csv'))[criteria].to_list()
        test_score = pd.read_csv(os.path.join(mos_dir,'test_'+str(i+1)+'.csv'))['mos'].to_list()
        y_output_logistic = fit_function(test_score, predict_score)
        plcc.append(stats.pearsonr(y_output_logistic, test_score)[0])
        srcc.append(stats.spearmanr(predict_score, test_score)[0])
        rmse.append(np.sqrt(((y_output_logistic-test_score) ** 2).mean()))
        krcc.append(stats.kendalltau(predict_score, test_score)[0])

    
    print("SRCC:  "+ str(sum(srcc)/len(srcc)))
    print("PLCC:  "+ str(sum(plcc)/len(plcc)))
    print("KRCC:  "+ str(sum(krcc)/len(krcc)))
    print("RMSE:  "+ str(sum(rmse)/len(rmse)))
    print("------------------------------------")



compute('wpc','p2point')
compute('wpc','p2point_hausdorf')
compute('wpc','p2plane')
compute('wpc','psnr_y')
# compute('sjtu','p2point')
# compute('sjtu','p2point_hausdorf')
# compute('sjtu','p2plane')
# compute('sjtu','psnr_y')