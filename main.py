##!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Author: Pangjing Wu
Last Update: 2019-09-29
"""
import os

import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.metrics import classification_report, silhouette_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

from core.dataloader import TimeSeries
from core.model import Hierarchy, period2daily, rolling_fit_pred
from transformer import senticnet

np.random.seed(1)


def grid_search_hierarchy(data, n_class, taus):
    scores = []
    for tau in taus:
        score = []
        for n in n_class:
            X = data.weekly(senticnet,tau)[0]
            X = StandardScaler().fit_transform(X)
            clster = Hierarchy(n, method='ward', criterion='maxclust').fit(X)
            score.append(clster.silhouette_score_)
        scores.append(score)
    scores   = np.array(scores).T
    best_tau = taus[np.argmax(score) % len(taus)]
    best_n   = n_class[np.argmax(score) // len(taus)]
    print('get best cluster parameters: tau = %d, n = %d' % (best_tau, best_n))
    return best_tau, best_n

def main(stock):
    n = 3
    tau = 4
    split = 0.8
    pre_n = 250
    cols = ['Date', 'pos', 'neu', 'neg', 'ple', 'att', 'sen',
            'apt', 'Volume', 'ma10', 'ptc_chg', 'rsi12', 'macd', 'mfi']
    cluster = Hierarchy(n_clusters=n, method='ward', criterion='maxclust')
    clf_parameters = {'C': np.logspace(0,3,10), 'gamma': np.logspace(-2,2,10)}
    clf = GridSearchCV(svm.SVC(kernel='rbf', class_weight='balanced'),
                    clf_parameters, iid=True, cv=5, n_jobs=-1,
                    scoring='f1_weighted', return_train_score=True)

    # read and preprocess data
    data = TimeSeries('data/%s.csv' % stock, usecols=cols)
    savepath = 'results/' 
    os.makedirs(savepath, exist_ok=True)

    # cluster
    X_p, date = data.weekly(senticnet ,tau)
    i_split = int(X_p.shape[0] * split)
    # make normalization based on train set
    X_p = StandardScaler().fit(X_p[:i_split]).transform(X_p)
    # make cluster based on train set
    style_p = cluster.fit(X_p[:i_split]).predict(X_p)
    style = period2daily(style_p, date)

    # classification
    X = data.X
    y = np.where(X['ptc_chg'] >= 0, 1, -1)
    
    
    y_pred, cv_results = rolling_fit_pred(clf, X[:-1], y[1:], style[:-1],
                                split=split, pre_n=pre_n, z_enable=True)
    #print(classification_report(y[-len(y_pred):], y_pred))
    df = pd.DataFrame({'date':X.index[-len(y_pred):].values, 'y_pred':y_pred})
    df.to_csv(savepath + stock + '_y.csv',index=False)
    cv_results.to_csv(savepath + stock + "_CVresult.csv", index=False)
    with open(savepath+stock+'.txt','w') as file:
        file.write(classification_report(y[-len(y_pred):], y_pred))
        file.write('\nbest param:\n%s\nbest score: %.2f\n' % (clf.best_params_, clf.best_score_))

    y_pred, cv_results = rolling_fit_pred(clf, X[:-1], y[1:], style[:-1],
                                split=split, pre_n=pre_n, z_enable=False)
    #print(classification_report(y[-len(y_pred):], y_pred))
    df = pd.concat([df, pd.DataFrame({'baseline':y_pred})])
    df.to_csv(savepath+stock+'_y.csv',index=False)
    cv_results.to_csv(savepath + stock + "_CVresult.baseline.csv", index=False)
    with open(savepath+stock+'.txt','a') as file:
        file.write(classification_report(y[-len(y_pred):], y_pred))
        file.write('\nbest param:\n%s\nbest score: %.2f\n' % (clf.best_params_, clf.best_score_))
if __name__ == '__main__':
    import sys
    argv = sys.argv[1:]
    main(*argv)
