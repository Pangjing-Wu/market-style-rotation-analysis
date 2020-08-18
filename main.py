##!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Author: Pangjing Wu
Last Update: 2020-08-16
"""
import argparse
import os

import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.metrics import classification_report, silhouette_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

from core.generator import weekly
from core.model import Hierarchy, grid_search_hierarchy, rolling_fit_pred
from transformer import daily2period, period2daily


def parse_args():
    parser = argparse.ArgumentParser(description= 'market style rotation analysis')
    parser.add_argument('-s', '--stock', required=True, type=str, help='stock id')
    parser.add_argument('-i', '--data_dir', required=True, type=str, help='direction of data file')
    parser.add_argument('-o', '--save_dir', required=True, type=str, help='direction of output file')
    parser.add_argument('--lexicon', required=True, type=str, help='sentiment lexicion {SenticNet5|LMFinance}')
    parser.add_argument('--pre_n', default=250, type=int, help='use previous n days data as support set')
    parser.add_argument('--split', default=0.8, type=float, help='train test split')
    return parser.parse_args()


# python -u main.py -s 0001.HK -i ./data/processed -o  ./results --lexicon LMFinance 
if __name__ == '__main__':
    params = parse_args()
    np.random.seed(1)
    
    XCOLS  = ['date', 'pos', 'neu', 'neg', 'ple', 'att', 'sen',
              'apt', 'Volume', 'ma10', 'ptc_chg', 'rsi12', 'macd', 'mfi']
    YCOL   = []

    # load data and periodization.
    factor    = pd.read_csv(os.path.join(params.data_dir, 'factors', '%s.csv' % params.stock))
    sentiment = pd.read_csv(os.path.join(params.data_dir, 'sentiments', params.lexicon, '%s.csv' % params.stock))
    data    = pd.merge(factor, sentiment, on='date')
    data_p  = [p for p in weekly(data, tau)]
    date_p  = [dat.index.values.tolist() for dat in data_p]
    data_p  = [daily2period(p) for p in data_p]
    i_split = int(len(data_p) * params.split)

    # spread features and label.
    y_p = np.array([dat[YCOL] for dat in data_p])
    X_p = np.array([dat[XCOLS] for dat in data_p])
    X_p = StandardScaler().fit(X_p[:i_split]).transform(X_p)
    
    # cluster.
    taus    = np.arange(1,8)
    n_class = np.arange(2,10)
    n, tau  = grid_search_hierarchy(X_p, taus=taus, n_class=n_class)
    cluster = Hierarchy(n_clusters=n, method='ward', criterion='maxclust')
    style_p = cluster.fit(X_p[:i_split]).predict(X_p)
    style   = period2daily(style_p, date_p)
    
    # classification.
    X = data[XCOLS].values
    y = np.where(data[YCOL] >= 0, 1, -1)

    clf_parameters = {'C': np.logspace(0,3,10), 'gamma': np.logspace(-2,2,10)}
    clf = GridSearchCV(svm.SVC(kernel='rbf', class_weight='balanced'),
                    clf_parameters, iid=True, cv=5, n_jobs=-1,
                    scoring='f1_weighted', return_train_score=True)

    # train baseline.
    y_pred = rolling_fit_pred(clf, X[:-1], y[1:], style[:-1],
                             split=params.split, pre_n=params.pre_n, z_enable=False)
    results = pd.DataFrame(index=X.index[-len(y_pred):])
    results['baseline'] = y_pred
    print('--- baseline results ---')
    print(classification_report(y[-len(y_pred):], y_pred), end='\n\n')

    # train our method.
    y_pred = rolling_fit_pred(clf, X[:-1], y[1:], style[:-1],
                             split=params.split, pre_n=params.pre_n, z_enable=True)
    result['ours'] = y_pred
    print('--- our method results ---')
    print(classification_report(y[-len(y_pred):], y_pred), end='\n\n')

    # save all prediction labels.
    result.to_csv(os.path.join(params.o, params.s+'.csv'))