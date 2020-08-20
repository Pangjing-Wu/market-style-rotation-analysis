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
from core.model import Hierarchy, rolling_fit_pred


def repeat(values, likes):
    ret = list()
    for i, like in zip(values, likes):
        ret += [i] * len(like)
    return np.array(ret)


def grid_search_hierarchy(features, taus, n_class, generator):
    scores = list()
    for tau in taus:
        score = list()
        X = np.array([x.mean().values for x in generator(features, tau)])
        X = StandardScaler().fit_transform(X)
        for n in n_class:
            clster = Hierarchy(n, method='ward', criterion='maxclust').fit(X)
            score.append(clster.silhouette_score_)
        scores.append(score)
    scores   = np.array(scores).T
    best_tau = taus[np.argmax(score) % len(taus)]
    best_n   = n_class[np.argmax(score) // len(taus)]
    print('get best cluster parameters: tau = %d, n = %d, score = %.3f' % (best_tau, best_n, scores.max()))
    return best_tau, best_n


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
    
    # load data.
    YCOL      = ['pct_chg']
    indicator = pd.read_csv(os.path.join(params.data_dir, 'indicators', '%s.csv' % params.stock), index_col='date', parse_dates=True)
    sentiment = pd.read_csv(os.path.join(params.data_dir, 'sentiments', params.lexicon, '%s.csv' % params.stock), index_col='date', parse_dates=True)
    data      = pd.merge(indicator, sentiment, how='left', left_index=True, right_index=True)
    data      = data.drop(['open', 'high', 'low', 'adj close'], axis=1)
    data      = data.fillna(0)

    # search best parameters of market styles.
    tau, n  = grid_search_hierarchy(data, taus=np.arange(1,9), n_class=np.arange(2,10), generator=weekly)

    # periodization.
    data_p  = [p.mean() for p in weekly(data, tau)]
    date_p  = [p.index for p in weekly(data, tau)]
    i_split = int(len(data_p) * params.split)
    X_p = np.array([x.values for x in data_p])
    X_p = StandardScaler().fit(X_p[:i_split]).transform(X_p)

    # cluster.
    cluster = Hierarchy(n_clusters=n, method='ward', criterion='maxclust')
    style_p = cluster.fit(X_p[:i_split]).predict(X_p)
    style   = repeat(style_p, date_p)
    
    # classification.
    X = data.values
    y = np.where(data[YCOL] >= 0, 1, -1).flatten()

    clf_parameters = {'C': np.logspace(0,3,10), 'gamma': np.logspace(-2,2,10)}
    clf = GridSearchCV(svm.SVC(kernel='rbf', class_weight='balanced'),
                       clf_parameters, cv=5, n_jobs=-1, scoring='f1_weighted')

    # train baseline.
    y_pred, cv = rolling_fit_pred(clf, X[:-1], y[1:], style[:-1],
                              split=params.split, pre_n=params.pre_n, z_enable=False)                        
    results = pd.DataFrame(index=data.index[-len(y_pred):])
    results['baseline'] = y_pred
    print('baseline results:')
    print('validation: mean = %.5f, std = %.5f' % (np.mean(cv), np.std(cv)))
    print(classification_report(y[-len(y_pred):], y_pred), end='\n\n')

    # train our method.
    y_pred, cv = rolling_fit_pred(clf, X[:-1], y[1:], style[:-1],
                             split=params.split, pre_n=params.pre_n, z_enable=True)
    results['ours'] = y_pred
    print('our method results:')
    print('validation: mean = %.5f, std = %.5f' % (np.mean(cv), np.std(cv)))
    print(classification_report(y[-len(y_pred):], y_pred), end='\n\n')

    # save all prediction labels.
    results.to_csv(os.path.join(params.save_dir, params.stock+'.csv'))