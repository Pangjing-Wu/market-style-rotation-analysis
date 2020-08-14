#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Author: Pangjing Wu
Last Update: 2019-09-29
"""

from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import fclusterdata, linkage
import numpy as np
import pandas as pd

class Hierarchy(object):

    def __init__(self, n_clusters=2, method='single', criterion='inconsistent'):
        self._n = n_clusters
        self._method = method
        self._criterion = criterion

    def __call__(self, n_clusters=None):
        self._n = n_clusters if n_clusters else self._n
        return self

    def fit(self, X):
        X = np.array(X)
        if X.ndim != 2:
            raise ValueError('Expect input array has 2 dimensions, but got %d.' % X.ndim)
        self._X = X
        self._y = fclusterdata(self._X, self._n, self._criterion, method=self._method) - 1
        self._center = self._cal_cluster_center()
        self._radius = self._cal_cluster_radius()
        return self

    def predict(self, X=None):
        if X is None:
            return self._y
        X = np.array(X)
        if X.ndim != 2:
            raise ValueError('Expect input array has 2 dimensions, but got %d.' % X.ndim)
        # For each x, find the closest center as x's class.
        y = [np.argmin([np.linalg.norm(x - c) for c in self._center]) for x in X]
        return np.array(y)

    def fit_predict(self, *args, **kwargs):
        self.fit(*args, **kwargs)
        return self._y

    @property
    def silhouette_score_(self):
        return silhouette_score(self._X, self._y)

    @property
    def center(self):
        return self._center

    @property
    def radius(self):
        return self._radius

    def _cal_cluster_center(self):
        return np.array([np.mean(self._X[self._y==i], axis=0) for i in range(self._n)])

    def _cal_cluster_radius(self):
        radius = [max([np.linalg.norm(x - self._center[i]) for x in self._X[self._y==i]]) for i in range(self._n)]
        return np.array(radius)


def period2daily(X_p, datelist):
    X = []
    for i, x in enumerate(X_p):
        for _ in datelist[i]:
            X.append(x)
    return np.array(X)


def rolling_fit_pred(clf, X, y, z, split=0.5, pre_n=None, z_enable=True, cvresult=True):
    '''
    z_enable is False, the X_ is composed by the final part of X which
    has the same length as the X_ when z_enable is True.
    '''
    result = []
    cv_results = pd.DataFrame()
    # sample_num = []
    i_split = int(X.shape[0] * split)
    # X_ consist of pre_n training samples and 1 sample for prediction.
    pre_n += 1 
    for i in range(i_split, X.shape[0]):
        # sample_num.append(len(X[:i][z[:i] == z[i]]))
        if z_enable:
            X_ = X[:i][z[:i] == z[i]]
            y_ = y[:i][z[:i] == z[i]]
        else:
            X_ = X[:i][-len(X[:i][z[:i] == z[i]]):]
            y_ = y[:i][-len(X[:i][z[:i] == z[i]]):]
        if pre_n is not None and X_.shape[0] > pre_n:
            X_, y_ = X_[-pre_n-1:], y_[-pre_n-1:]
        X_ = StandardScaler().fit_transform(X_)
        clf.fit(X_[:-1], y_[:-1])
        result.append(clf.predict([X_[-1]]))
        cv_result = pd.DataFrame(clf.cv_results_)
        cv_results = cv_results.append(cv_result[cv_result['rank_test_score']==1],
                                       ignore_index=True)
    # print(sample_num)
    result = np.array(result).flatten()
    return (result, cv_results) if cvresult else result