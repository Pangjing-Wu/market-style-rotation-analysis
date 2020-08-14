import os
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from core.dataloader import TimeSeries
from core.model import Hierarchy
from transformer import senticnet

savepath = 'results/'
os.makedirs(savepath, exist_ok=True)

stocks = ['0001.HK', '0012.HK', '0016.HK', '0002.HK', '0003.HK', '0006.HK',
          '0005.HK', '0011.HK', '2388.HK', '0013.HK', '0762.HK', '0941.HK']
cols = ['Date', 'pos', 'neu', 'neg', 'ple', 'att', 'sen',
        'apt', 'Volume', 'ma10', 'ptc_chg', 'rsi12', 'macd', 'mfi']

with open(savepath + 'cluster-parameters.txt','w') as file:
    file.write('Stock \t& tau \t& n \t& Score\\\\')

for stock in stocks:
    data = TimeSeries('data/%s.csv' % stock, usecols=cols)
    taus = np.arange(1, 8)
    n_class = np.arange(2, 10)

    score = []
    for tau in taus:
        score_ = []
        for n in n_class:
            X = data.weekly(senticnet,tau)[0]
            X = StandardScaler().fit_transform(X)
            clster = Hierarchy(n, method='ward', criterion='maxclust').fit(X)
            score_.append(clster.silhouette_score_)
        score.append(score_)
    score = np.array(score).T
    tau_ = taus[np.argmax(score) % len(taus)]
    n_ = n_class[np.argmax(score) // len(taus)]
    with open(savepath + 'cluster-parameters.txt','a') as file:
        file.write('%s \t& %s \t& %s \t& %.2f\\\\' % (stock, tau_, n_, np.max(score)))