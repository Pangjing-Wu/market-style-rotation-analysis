# -*- coding: utf-8 -*-
"""
Author: Pangjing Wu
Last Update: 2019-09-09
"""
import pandas as pd


def daily2period(df):
    data = pd.DataFrame(
        {'pos':df['pos'].mean(), 'neu':df['neu'].mean(), 'neg':df['neg'].mean(),
         'ple':df['ple'].mean(), 'att':df['att'].mean(), 'sen':df['sen'].mean(),
         'apt':df['apt'].mean(), 'Volume':df['Volume'].mean(),
         'ma10':df['ma10'].mean(), 'ptc_chg':df['ptc_chg'].mean(),
         'rsi12':df['rsi12'].mean(), 'macd':df['macd'].mean(),
         'mfi':df['mfi'].mean()},
        index=[df.index[0]])
    return data


def period2daily(styles, date_p):
    Z = list()
    for z in styles:
        Z += [z] * len(date_p[i])
    return np.array(Z)