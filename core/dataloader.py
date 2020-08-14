#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Pangjing Wu
Last Update: 2019-09-28
"""

from dateutil.parser import parse
import pandas as pd


class TimeSeries(object):

    def __init__(self, filepath, date_col=None, ycol=None, **kwargs):
        date_col = date_col if date_col else 0
        self._data = pd.read_csv(filepath, index_col=date_col, **kwargs)
        if type(ycol) is int:
            self._ycol = self._data.columns[ycol]
        elif type(ycol) is str:
            self._ycol = ycol
        else:
            self._ycol = None

    @property
    def X(self):
        return self._data.drop(columns=self._ycol) if self._ycol else self._data

    @property
    def y(self):
        return self._data[self._ycol] if self._ycol else None

    def daily(self, transformer, period):
        if not hasattr(transformer, '__call__'):
            raise TypeError('transformer must be a callable function.')
        df = pd.DataFrame()
        date = []
        for i in range(0, len(self._data) - period, period):
            batch = self._data.iloc[i:i+period]
            if not batch.empty:
                df = df.append(transformer(batch), sort=True)
                date.append(self._data.index[i:i+period].values.tolist())
        return (df.drop(columns=self._ycol), df[self._ycol], date) if self._ycol else (df, date)


    def weekly(self, transformer, period=1):
        if not hasattr(transformer, '__call__'):
            raise TypeError('transformer must be a callable function.')
        if period > 51 or period < 1:
            raise ValueError("The legal range of period is [1, 51].")
        df = pd.DataFrame()
        date = []
        i = 0
        while i < len(self._data):
            year_i = parse(self._data.index[i]).year
            week_i = parse(self._data.index[i]).isocalendar()[1]
            last_week = max(52, parse(str(year_i)+'-12-31').isocalendar()[1])
            j, week_j = i, week_i
            while j < len(self._data) and ((week_j - week_i) % last_week < period):
                j += 1
                if j < len(self._data):
                    week_j = parse(self._data.index[j]).isocalendar()[1]
            batch = self._data.iloc[i:j]
            if not batch.empty:
                df = df.append(transformer(batch), sort=True)
                date.append(self._data.index[i:j].values.tolist())
            i = j
        return (df.drop(columns=self._ycol), df[self._ycol], date) if self._ycol else (df, date)


    def monthly(self, transformer, period=1):
        if not hasattr(transformer, '__call__'):
            raise TypeError('transformer must be a callable function.')
        if period > 11 or period < 1:
            raise ValueError("The range of period is [1, 11].")
        df = pd.DataFrame()
        date = []
        i = 0
        while i < len(self._data):
            month_i = parse(self._data.index[i]).month
            j, month_j = i, month_i
            while j < len(self._data) and ((month_j - month_i) % 12 < period):
                j += 1
                if j < len(self._data):
                    month_j = parse(self._data.index[j]).month
            batch = self._data.iloc[i:j]
            if not batch.empty:
                df = df.append(transformer(batch), sort=True)
                date.append(self._data.index[i:j].values.tolist())
            i = j
        return (df.drop(columns=self._ycol), df[self._ycol], date) if self._ycol else (df, date)