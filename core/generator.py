#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Pangjing Wu
Last Update: 2019-09-28
"""

from dateutil.parser import parse
import pandas as pd


def daily(data:pd.DataFrame, period:int, time_col=None):
    index = data[time_col] if time_col else data.index
    i = 0
    while i < len(data):
        j = i
        while j < len(data) and (index[j] - index[i]).days < period:
            j += 1
        yield data.iloc[i:j]
        i = j


def weekly(data:pd.DataFrame, period:int, time_col=None):
    if period > 11 or period < 1:
        raise ValueError("period value must be in range of (1, 51).")
    
    index = data[time_col] if time_col else data.index
    i = 0
    while i < len(data):
        j = i
        last_day_of_year = str(index[i].year) + '-12-31'
        last_week = max(52, parse(last_day_of_year).isocalendar()[1])
        while j < len(data) and (index[j].week - index[i].week) % last_week < period:
            j += 1
        yield data.iloc[i:j]
        i = j


def monthly(data:pd.DataFrame, period:int, time_col=None):
    if period > 11 or period < 1:
        raise ValueError("period value must be in range of (1, 11).")
    
    index = data[time_col] if time_col else data.index
    i = 0
    while i < len(data):
        j = i
        while j < len(data) and (index[j].month - index[i].month) % 12 < period:
            j += 1
        yield data.iloc[i:j]
        i = j