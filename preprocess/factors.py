import talib as ta
import pandas as pd

def factors(df):
    data = df[['open', 'high', 'low', 'close', 'volume']].to_dict()
    