import talib as ta
import pandas as pd

import numpy as np
np.random.seed(0)
length = 500
data = pd.DataFrame(dict(
        open   = np.random.random(length),
        high   = np.random.random(length),
        low    = np.random.random(length),
        close  = np.random.random(length),
        volume = np.random.random(length)
        )
    )

def factors(df):
    # overlap studies
    df['ma7']  = ta.MA(df.close, timeperiod=7)
    df['ma14'] = ta.MA(df.close, timeperiod=14)
    df['ma21'] = ta.MA(df.close, timeperiod=21)
    df['upper'], df['mid'], df['lower'] = ta.BBANDS(df.close, timeperiod=5, nbdevup=2, nbdevdn=2)

    # momentum indicators
    df['adx'] = ta.ADX(df.high, df.low, df.close, timeperiod=14)
    df['cci'] = ta.CCI(df.high, df.low, df.close, timeperiod=14)
    df['mfi']  = ta.MFI(df.high, df.low, df.close, df.volume, timeperiod=14)
    df['rocp']  = ta.ROCP(df.close, timeperiod=10)
    df['rsi7']  = ta.RSI(df.close, timeperiod=7)
    df['rsi14'] = ta.RSI(df.close, timeperiod=14)
    df['rsi21'] = ta.RSI(df.close, timeperiod=21)
    df['aroondown'], df['aroonup'] = ta.AROON(df.high, df.low, timeperiod=14)
    df['macd'], df['macdsignal'], df['macdhist'] = ta.MACD(df.close, fastperiod=12, slowperiod=26, signalperiod=9)

    # volume indicators
    df['ad'] = ta.AD(df.high, df.low, df.close, df.volume)

    # volatility indicators
    df['nart'] = ta.NATR(df.high, df.low, df.close, timeperiod=14)

    # cycle indicators
    df['dcperiod'] = ta.HT_DCPERIOD(df.close)
    df['dcphase'] = ta.HT_DCPHASE(df.close)
    df['inhpase'], df['quadrature'] = ta.HT_PHASOR(df.close)
    df['sine'], df['leadsine'] = sine, leadsine = ta.HT_SINE(df.close)
    df['trendmode'] = ta.HT_TRENDMODE(df.close)

    return df

print(factors(data))