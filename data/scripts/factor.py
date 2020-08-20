import argparse
import glob
import os

import pandas as pd
import talib as ta


def calculate_indicators(data, inplace=False):
    df = data.copy() if inplace == False else data

    # overlap studies
    df['ma7']  = ta.MA(data.close, timeperiod=7)
    df['ma14'] = ta.MA(data.close, timeperiod=14)
    df['ma21'] = ta.MA(data.close, timeperiod=21)
    df['pct_chg'] = ta.ROC(data.close, timeperiod=1)
    df['upper'], df['mid'], df['lower'] = ta.BBANDS(data.close, timeperiod=5, nbdevup=2, nbdevdn=2)

    # momentum indicators
    df['adx'] = ta.ADX(data.high, data.low, data.close, timeperiod=14)
    df['cci'] = ta.CCI(data.high, data.low, data.close, timeperiod=14)
    df['mfi']  = ta.MFI(data.high, data.low, data.close, data.volume, timeperiod=14)
    df['rsi7']  = ta.RSI(data.close, timeperiod=7)
    df['rsi14'] = ta.RSI(data.close, timeperiod=14)
    df['rsi21'] = ta.RSI(data.close, timeperiod=21)
    df['aroondown'], df['aroonup'] = ta.AROON(data.high, data.low, timeperiod=14)
    df['macd'], df['macdsignal'], df['macdhist'] = ta.MACD(data.close, fastperiod=12, slowperiod=26, signalperiod=9)

    # volume indicators
    df['ad'] = ta.AD(data.high, data.low, data.close, data.volume)

    # volatility indicators
    df['nart'] = ta.NATR(data.high, data.low, data.close, timeperiod=14)

    # cycle indicators
    df['dcperiod'] = ta.HT_DCPERIOD(data.close)
    df['dcphase'] = ta.HT_DCPHASE(data.close)
    df['inhpase'], df['quadrature'] = ta.HT_PHASOR(data.close)
    df['sine'], df['leadsine'] = sine, leadsine = ta.HT_SINE(data.close)
    df['trendmode'] = ta.HT_TRENDMODE(data.close)

    if inplace == False:
        return df


def parse_args():
  parser = argparse.ArgumentParser(description= 'calculate technical indicators')
  parser.add_argument('-i', '--data_dir', required=True, type=str, help='direction of data file')
  parser.add_argument('-o', '--save_dir', required=True, type=str, help='direction of output file')
  return parser.parse_args()


# python -u ./data/scripts/indicator.py -i ./data/raw/price -o data/processed/indicators
if __name__ == '__main__':
    params = parse_args()

    if os.path.isdir(params.data_dir):
        csvlist = glob.glob(os.path.join(params.data_dir, '*.csv'))
    elif os.path.isfile(params.data_dir):
        csvlist = [params.data_dir]
    else:
        raise KeyError('unknown data direction')

    os.makedirs(params.save_dir, exist_ok=True)

    print('load data from %s, save to %s.' % (params.data_dir, params.save_dir))

    for i, csvfile in enumerate(csvlist):
        price = pd.read_csv(csvfile)
        data  = calculate_indicators(price)
        data  = data.dropna(axis=0)
        filename = os.path.basename(csvfile)
        data.to_csv(os.path.join(params.save_dir, filename), index=False, float_format='%.5f')
        print('[%d/%d] %s was preprocessed.' % (i+1, len(csvlist), filename))
        
    print('All files have been preprocessed.')