import pandas as pd
import matplotlib.pyplot as plt


def backtest(decision:list, open_price:list, close_price:list, init_money=100):
    money = [init_money]
    for i, d in enumerate(decision):
        if d == 1: 
            profit = close_price[i] - open_price[i]
        elif d == -1:
            profit = open_price[i] - close_price[i]
        else:
            continue
        money.append(round(money[-1] + profit, 3))
    return money


if __name__ == '__main__':
    STOCKS  = ['0001.HK', '0012.HK', '0016.HK', '0002.HK', '0003.HK', '0006.HK',
               '0005.HK', '0011.HK', '2388.HK', '0013.HK', '0762.HK', '0941.HK']
    lexicon = 'SenticNet5 ' # ['LMFinance', 'SenticNet5']

    fig, ax = plt.subplots(nrows=4, ncols=3, sharey=True, figsize=(10,7))

    for ax_i, stock in zip(ax.flatten(), STOCKS):
        data     = pd.read_csv('./data/processed/indicators/%s.csv' % stock, index_col='date', parse_dates=True)
        decision = pd.read_csv('./results/%s/%s.csv' % (lexicon, stock), index_col='date', parse_dates=True)
        data     = data.loc[decision.index]
        baseline = backtest(decision['baseline'], data['open'].values, data['close'].values)
        ours     = backtest(decision['ours'], data['open'].values, data['close'].values)
        ax_i.grid(linestyle='--')
        ax_i.plot(ours, c='C1', label='MS-SVM')
        ax_i.plot(baseline, c='C0', label='Baseline')
        ax_i.set_title(stock)
        ax_i.set_xticklabels([])
        ax_i.set_xlabel('Time')
        ax_i.legend(loc='lower left')
    plt.tight_layout()
    plt.show()