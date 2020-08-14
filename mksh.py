##!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@Author: Pangjing Wu
Last Update: 2019-10-27
'''
import os
import sys

stocks = ['0001.HK', '0012.HK', '0016.HK', '0002.HK', '0003.HK', '0006.HK',
          '0005.HK', '0011.HK', '2388.HK', '0013.HK', '0762.HK', '0941.HK']

def main(argv):
    if len(argv) <= 1:
        filename, logpath = 'run.sh', 'logs'
    elif len(argv) == 2:
        filename, logpath = argv[1], 'logs'
    else:
        filename, logpath = argv[1], argv[2]
    
    os.makedirs(logpath, exist_ok=True)
    with open(filename, 'w') as f:
        for stock in stocks:
            f.write('python3 main.py %s >%s/%s.log 2>&1\n' % (stock, logpath, stock))

if __name__ == '__main__':
    main(sys.argv)