# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


#import numpy as np
import threading
import time
from mpplot import NBPlot

print_lock = threading.Lock()

def main():
    pl = NBPlot()
    for ii in range(1,100):
        pl.plot(x_y = [[1,2,3],[x*ii for x in range(1,4)]])
        time.sleep(0.5)
        print('stuff')
    pl.plot(finished=True)
if __name__=='__main__':
    main()
print('I Shall Slain My Daemonic Children')