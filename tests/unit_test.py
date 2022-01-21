from datetime import datetime
import unittest
import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

import AlphaGradient as ag

class Standard(unittest.TestCase):

    def test_stock(self):
        spy = ag.finance.Stock('SPY', date=datetime.today())
        print(spy.price)

        self.assertEquals(spy.price, 430.429993)

if __name__ == '__main__':
    unittest.main()
