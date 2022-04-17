from datetime import datetime
import unittest
import os
import sys
import inspect
import alphagradient as ag

class Standard(unittest.TestCase):

    def test_stock(self):
        self.assertEquals(1, 1)

if __name__ == '__main__':
    unittest.main()
