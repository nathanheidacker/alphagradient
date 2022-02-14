# -*- coding: utf-8 -*-
"""AG module containing classes for holding and organizing assets

Todo:
    * na
"""
# Standard imports
from numbers import Number

# Third Party imports
import pandas as pd
import numpy as np

# Local Imports
from .._globals import __globals as gbs
from .asset import Asset, types
from .portfolio import Portfolio, Cash
from .standrd import Currency, Stock, BrownianStock, Call, Put

class Basket:
    
    def __init__(self, start=None, end=None, resolution=None, assets=None, portfolios=None):
        pass


class Universe:
    pass
