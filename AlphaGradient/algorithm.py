'''
This file contains the class for the base Algorithm, 
from which all AlphaGradient algorithms should inherit.

All Algorithms must define a run method, which specifies the 
algorithm's operation to be performed on the portfolio that is passed in.
'''
from abc import ABC, abstractmethod
from . import finance


class Algorithm(ABC):

    def __init__(self, verbose=False):
        self.verbose = bool(verbose)
        self.verboseprint = print if self.verbose else lambda *args, **kwargs: None

    def __call__(self, *args, **kwargs):
        self.run(*args, **kwargs)

    @abstractmethod
    def run(*args, **kwargs):
        pass

    def intitialize_inputs(portfolio, start_date, end_date):
        pass
