# -*- coding: utf-8 -*-
"""AG module containing algorithm class

Todo:
    * Implement algorithms
"""

# Standard imports
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from numbers import Number

# Third party imports
import pandas as pd

# Local imports
from ._globals import __globals as glbs
from .finance import types, Basket, Portfolio

class Stats:

    def __init__(self, portfolio):
        pass

class Run:

    def __init__(self, env):
        pass


class Algorithm(ABC):

    def __init__(self, start=None, end=None, resolution=None, verbose=False):
        self.start = glbs.start if start is None else self.validate_date(start)
        self.end = glbs.end if end is None else self.validate_date(end)
        self.resolution = glbs.resolution if resolution is None else self.validate_resolution(resolution)
        self._runs = []
        self._environment = Basket()
        self.verbose = print if verbose else lambda *args, **kwargs: None

    def __call__(self, *args, start=None, end=None, **kwargs):
        start = self.start if start is None else self.validate_date(start)
        end = self.end if end is None else self.validate_end(end)
        self.env = self.setup(*args, start=start, end=end, **kwargs)
        self.run(*args, start=start, end=end, **kwargs)
        self._runs.append(Run(self.env))

    @abstractmethod
    def setup(self, *args, **kwargs):
        return Basket()

    @abstractmethod
    def run(self, *args, **kwargs):
        return Portfolio(0, date=self.start)

    @property
    def env(self):
        return self._environment

    @property
    def date(self):
        return self.env.date

    @env.setter
    def env(self, environment):
        if not isinstance(environment, Basket):
            raise TypeError(f"Environment must be a basket object, received {environment=}")
        self._environment = environment

    def intitialize_inputs(self, start=None, end=None):
        start = self.start if start is None else validate_date(start)
        end = self.end if end is None else self.validate_end(end)

    @staticmethod
    def validate_date(date):
        if isinstance(date, str):
            date = datetime.fromisoformat(date)

        elif isinstance(date, pd.Timestamp):
            date = date.to_pydatetime()

        if not isinstance(date, datetime):
            raise TypeError(f"Unable to convert {date=} to datetime object")

        return date

    def validate_end(self, date):
        try:
            return self.validate_date(date)
        except TypeError as e:
            if isinstance(date, Number):
                date = self.start + timedelta(days=date)

            elif isinstance(date, timedelta):
                date = self.start + date

            if not isinstance(date, datetime):
                raise e

            return date

    def validate_resolution(self, delta):
        if isinstance(delta, Number):
            return timedelta(days=delta)

        elif isinstance(delta, timedelta):
            return delta

        else:
            raise TypeError(f"Invalid resolution type {delta=}. "
                            "Resolution must be a number or timedelta")



setattr(Algorithm, "type", types.algorithm)
setattr(types.algorithm, 'c', Algorithm)
