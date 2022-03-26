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

class Performance:

    def __init__(self, portfolio):
        self.data = portfolio.history
        self.base = portfolio.base
        self.symbol = portfolio.cash.symbol

    def changes(self, date=None, header_info=False):
        date = self.date if date is None else date
        index = self.data.index
        value = index.asof(date)
        ind = index.get_loc(value)
        current = self.data.iloc[ind]
        previous_name = None
        if len(self.data) < 2:
            previous_positions = [pos.empty() for pos in current["POSITIONS"]]
            previous = {"POSITIONS": previous_positions, "VALUE": 0}
            previous_name = current.name
        else:
            previous = self.portfolio.history.iloc[ind - 1]

        changes = {pos.key: [pos.quantity, 0] for pos in previous["POSITIONS"]}

        for pos in current["POSITIONS"]:
            if pos.key in changes:
                if pos.quantity == changes[pos.key][0]:
                    changes.pop(pos.key)
                else:
                    changes[pos.key][1] = pos.quantity
            else:
                changes[pos.key] = [0, pos.quantity]

        changes = {k: tuple(v) for k, v in changes.items()}
        prev_value = round(previous["VALUE"], 2)
        prev_value = f"{self.symbol}{prev_value}"
        curr_value = round(current["VALUE"], 2)
        curr_value = f"{self.symbol}{curr_value}"
        changes["TOTAL VALUE"] = (prev_value, curr_value)

        if header_info:
            return changes, current.name, previous_name or previous.name
        return changes

    def change_report(self, date=None, header=False, border=False):
        changes, current, previous = self.changes(date, header_info=True)
        result = ""
        title = ""
        border = ""

        if header:
            if current == previous:
                if current == self.data.index[0]:
                    title = f"CHANGES FROM INITIALIZATION ON {current.date()} {utils.timestring(current)}, {utils.get_weekday(current)[:3]}\n"
                else:
                    return
            else:
                title = f"CHANGES FROM {previous.date()} {utils.timestring(previous)}, {utils.get_weekday(previous)[:3]} --> {current.date()} {utils.timestring(current)}, {utils.get_weekday(current)[:3]}\n"

        entries = []
        m = max([len(key) for key in changes]) + 1
        if m:
            before_len = max([len(str(pos[0])) for pos in changes.values()])
            for key, change in changes.items():
                diff = " " * (m - len(key))
                before = " " * (before_len - len(str(change[0])))
                before = f"{change[0]}{before}"
                entries.append(f"{key}{diff}: {before} --> {change[1]}")

        if border:
            if title:
                border = "-" * len(title) + "\n"
            elif entries:
                border = "-" * max(len(entry) for entry in entries)
            border += "\n"

        changes = ""
        for entry in entries:
            changes += f"{entry}\n"

        return f"{title}{border}{changes}{border}"[:-1]

class Run:

    def __init__(self, env, start=None, end=None):
        self.env = env
        self.start = start or env.start
        self.end = end or env.end
        self.duration = self.start - self.end

        self.timesteps = 0
        self.timesteps_today = 0
        self.complete = False
        self.performances = {name: Performance(pf) for name, pf in env.portfolios.items()}

        if self.env.times:
            total = len(self.env.times) * self.duration.days
            getter = lambda self: round(((self.timesteps / total) * 100), 2)
            self.completion = property(getter)

        else:
            total = self.duration.days
            getter = lambda self: round(((self.days / total) * 100), 2)
            self.completion = property(getter)

    def finalize(self):
        del self.env
        self.complete = True
        self.competion = property(lambda self: 100)

    @property
    def day(self):
        return (env.date - self.start).days


class Stats:

    def __init__(self):
        self.runs = []
        self.current = None

    def reset(self, env, start, end):
        if self.current:
            self.current.finalize()
            self.runs.append(current)
        self.current = Run(env)

    def step_report(self):
        pass

    def __bool__(self):
        return True


class Algorithm(ABC):

    def __init__(self, start=None, end=None, resolution=None, verbose=False):
        self.start = glbs.start if start is None else self.validate_date(start)
        self.end = glbs.end if end is None else self.validate_end(self.start, end)
        self.resolution = glbs.resolution if resolution is None else self.validate_resolution(resolution)
        self._environment = self.setup(self.start, self.end)
        self.verbose = print if verbose else lambda *args, **kwargs: None
        self.stats = Stats()

    def __call__(self, *args, start=None, end=None, **kwargs):

        # Initializing start and end
        start = self.start if start is None else self.validate_date(start)
        self.env.date = start
        end = self.end if end is None else self.validate_end(start, end)

        # Reset stats for new run
        self.stats.reset(self.env, start, end)

        # Dont run setup on first call
        if self.stats.runs:
            self.env = self.setup(*args, start=start, end=end, **kwargs)

        # Ensure that end is bounded the env, prevent infinite loop
        end = end if end < self.env.end else self.env.end

        # Run with args
        self.run(*args, start=start, end=end, **kwargs)

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

    @property
    def active(self):
        return self.env.open and not self.stats.current.complete

    @env.setter
    def env(self, environment):
        if not isinstance(environment, Basket):
            raise TypeError("Environment must be a basket object, received {environment=}")
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

    def validate_end(self, start, end):
        try:
            return self.validate_date(end)
        except TypeError as e:
            if isinstance(end, Number):
                end = start + timedelta(days=end)

            elif isinstance(date, timedelta):
                end = start + end

            if not isinstance(end, datetime):
                raise e

            return end

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
