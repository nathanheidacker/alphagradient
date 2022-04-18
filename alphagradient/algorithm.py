# -*- coding: utf-8 -*-
"""AG module containing algorithm class

Todo:
    * Finish finalize method for Runs
        1) Calculate weights for each performance based on initial account values
        2) Aggregate total value data across all portfolios (for use in calculating best and worst metrics)
    * Profile() function to generate a comprehensive report
    * Best() and Worst() functions with timedelta as a parameter
        For all possible periods of length determined by the given timedelta, which period had the best or worst performance (change in total value)
        * Should be
"""

# Standard imports
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from numbers import Number
import math

# Third party imports
import pandas as pd
import numpy as np
from tqdm import tqdm

# Local imports
from .finance import types, Asset, Basket, Portfolio
from . import utils

class Performance:
    """The statistical profile of the performance of a single portfolio in a given Run"""
    def __init__(self, portfolio):
        self.portfolio = portfolio
        self.name = portfolio.name
        self._data = None
        self.base = portfolio.base
        self.symbol = portfolio.cash.symbol

    @property
    def data(self):
        # self._data will become the full portfolio history after the Performance is finalized (completed)
        return self._data if self._data is not None else self.portfolio.history

    def finalize(self):
        """Finalizes this performance after it's respective run has been completed, allowing the attached portfolio object to be garbage collected and providing a range of functionality

        This function is automatically called on every perforamnce within a backtest when the run finishes. It deletes the local reference to the portfolio from which it gathers historical data, allowing it to be garbage collected, and additional installs a range of statistical
        analysis and performance functionality on the instance
        """

        # Portfolio reference is needed while the run is active to be able to dynamically
        # reference the portfolio's history, but is no longer needed after the run finishes
        # Here, we just save the data so that the portfolio can be garbage collected
        self._data = self.portfolio.history
        del self.__dict__["portfolio"]

        # Gathering the first and last indices of the dataset, and attaching a reference
        # to the global benchmark
        self.start = self._data.index[0]
        self.end = self._data.index[-1]
        self.benchmark = Asset.benchmark.fget(1)

        # Calculating the pct change
        if len(self._data) > 1:
            shifted = self._data["VALUE"].shift(1)
            shifted[0] = shifted[1]
            self._data["CHANGE"] = (self._data["VALUE"] / shifted) - 1

        elif not self._data.empty:
            self._data["CHANGE"] = [0]

        else:
            self._data["CHANGE"] = []

        def get_index(date):
            """Given a date, returns the nearest index that is equal to or before the input date, bounded to 0"""
            if date <= self._data.index[0]:
                return 0
            return self._data.index.get_loc(self._data.index.asof(date))

        def get_benchmark_index(date, benchmark=None):
            """get_index for the global benchmark"""
            benchmark = self.benchmark if benchmark is None else benchmark
            if date <= benchmark.data.first:
                return 0
            return benchmark.data._data.index.get_loc(benchmark.data._data.index.asof(date))

        def get_dates(date=None, days=365):
            """given a date and timedelta as an int (number of days), returns the appropriate
            start and end dates for indexing"""
            start=None
            end=None
            if date is None:
                end = self.end
            else:
                end = date
            if days=="max":
                start = self.start
            else:
                start = end - timedelta(days=days)
            return start, end

        def _range(start, end):
            """Given a start and end date, safely returns an interval of the data corresponding to that range"""
            start = get_index(start)
            end = get_index(end)
            if end <= start:
                end = min(start + 1, len(self._data))
            return self._data.iloc[start:end, :]

        def benchmark_range(start, end, benchmark=None):
            """_range for the benchmark"""
            benchmark = self.benchmark if benchmark is None else benchmark
            start = get_benchmark_index(start, benchmark)
            end = get_benchmark_index(end, benchmark)
            if end <= start:
                end = min(start + 1, len(benchmark.data._data))
            return benchmark.data._data.iloc[start:end, :]

        def td_range(date=None, days=365):
            """range function that accepts date and timedelta as inputs"""
            return _range(*get_dates(date, days))

        def td_benchmark_range(date=None, days=365, benchmark=None):
            """td_range for the global benchmark"""
            return benchmark_range(*get_dates(date, days), benchmark)

        def quote(date):
            """Returns the backtest's total value on a given date"""
            return self._data["VALUE"].asof(date)

        def vol(date=None, days=365):
            """Returns the backtest's volatility over the specified period"""
            multiplier = ((days / (252 / 365)) ** 0.5)
            data = td_range(date, days)
            return data["CHANGE"].std() * multiplier

        def benchmark_vol(date=None, days=365, benchmark=None):
            """vol for the global benchmark"""
            benchmark = self.benchmark if benchmark is None else benchmark
            multiplier = ((days / (252 / 365)) ** 0.5)
            data = td_benchmark_range(date, days, benchmark)
            return data["CHANGE"].std() * multiplier

        def ma(date=None, days=365):
            """Returns the moving average over the previous period"""
            data = td_range(date, days)
            return data["VALUE"].mean()

        def beta(date=None, days=365, benchmark=None):
            """Returns the backtest's total beta value over the previous period"""
            benchmark = self.benchmark if benchmark is None else benchmark
            self_vol = vol(date, days)
            bench_vol = benchmark_vol(date, days, benchmark)
            self_data = td_range(date, days)["CHANGE"]
            bench_data = td_benchmark_range(date, days, benchmark)["CHANGE"].asof(self_data.index)
            r = self_data.corr(bench_data)
            if bench_vol == 0:
                return 0
            return r * (self_vol / bench_vol)

        def roi(date=None, days=365):
            """Returns the backtest's return on investment over the previous period"""
            start, end = get_dates(date, days)
            initial = quote(start)
            if initial == 0:
                return 0
            return (quote(end) / initial) - 1

        def benchmark_roi(date=None, days=365, benchmark=None):
            """roi function for the global benchmark"""
            benchmark = self.benchmark if benchmark is None else benchmark
            start, end = get_dates(date, days)
            initial = benchmark.quote(start)
            if initial == 0:
                return 0
            return (benchmark.quote(end) / initial) - 1

        def cagr(date=None, days=365):
            """Returns this backtest's compounding annual growth rate"""
            return (roi(date, days) + 1) ** (365 / days) - 1

        def alpha(date=None, days=365, benchmark=None):
            """Returns this backtest's alpha over the specified period, relative to a specified
            benchmark"""
            benchmark = self.benchmark if benchmark is None else benchmark
            run_roi = roi(date, days)
            bench_roi = benchmark_roi(date, days, benchmark)
            expected_roi = beta(date, days, benchmark) * (bench_roi - Asset.rfr.fget(1))
            return run_roi - expected_roi

        def periods(delta=365, start=None):
            """Beginning from start, returns a tuple of datetime intervals which correspond to the
            index from this backtest's data broken up into periods with interval length = delta"""
            start = self.start if start is None else start
            delta = timedelta(days=delta) if not isinstance(delta, timedelta) else delta
            periods = []
            while start < self.end:
                period_end = min(start + delta, self.end)
                periods.append((start, period_end))
                start = period_end
            return tuple(periods)

        def index_periods(delta=365, start=None):
            """Returns the same datetime interval tuple as periods, but intervals are instead represented with index values rather than datetimes"""
            return tuple([tuple([get_index(t) for t in period]) for period in periods(delta, start)])

        # Providing the additional functionality
        self.range = _range
        self.quote = quote
        self.vol = vol
        self.ma = ma
        self.beta = beta
        self.roi = roi
        self.cagr = cagr
        self.alpha = alpha
        self.periods = periods
        self.index_periods = index_periods


    def changes(self, date=None, *, steps=None, delta=None, header_info=False):
        """Returns a dictionary of changes from a portfolio's position history given a date"""

        # Variable Initialization
        date = self.date if date is None else date
        index = self.data.index
        value = index.asof(date)
        ind = index.get_loc(value)
        current = self.data.iloc[ind]
        previous_name = None

        # Handling step and delta inputs
        if steps is not None and delta is not None:
            raise ValueError(f"Received both a steps and delta input for changes. Only one of these inputs may be specified")

        # When delta is specified
        elif delta is not None:
            prev_date = date - delta
            prev_ind = 0
            if prev_date > index[0]:
                prev_value = index.asof(date - delta)
                prev_ind = index.get_loc(prev_value)
            steps = ind - prev_ind

        # When neither input is specified, default to a step size of 1
        elif steps is None:
            steps = 1

        # When called directly after initialization
        if len(self.data) <= 1:
            previous_positions = [pos.empty() for pos in current["POSITIONS"]]
            previous = {"POSITIONS": previous_positions, "VALUE": 0}
            previous_name = current.name

        elif ind - steps < 0:
            previous = self.data.iloc[0]

        # Safely decrement index by step count
        else:
            previous = self.data.iloc[ind - steps]

        # Gathering initial position states from the previous positions in portfolio history
        changes = {pos.key: [pos.quantity, 0] for pos in previous["POSITIONS"]}

        # Updating position states to reflect changes that lead to current state
        for pos in current["POSITIONS"]:

            # If the position currently held was held previously
            if pos.key in changes:

                # Remove the change if the quantity is unchanged (no transactions have occured
                # regarding this position)
                if pos.quantity == changes[pos.key][0]:
                    changes.pop(pos.key)

                # Quantity HAS changed, meaning a transaction has occured that needs to be recorded
                # Update the quantity in at the second index to reflect new quantity
                else:
                    changes[pos.key][1] = pos.quantity

            # If the position is a new position
            else:
                changes[pos.key] = [0, pos.quantity]

        # Transforming quantity change values from lists to tuples (immutability)
        changes = {k: tuple(v) for k, v in changes.items()}

        # Adding a change to reflect change in the total portfolio value
        # Added even when total portfolio value remains unchanged
        prev_value = round(previous["VALUE"], 2)
        prev_value = f"{self.symbol}{prev_value}"
        curr_value = round(current["VALUE"], 2)
        curr_value = f"{self.symbol}{curr_value}"
        changes["TOTAL VALUE"] = (prev_value, curr_value)

        # Return additional info for creating a header in a step report when requested
        if header_info:
            return changes, current.name, previous_name or previous.name
        return changes

    def change_report(self, date=None, *, steps=None, delta=None, header=False, border=False):
        """Returns a formatted list of changes from the given date to the index before it"""

        # Grabbing changes
        changes, current, previous = self.changes(date, steps=steps, delta=delta, header_info=True)

        # Initializing components of the final report
        title = ""

        if header:
            if current == previous:

                #Occurs when reporting the first index, ie during initialization
                if current == self.data.index[0]:
                    title = f"CHANGES FROM INITIALIZATION ON {current.date()} {utils.timestring(current)}, {utils.get_weekday(current)[:3]}\n"

                # Otherwise, no changes have occurred (dates are identical)
                else:
                    return
            else:
                title = f"CHANGES FROM {previous.date()} {utils.timestring(previous)}, {utils.get_weekday(previous)[:3]} --> {current.date()} {utils.timestring(current)}, {utils.get_weekday(current)[:3]}\n"

        # Formatting all change entries so that distances are equalized
        entries = []
        longest_pos = max([len(key) for key in changes]) + 1
        before_len = max([len(str(pos[0])) for pos in changes.values()])
        for pos, change in changes.items():
            diff = " " * (longest_pos - len(pos))
            before = " " * (before_len - len(str(change[0])))
            before = f"{change[0]}{before}"
            entries.append(f"{pos}{diff}: {before} --> {change[1]}")

        # Border is length of header if present, otherwise length of longest change entry
        if border:
            border = ""
            if title:
                border = "-" * len(title) + "\n"
            elif entries:
                border = "-" * max(len(entry) for entry in entries)

        # No border requested
        else:
            border = ""

        # String summing all change entries
        changes = ""
        for entry in entries:
            changes += f"{entry}\n"

        # Creating the full report
        report = f"{title}{border}{changes}{border}"

        # Removes last newline character (if no border is requested)
        if not border:
            return report[:-1]
        return report

class Run:
    """A statistical profile of an algorithm's performance across a single 'run', also called a
    backtest"""

    def __init__(self, env, start, end, algo_name):
        self._previous = self.date
        self.start = start
        self.end = end
        self.duration = self.end - self.start
        self.remaining = self.duration
        self.timesteps = 0
        self.timesteps_today = 0
        self.complete = False
        self.performances = {name: Performance(pf) for name, pf in env.portfolios.items()}
        self.main = self.performances[env.main.name]
        self.algo_name = algo_name

        # If the run's environment has been finalized, we can get a more precise measurement of
        # the completeness of the run by using timesteps over days
        if env.times:
            self._total = len(env.times) * self.duration.days
            step_getter = (lambda self: (self.timesteps, self._total)).__get__(self, Run)
            completion_getter = (lambda self: round(((self.timesteps / self._total) * 100), 2)).__get__(self, Run)
            self._step = step_getter
            self._completion = completion_getter

        # Calling __get__ with self and Run turns the func into a bound method of this Run instance

        else:
            self._total = self.duration.days
            step_getter = (lambda self: (self.day, self._total)).__get__(self, Run)
            completion_getter = (lambda self: round(((self.day / self._total) * 100), 2)).__get__(self, Run)
            self._step = step_getter
            self._completion = completion_getter
            # CAN PROPERTY TAKE IN AN OBJECT TO BIND TO?

    def __bool__(self):
        return True

    def __str__(self):
        return f"<{self.algo_name} Algorithm Backtest: {self.duration}>"

    def __repr__(self):
        return self.__str__()

    def __getattr__(self, attr):
        if self.performances:
            perf = list(self.performances.values())[0]
            if getattr(perf, attr, False):
                if utils.is_func(getattr(perf, attr)):
                    return self._aggregate(attr)
                return {pname : getattr(p, attr) for pname, p in self.performances}
        raise AttributeError(f"AlphaGradient Run object has no attribute '{attr}'")


    def finalize(self):
        """Updates the status of the run to complete and performs parallelized
        performance metric calculation"""
        self.complete = True
        for performance in self.performances.values():
            performance.finalize()

        data = []
        for name, performance in self.performances.items():
            series = performance.data["VALUE"]
            series.name = name
            data.append(series)

        self.history = pd.DataFrame(data).transpose()
        self.initial = self.history.iloc[0, :].to_dict()
        self.history["TOTAL"] = self.history.sum(axis=1)
        self.weights = {k : v / self.history.iloc[0, :]["TOTAL"] for k, v in self.initial.items()}
        self.history["CHANGE"] = self.history["TOTAL"].pct_change()

    @property
    def completion(self):
        if self.complete:
            return 100
        return self._completion()

    @property
    def day(self):
        return (self.date - self.start).days

    def update(self):
        """Updates the current run at each timestep"""

        # Only perform when the run is ongoing
        if not self.complete:

            # Incrementing steps
            self.timesteps += 1

            # Determining if the timesteps_today counter should be reset
            previous = utils.set_time(self._previous + timedelta(days=1), "0:00:00")
            if self.date >= previous:
                self.timesteps_today = 1
            else:
                self.timesteps_today += 1

            # Updating stored date
            self._previous = self.date
            self.remaining = self.end - self.date

    def completion_report(self):
        """Returns a small report of the state of completion of the current run"""
        counter, total = self._step()
        return f"DAY {self.day} | STEP {self.timesteps_today} | COMPLETION: {self.completion}% ({counter} / {total}) | REMAINING: {self.remaining}"

    def _aggregate(self, func_name):
        """When accessing an attribute or method that exists on a performance, returns a function that produces a dictionary with the attribute calculated for each performance contained within this backtest"""
        def aggregated(*args, aggregate=False, **kwargs):
            attr_dict = {k: getattr(v, func_name)(*args, **kwargs) for k, v in self.performances.items()}
            if aggregate:
                attr_dict = {k:v for k, v in attr_dict.items() if not math.isnan(v)}
                adj_weights = {k:v for k, v in self.weights.items() if k in attr_dict}
                new_total = sum(adj_weights.values())
                adj_weights = {k : v / new_total for k, v in adj_weights.items()}
                values = np.array(list(attr_dict.values()))
                weights = np.array(list(adj_weights.values()))
                return (values * weights).sum()
            return attr_dict

        return aggregated

    def profile(self):
        """Creates an HTML report of this backtest's performance

        TODO, currently just shows the total profit and roi

        Parameters:
            path (path-like): The path to output the performance report

        Returns:
            None
        """
        profit = self.history["TOTAL"][-1] - self.history["TOTAL"][0]
        roi = round(self.roi(days="max", aggregate=True) * 100, 2)
        return {"PROFIT": f"${round(profit, 2)}", "ROI": f"{roi}% | {round(roi / 100, 2) + 1}x"}


class Stats:

    def __init__(self, algo):
        self._runs = []
        self.current = None
        self._algoname = algo.__class__.__name__

    def _reset(self, env, start, end):
        self.current = Run(env, start, end, self._algoname)

    def _finish(self):
        self.current.finalize()
        self._runs.append(self.current)
        run = self.current
        self.current = None
        return run

    def change_report(self, date=None, *, steps=None, delta=None):
        """Returns a report of changes to the main portfolio resulting from the most recent timestep"""
        comp_report = self.current.completion_report()
        change_report = self.current.main.change_report(date=date,
                                                        steps=steps,
                                                        delta=delta,
                                                        header=True,
                                                        border=True)
        return f"{comp_report}\n{change_report}"

    @property
    def runs(self):
        if self.current.complete:
            return self._runs + [self.current]
        return self._runs

    @property
    def completion(self):
        return self.current.completion

    def update(self):
        """Updates the current run at each time step. This function is called by both global and local step() methods"""

        # Only perform is a run is currently progressing
        if self.current:
            self.current.update()

    def __bool__(self):
        return True


class Algorithm(ABC):
    """A base class that provides the functionality for creating backtestable alphagradient algorithms"""
    def __init__(self, start=None, end=None, resolution=None, verbose=False, progress=True):
        self.start = self._global_start if start is None else self.validate_date(start)
        self.end = self._global_end if end is None else self.validate_end(end)
        self.resolution = self._global_res if resolution is None else self.validate_resolution(resolution)
        self._environment = self.setup(start=self.start, end=self.end)
        self.verbose = verbose
        self.print = print if verbose else utils.NullClass()
        self.stats = Stats(self)
        self.type.instances[self._generate_name()] = self

        # Reroutes some traditional tqdm functionality to work automatically with Run objects
        # without requiring user input. Essentially a partial class version of tqdm
        class tqdm_partial(tqdm):
            def __init__(this, *args, total=None, **kwargs):
                total = self.stats.current._total if total is None else total
                this.ncols = 150
                super().__init__(*args, total=total, ncols=this.ncols, **kwargs)
                this.set_description(str(self.stats.current))
            def update(this, n=None):
                if n is None:
                    n = self.stats.current._step()[0] - this.n
                super().update(n=n)
            def close(this, *args, **kwargs):
                this.update(this.total - this.n)
                super().close(*args, **kwargs)

        self.progress = tqdm_partial if (progress and not verbose) else utils.NullClass()

    def __call__(self, *args, start=None, end=None, **kwargs):

        # Initializing start and end
        start = self.start if start is None else self.validate_date(start)
        end = self.end if end is None else self.validate_end(end, start=start)

        # Reset stats for new run
        self.stats._reset(self.env, start, end)

        # Dont run setup on first call (runs will be empty after initialization)
        if self.stats.runs:
            self.env = self.setup(*args, start=start, end=end, **kwargs)

        # Ensure that end is bounded the env, prevent infinite loop
        end = end if end < self.env.end else self.env.end

        # Ensure that the current global date is reset to the start
        self.env.sync(start)

        # Run with args
        self._run(*args, end=end, start=start, **kwargs)
        return self.stats._finish()

    def default_run(self, *args, start, end, **kwargs):
        """The default run cycle when none is implemented by an algorithm"""

        # Manual control of tqdm progress bar
        with self.progress() as progress:
            while self.date < end:
                if self.env.open:
                    self.cycle(*args, start=start, end=end, **kwargs)
                self.env.next()
                progress.update()

    @classmethod
    def _generate_name(cls):
        return len(cls.type.instances)

    @property
    def _run(self):
        if getattr(self, "run", False):
            return self.run
        return self.default_run

    @abstractmethod
    def setup(self, *args, **kwargs):
        """A required implementation of all new algorithms, sets up the local environment for the
        algorithm to operate on. Required to return and Environment object"""
        return Basket()

    @abstractmethod
    def cycle(self, *args, **kwargs):
        """A required implementatuon of all new algorithms, defines the algorithm's behavior at each new timestep. Should return None"""
        #return Portfolio(0, date=self.start)
        return None

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
        """Ensures that start and end parameter inputs are viable"""
        start = self.start if start is None else validate_date(start)
        end = self.end if end is None else self.validate_end(end, start=start)

    @staticmethod
    def validate_date(date):
        """Given a date-like object, converts it to an algorithm-viable datetime"""
        if isinstance(date, str):
            date = datetime.fromisoformat(date)

        elif isinstance(date, pd.Timestamp):
            date = date.to_pydatetime()

        if not isinstance(date, datetime):
            raise TypeError(f"Unable to convert {date=} to datetime object")

        return date

    def validate_end(self, end, start=None):
        """Given a date-like object, returns an algorithm-viable datetime to act as the ending date of an algorithm run, based on some start input"""
        start = self.start if start is None else start

        # See if the datetime can be validated normally
        try:
            return self.validate_date(end)

        # For convenience, end inputs have many more acceptable / coherent input types (in reference to some starting point)
        except TypeError as e:
            if isinstance(end, Number):
                end = start + timedelta(days=end)

            elif isinstance(date, timedelta):
                end = start + end

            # No conversion possible
            if not isinstance(end, datetime):
                raise e

            return end

    def validate_resolution(self, delta):
        """Given a timedelta-like object, converts it to a algorithm-viable timedelta object"""
        if isinstance(delta, Number):
            return timedelta(days=delta)

        elif isinstance(delta, timedelta):
            return delta

        else:
            raise TypeError(f"Invalid resolution type {delta=}. "
                            "Resolution must be a number or timedelta")



setattr(Algorithm, "type", types.algorithm)
setattr(types.algorithm, 'c', Algorithm)
