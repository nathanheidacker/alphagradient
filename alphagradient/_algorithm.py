# -*- coding: utf-8 -*-
"""Algorithm framework and other tools for tracking backtest performance

TODO:
    * Finish finalize method for backtests
        1) Calculate weights for each performance based on initial account values
        2) Aggregate total value data across all portfolios (for use in 
           calculating best and worst metrics)
    * Profile() function to generate a comprehensive report
    * Best() and Worst() functions with timedelta as a parameter
        For all possible periods of length determined by the given timedelta, 
        which period had the best or worst performance (change in total value)
"""

# Standard imports
from __future__ import annotations

from datetime import datetime, timedelta
from abc import ABC, abstractmethod
from numbers import Number
import math

# Third party imports
from tqdm import tqdm
import pandas as pd
import numpy as np

# Local imports
from ._finance import types, Asset, Environment, Portfolio
from . import utils

# Typing
from typing import (
    Any,
    Callable,
    Literal,
    Union,
    Optional,
    no_type_check,
)

from .utils import (
    DatetimeLike,
)

Changes = dict[str, tuple[float, float]]
"""Changes in a portfolio's positions across a time period"""

ChangesWithInfo = tuple[Changes, pd.Timestamp, pd.Timestamp]
"""Changes, but with information about the relevant time period"""

Max = Literal["max"]
"""Shortcut alias for literal 'max'"""


def _attr_error(cls, attr):
    return AttributeError(f"'{cls.__name__}' object has no attribute {repr(attr)}")


class Performance:
    """
    The statistical profile of a Portfolio's Backtest performance

    A Performance is an object which tracks the performance of a portfolio over
    the duration of an algorithm backtest. In this way, it is extremely similar
    to a Backtest and offers almost all of the same functionality, but is
    portfolio specific, whereas a Backtest is an aggregation of performance
    objects.

    These objects are not intended to be interacted with directly by end users,
    although they may be sometimes when the statisical measures supported by
    Backtest objects are insufficient.

    Parameters:
        portfolio:
            The portfolio tracked by this performance

    Attributes:
        base (str):
            A string representing a national currency code, representing the
            base currency used by the tracked portfolio. Default is 'USD'

        name (str):
            The name of the tracked portfolio

        portfolio (Portfolio):
            A reference to the portfolio that is being tracked. This reference
            is deleted when the portfolio is finalized. This means that
            accessing self.portfolio after finalization will raise an
            AttributeError, rather than return None

        symbol (str):
            The currency symbol corresponding to the base currency (self.base)
    """

    def __init__(self, portfolio: Portfolio) -> None:
        self.portfolio: Portfolio = portfolio
        self.name: str = portfolio.name
        self._data: Optional[pd.DataFrame] = None
        self.base: str = portfolio.base
        self.symbol: str = portfolio.cash.symbol

    @property
    def data(self) -> pd.DataFrame:
        """
        A performance's historical data

        A dataframe with data from the portfolio's performance history.
        During algorithm runtime, this contains an actual reference to the
        portfolio history as it is being updated. After a performance is
        finalized, it is a static dataframe.
        """

        # self._data will become the full portfolio history after the
        # Performance is finalized. Performances are automatically finalized
        # upon algorithm runtime completion
        return self._data if self._data is not None else self.portfolio.history

    def alpha(
        self,
        date: Optional[datetime] = None,
        days: Union[Max, float] = 365,
        benchmark: Optional[Asset] = None,
    ) -> float:
        """
        Returns this backtest's alpha over the specified period

        Calculates this performance's alpha by returning the actual roi
        less some expected roi value. Expected roi is the benchmark roi
        less the global risk free rate, weighted by the performance's beta
        using the benchmark asset as a benchmark.

        This function is only available after finalization.

        Parameters:
            date:
                The date from which to calculate alpha

            days:
                The duration of the date interval

            benchmark:
                The asset used as a benchmark

        Returns:
            The Performance's return on investment less its expected roi
        """
        raise _attr_error(Performance, "alpha")

    def beta(
        self,
        date: Optional[datetime] = None,
        days: Union[Max, float] = 365,
        benchmark: Optional[Asset] = None,
    ) -> float:
        """
        Returns the backtest's total value over the previous period

        Returns a historical beta calculation for the defined date interval.
        This is a ratio of this performance's historical volatility to that
        of some benchmark asset, weighted by the correlation of the two
        datasets

        This function is only available after finalization.

        Parameters:
            date:
                The date from which to calculate beta

            days:
                The duration in days of the interval to use in the beta calculation

            benchmark:
                The asset to compare this performance against

        Returns:
            The beta value for the defined period against the benchmark
        """
        raise _attr_error(Performance, "beta")

    def cagr(
        self,
        date: Optional[datetime] = None,
        days: Union[Max, float] = 365,
    ) -> float:
        """
        Returns this backtest's compounding annual growth rate

        Returns this performance's compounding annual growth rate
        extrapolated from the given period. Calculated by returning a
        return on investment for the given period, and extrapolating that
        value to an annualized period

        This function is only available after finalization.

        Parameters:
            date:
                The date from which to calculate cagr

            days:
                The duration of the interval

        Returns:
            Anuualized roi for the period (compounding annual growth rate)
        """
        raise _attr_error(Performance, "cagr")

    def changes(
        self,
        date: Optional[datetime] = None,
        *,
        steps: Optional[int] = None,
        delta: Optional[timedelta] = None,
        header_info: bool = False,
    ) -> Union[Changes, ChangesWithInfo]:
        """
        Returns a dictionary of changes from a portfolio's position history

        Given either an argument for the number of steps to backtrack of the
        amount of time to backtrack, determines an interval of time using
        'date' as the endpoint. Returns a dictionary of changes, where keys
        are the asset keys and values a two-values tuple containing initial
        values and end values of that asset. If requested, can also return
        datetime information for the beginning and end for use in producing a
        header (reporting)

        Note that of optional keyword arguments 'steps' and 'delta', only ONE
        may be defined

        Types Explanation:
            * Changes (dict[str, tuple[float, float]]):
                A dictionary of changes for the period

            * ChangesWithInfo (tuple[Changes, end: pd.Timestamp, start: pd.Timestamp]):
                Changes with header information

        Parameters:
            date:
                The date defining the end of the period. Defaults to the
                current date if none is provided

            steps:
                The number of steps to lookback

            delta:
                The amount of time to lookback

            header_info:
                Whether or not to return header information

        Returns:
            Union[Changes, ChangesWithInfo]:
                By default, returns a dictionary of changes for the period. If
                the parameter header_info is passed in as True, returns a tuple
                containing the change dictionary, an ending datetime, and a
                beginning datetime for the period from which changes were
                derived.
        """

        # Variable Initialization
        date = self.date if date is None else date  # type: ignore[attr-defined]
        index = self.data.index
        value = index.asof(date)
        ind = index.get_loc(value)
        current = self.data.iloc[ind]
        previous_name = None

        # Handling step and delta inputs
        if steps is not None and delta is not None:
            raise ValueError(
                "Received both a steps and delta input for changes. "
                "Only one of these inputs may be specified"
            )

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
        changes: dict[str, Any]
        changes = {pos.key: [pos.quantity, 0] for pos in previous["POSITIONS"]}  # type: ignore[attr-defined]

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
        changes: dict[str, tuple[float, float]]  # type: ignore[no-redef]
        changes = {k: tuple(v) for k, v in changes.items()}

        # Adding a change to reflect change in the total portfolio value
        # Added even when total portfolio value remains unchanged
        prev_value = round(previous["VALUE"], 2)  # type: ignore[call-overload]
        prev_value = f"{self.symbol}{prev_value}"
        curr_value = round(current["VALUE"], 2)
        curr_value = f"{self.symbol}{curr_value}"
        changes["TOTAL VALUE"] = (prev_value, curr_value)

        # Return additional info for creating a header in a step report when requested
        if header_info:
            return (
                changes,
                current.name,
                previous_name or previous.name,  # type: ignore[attr-defined]
            )
        return changes

    def change_report(
        self,
        date: Optional[datetime] = None,
        *any: Any,
        steps: Optional[int] = None,
        delta: Optional[timedelta] = None,
        header: bool = False,
        border: bool = False,
    ) -> str:
        """
        Returns a formatted report of the changes that occured in this performance

        Given either an argument for the number of steps to backtrack of the
        amount of time to backtrack, determines an interval of time using
        'date' as the endpoint. Using this interval, produces a formatted
        'report' string of the changes from the beginning to the end of the
        interval.

        Note that of optional keyword arguments 'steps' and 'delta', only ONE may be defined

        Optional settings include:
            #. | Producing a header, which will include information about the
               | period to which the report pertains

            #. | Producing borders inbetween the header and the report, as well
               | as at the bottom of the report. If many reports are logged in
               | succession, these dramatically improve readability.

        Parameters:
            date:
                The date defining the end of the period. Defaults to the
                current date if none is provided.

            steps:
                The number of steps to lookback

            delta:
                The amount of time to lookback

            header:
                Whether or not to produce header information in the report string

            border:
                Whether or not to produce borders in the report string

        Returns:
            The report of the changes for the defined period
        """

        # Grabbing changes
        info: ChangesWithInfo  # Our call is GUARANTEED to be ChangesWithInfo, not Changes
        info = self.changes(date, steps=steps, delta=delta, header_info=True)  # type: ignore[assignment]
        changes, current, previous = info

        # Initializing components of the final report
        title = ""

        if header:
            if current == previous:

                # Occurs when reporting the first index, ie during initialization
                if current == self.data.index[0]:
                    title = (
                        f"CHANGES FROM INITIALIZATION ON {current.date()} "
                        f"{utils.timestring(current)}, "
                        f"{utils.get_weekday(current)[:3]}\n"
                    )

                # Otherwise, no changes have occurred (dates are identical)
                else:
                    return ""
            else:
                title = (
                    f"CHANGES FROM {previous.date()} "
                    f"{utils.timestring(previous)}, "
                    f"{utils.get_weekday(previous)[:3]} --> {current.date()} "
                    f"{utils.timestring(current)}, {utils.get_weekday(current)[:3]}\n"
                )

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
            _border = ""
            if title:
                _border = "-" * len(title) + "\n"
            elif entries:
                _border = "-" * max(len(entry) for entry in entries)

        # No border requested
        else:
            _border = ""

        # String summing all change entries
        _changes = ""
        for entry in entries:
            _changes += f"{entry}\n"

        # Creating the full report
        report = f"{title}{_border}{_changes}{_border}"

        # Removes last newline character (if no border is requested)
        if not border:
            return report[:-1]
        return report

    def finalize(self) -> None:
        """
        Finalizes the performance after backtest completion

        This function is automatically called on every perforamnce within a
        backtest when the backtest finishes. It deletes the local reference to
        the portfolio from which it gathers historical data, allowing it to be
        garbage collected, and additional installs a range of statistical
        analysis and performance functionality on the instance

        Returns:
            Modifies this Performance in place.
        """

        # Portfolio reference is needed while the backtest is active to be
        # able to dynamically reference the portfolio's history, but is no
        # longer needed after the backtest finishes. Here, we just save the
        # data so that the portfolio can be garbage collected
        self._data = self.portfolio.history
        del self.__dict__["portfolio"]

        # Gathering the first and last indices of the dataset, and attaching
        # a reference to the global benchmark
        self.start = self._data.index[0]
        self.end = self._data.index[-1]
        self.benchmark = Asset.benchmark.fget(1)  # type: ignore[attr-defined]

        # Calculating the pct change
        if len(self._data) > 1:
            shifted = self._data["VALUE"].shift(1)
            shifted[0] = shifted[1]
            self._data["CHANGE"] = (self._data["VALUE"] / shifted) - 1

        elif not self._data.empty:
            self._data["CHANGE"] = [0]

        else:
            self._data["CHANGE"] = []

        # --- NEW FUNCTIONALITY AFTER FINALIZATION DEFINED BELOW THIS POINT ---

        def get_index(date: datetime) -> int:
            """
            Returns the nearest index that is equal to or before the input date

            Given a datetime object 'date', finds the nearest value in the
            time-series index of self.data which is on or before 'date'. Then,
            returns the numerical index corresponding to the identified
            time-series index.

            If the input is before the beginning of the date interval, returns 0
            rather than raising a keyerror

            This function is only available after finalization.

            Parameters:
                date (datetime): the datetime whose index to find

            Returns:
                index value (int): The numerical index corresponding to the date
            """

            # Return the first available date if the given date is before the
            # beginning of the data
            if date <= self.data.index[0]:
                return 0

            # Performing asof on the index is far more performant than on the
            # dframe
            return self.data.index.get_loc(self.data.index.asof(date))

        def _get_benchmark_index(
            date: datetime, benchmark: Optional[Asset] = None
        ) -> int:
            """
            get_index for the global benchmark, or any input benchmark
            """

            # Ensuring that benchmark is an asset
            benchmark = self.benchmark if benchmark is None else benchmark
            assert benchmark.data is not None

            # Return the first available date if the given date is before the
            # beginning of the data
            if date <= benchmark.data.first:
                return 0

            # Performing asof on the index is far more performant than on
            # the dframe
            return benchmark.data._data.index.get_loc(
                benchmark.data._data.index.asof(date)
            )

        def get_dates(
            date: Optional[datetime] = None, days: Union[Max, float] = 365
        ) -> tuple[datetime, datetime]:
            """
            Returns the appropriate start and end dates for indexing

            Given a datetime and a number of days, returns a tuple containing
            the corresponsding date interval. The user may pass in "max" for
            days to start the interval at the Performance data's start.
            Passing in no argument for days will default to 365. Passing in
            no date parameter will begin the interval at the Performance
            data's end.

            This function is only available after finalization.

            Parameters:
                date:
                    A datetime or "max"

                days:
                    An integer representing the length of the interval

            Returns:
                The interval of time defined by the inputs
            """
            start: datetime
            end: datetime

            # By default, the end is the end of the available data.
            if date is None:
                end = self.end
            else:
                end = date

            # special argument "max" chooses the first available date
            if days == "max":
                start = self.start
            else:
                days = float(days)
                start = end - timedelta(days=days)

            # Returning the interval
            return start, end

        def _range(start: datetime, end: datetime) -> pd.DataFrame:
            """
            Safely returns an interval of the data corresponding to that range

            Given a starting and ending datetime, returns all of the data in
            this Performance's data which exists inside the corresponding
            interval.

            This function is only available after finalization.

            Parameters:
                start:
                    The beginning of the interval

                end:
                    The end of the interval

            Returns:
                The interval of this performance's data corresponding to the
                range defined by the inputs
            """
            i_start = get_index(start)
            i_end = get_index(end)
            if i_end <= i_start:
                i_end = min(i_start + 1, len(self.data))
            return self.data.iloc[i_start:i_end, :]

        def benchmark_range(
            start: datetime, end: datetime, benchmark: Optional[Asset] = None
        ) -> pd.DataFrame:
            """_range for the benchmark. Accepts other benchmarks as inputs"""
            benchmark = self.benchmark if benchmark is None else benchmark
            assert benchmark.data is not None

            # Getting index values of start and end
            i_start: int = _get_benchmark_index(start, benchmark)
            i_end: int = _get_benchmark_index(end, benchmark)

            # End must be AFTER the start
            if i_end <= i_start:
                i_end = min(i_start + 1, len(benchmark.data._data))

            return benchmark.data._data.iloc[i_start:i_end, :]

        def td_range(
            date: Optional[datetime] = None, days: Union[Max, float] = 365
        ) -> pd.DataFrame:
            """
            Range function that accepts date and timedelta as inputs

            Timedelta based range operation. This function is identical to
            _range in its operation but accepts the same arguments as get_dates.

            This function is only available after finalization.

            Parameters:
                date:
                    The date defining the end of the range

                days:
                    defines the length of the time interval for the data range

            Returns:
                The interval of this performance's data corresponding to the
                range defined by the inputs
            """

            # get_dates returns an interval that can be unpacked into range,
            # flipping the arguments
            return _range(*get_dates(date, days))

        def td_benchmark_range(
            date: Optional[datetime] = None,
            days: Union[Max, float] = 365,
            benchmark: Optional[Asset] = None,
        ) -> pd.DataFrame:
            """td_range for the global benchmark"""
            return benchmark_range(*get_dates(date, days), benchmark)

        def quote(date: datetime) -> float:
            """
            Returns the backtest's total value on a given date

            Returns this performance's total value as of the given datetime.

            This function is only available after finalization.

            Parameters:
                date:
                    The date to valuate on

            Returns:
                value:
                    The total value of this performance's tracked portfolio
                    as of the given datetime
            """

            # total value data for a performance is stored in "VALUE"
            # performing asof on the series by accessing "VALUE" first, must faster
            return self.data["VALUE"].asof(date)

        def vol(
            date: Optional[datetime] = None, days: Union[Max, float] = 365
        ) -> float:
            """
            Returns the backtest's volatility over the specified period

            Returns a calculation of this perfomance's historical volatility
            across the interval defined by the inputs. Historical volatility
            for assets have a multiplier defined by the number of business
            days in a year

            This function is only available after finalization.

            Parameters:
                date:
                    The historical volatility as of this date

                days:
                    The duration of the interval to use for the volatility
                    calculation

            Returns:
                The historical volatility for the given period
            """
            # Getting the period duration
            if date is None:
                date = self.end

            if days == "max":
                duration = (date - self.start).days
            else:
                duration = days

            # The number of business days in a year
            multiplier = (duration / (252 / 365)) ** 0.5

            # The relevant range of data
            data = td_range(date, days)

            return data["CHANGE"].std() * multiplier

        def benchmark_vol(
            date: Optional[datetime] = None,
            days: Union[Max, float] = 365,
            benchmark: Optional[Asset] = None,
        ) -> float:
            """vol for the global benchmark"""
            # Getting period information
            if date is None:
                date = self.end

            if days == "max":
                duration = (date - self.start).days
            else:
                duration = days

            # Ensuring that benchmark is an asset
            benchmark = self.benchmark if benchmark is None else benchmark

            # The number of business days in a year
            multiplier = (duration / (252 / 365)) ** 0.5

            # The relevant range of data
            data = td_benchmark_range(date, days, benchmark)
            return data["CHANGE"].std() * multiplier

        def ma(date: Optional[datetime] = None, days: Union[Max, float] = 365) -> float:
            """
            Returns the moving average over the previous period

            The moving average of this performance's total value across the
            given date interval.

            This function is only available after finalization.

            Parameters:
                date:
                    The date from which to calculate the ma

                days:
                    The duration of the date interval (number of days) from
                    which to calculate the ma

            Returns:
                This performance's average total value across the defined date
                interval (the 'moving average')
            """
            data = td_range(date, days)
            return data["VALUE"].mean()

        def beta(
            date: Optional[datetime] = None,
            days: Union[Max, float] = 365,
            benchmark: Optional[Asset] = None,
        ) -> float:
            """
            Returns the backtest's total value over the previous period

            Returns a historical beta calculation for the defined date interval.
            This is a ratio of this performance's historical volatility to that
            of some benchmark asset, weighted by the correlation of the two
            datasets

            This function is only available after finalization.

            Parameters:
                date:
                    The date from which to calculate beta

                days:
                    The duration in days of the interval to use in the beta calculation

                benchmark:
                    The asset to compare this performance against

            Returns:
                The beta value for the defined period against the benchmark
            """
            # Ensuring that benchmark is an asset
            benchmark = self.benchmark if benchmark is None else benchmark

            # Getting historical volatilities
            self_vol = vol(date, days)
            bench_vol = benchmark_vol(date, days, benchmark)

            # Getting the relevant data ranges for both self and benchmark
            self_data = td_range(date, days)["CHANGE"]
            bench_data = td_benchmark_range(date, days, benchmark)["CHANGE"].asof(
                self_data.index
            )

            # Calculating correlation
            r = self_data.corr(bench_data)

            # Safe return value in the case that we get a divbyzero error
            if bench_vol == 0:
                return 0
            return r * (self_vol / bench_vol)

        def roi(
            date: Optional[datetime] = None, days: Union[Max, float] = 365
        ) -> float:
            """
            Returns the backtest's return on investment over the previous period

            Given a date and a duration to define a date interval, returns the
            difference in total performance value from the beginning of the
            period to the end of the period as a proportion of the total value
            at the beginning.

            For safety purposes, a division by zero (total value at the
            beginning of the period is 0), will result in a return value of 0,
            rather than raising an error.

            This function is only available after finalization.

            Parameters:
                date:
                    The end of the date interval

                days:
                    The duration of the date interval

            Returns:
                The performance's return on investment (as a percentage) for
                the given date interval
            """
            # Getting the relevant interval
            start, end = get_dates(date, days)

            # The initial total value
            initial = quote(start)

            # Safe return value in case of divbyzero
            if initial == 0:
                return 0
            return (quote(end) / initial) - 1

        def benchmark_roi(
            date: Optional[datetime] = None,
            days: Union[Max, float] = 365,
            benchmark: Optional[Asset] = None,
        ) -> float:
            """roi function for the global benchmark"""

            # Ensuring that benchmark is an asset
            benchmark = self.benchmark if benchmark is None else benchmark

            # Getting the relevant interval
            start, end = get_dates(date, days)

            # Getting the initial total value
            initial = benchmark.quote(start)

            # Safe return value in case of divbyzero
            if initial == 0:
                return 0
            return (benchmark.quote(end) / initial) - 1

        def cagr(
            date: Optional[datetime] = None, days: Union[Max, float] = 365
        ) -> float:
            """
            Returns this backtest's compounding annual growth rate

            Returns this performance's compounding annual growth rate
            extrapolated from the given period. Calculated by returning a
            return on investment for the given period, and extrapolating that
            value to an annualized period

            This function is only available after finalization.

            Parameters:
                date:
                    The date from which to calculate cagr

                days:
                    The duration of the interval

            Returns:
                Anuualized roi for the period (compounding annual growth rate)
            """
            # Getting the interval duration
            if date is None:
                date = self.end
            if days == "max":
                duration = (date - self.start).days

            return (roi(date, days) + 1) ** (365 / duration) - 1

        def alpha(
            date: Optional[datetime] = None,
            days: Union[Max, float] = 365,
            benchmark: Optional[Asset] = None,
        ) -> float:
            """
            Returns this backtest's alpha over the specified period

            Calculates this performance's alpha by returning the actual roi
            less some expected roi value. Expected roi is the benchmark roi
            less the global risk free rate, weighted by the performance's beta
            using the benchmark asset as a benchmark.

            This function is only available after finalization.

            Parameters:
                date:
                    The date from which to calculate alpha

                days:
                    The duration of the date interval

                benchmark:
                    The asset used as a benchmark

            Returns:
                The Performance's return on investment less its expected roi
            """

            # Ensuring that benchmark is an asset
            benchmark = self.benchmark if benchmark is None else benchmark

            # Gathering roi values
            backtest_roi = roi(date, days)
            bench_roi = benchmark_roi(date, days, benchmark)

            # Calculating expected roi by weighting benchmark alpha against this performance's beta
            expected_roi = beta(date, days, benchmark) * (bench_roi - Asset.rfr.fget(1))  # type: ignore[attr-defined]

            # Alpha is the difference between actual and expected roi
            return backtest_roi - expected_roi

        def periods(
            delta: Union[float, timedelta] = 365, start: Optional[datetime] = None
        ) -> tuple[tuple[datetime, datetime], ...]:
            """
            Beginning from start, returns a tuple of datetime intervals which correspond to the
            index from this backtest's data broken up into periods with interval length = delta

            From the inputs, a duration of time 'delta' is determined. The date
            interval defined by self.end - start is then split into periods of
            length delta, where intervals are represented as
            tuple[start:datetime, end:datetime]. A tuple of interval objects
            covering the entire period is returned.

            This function is only available after finalization.

            Parameters:
                delta:
                    An object representing desired interval length

                start:
                    The beginning of the greater period to split into intervals
                    of length 'delta'

            Returns:
                The greater interval split into intervals of length delta.
                Intervals are tuple[start: datetime, end: datetime]
            """

            # Ensuring that start is a datetime
            _start = self.start if start is None else start

            # Ensuring that delta is a timedelta
            delta = timedelta(days=delta) if not isinstance(delta, timedelta) else delta

            # Getting the intervals
            periods = []
            while _start < self.end:
                period_end = min(_start + delta, self.end)
                periods.append((_start, period_end))
                _start = period_end

            # Making immutable
            return tuple(periods)

        def index_periods(
            delta: Union[float, timedelta] = 365, start: Optional[datetime] = None
        ) -> tuple[tuple[int, int], ...]:
            """
            Identical to periods, but intervals are tuple[int, int] index values

            Returns exactly the same thing as periods (defined above, see
            documentation), but the datetimes defining the beginning and end
            of each interval are instead represented as index values
            corresponding to the index of this performance's dataset

            This function is only available after finalization.

            Parameters:
                delta:
                    An object representing desired interval length

                start:
                    The beginning of the greater period to split into
                    intervals of length 'delta'

            Returns:
                The greater interval split into intervals of length delta.
                Intervals are tuple[start_index: int, end_index: int].
            """
            # func that creates interval from a period
            mk_interval = lambda period: (get_index(period[0]), get_index(period[1]))

            # Returning tuple for immutability
            return tuple([mk_interval(period) for period in periods(delta, start)])

        # Providing the additional functionality after finalization
        self.range = _range  # type: ignore[assignment]
        self.quote = quote  # type: ignore[assignment]
        self.vol = vol  # type: ignore[assignment]
        self.ma = ma  # type: ignore[assignment]
        self.beta = beta  # type: ignore[assignment]
        self.roi = roi  # type: ignore[assignment]
        self.cagr = cagr  # type: ignore[assignment]
        self.alpha = alpha  # type: ignore[assignment]
        self.periods = periods  # type: ignore[assignment]
        self.index_periods = index_periods  # type: ignore[assignment]

    def index_periods(
        self, delta: Union[float, timedelta] = 365, start: Optional[datetime] = None
    ) -> tuple[tuple[int, int], ...]:
        """
        Identical to periods, but intervals are tuple[int, int] index values

        Returns exactly the same thing as periods (defined above, see
        documentation), but the datetimes defining the beginning and end
        of each interval are instead represented as index values
        corresponding to the index of this performance's dataset

        This function is only available after finalization.

        Parameters:
            delta:
                An object representing desired interval length

            start:
                The beginning of the greater period to split into
                intervals of length 'delta'

        Returns:
            The greater interval split into intervals of length delta.
            Intervals are tuple[start_index: int, end_index: int].
        """
        raise _attr_error(Performance, "index_periods")

    def ma(
        self, date: Optional[datetime] = None, days: Union[Max, float] = 365
    ) -> float:
        """
        Returns the moving average over the previous period

        The moving average of this performance's total value across the
        given date interval.

        This function is only available after finalization.

        Parameters:
            date:
                The date from which to calculate the ma

            days:
                The duration of the date interval (number of days) from
                which to calculate the ma

        Returns:
            This performance's average total value across the defined date
            interval (the 'moving average')
        """
        raise _attr_error(Performance, "ma")

    def periods(
        self,
        delta: Union[float, timedelta] = 365,
        start: Optional[datetime] = None,
    ) -> tuple[tuple[datetime, datetime], ...]:
        """
        Beginning from start, returns a tuple of datetime intervals which correspond to the
        index from this backtest's data broken up into periods with interval length = delta

        From the inputs, a duration of time 'delta' is determined. The date
        interval defined by self.end - start is then split into periods of
        length delta, where intervals are represented as
        tuple[start:datetime, end:datetime]. A tuple of interval objects
        covering the entire period is returned.

        This function is only available after finalization.

        Parameters:
            delta:
                An object representing desired interval length

            start:
                The beginning of the greater period to split into intervals
                of length 'delta'

        Returns:
            The greater interval split into intervals of length delta.
            Intervals are tuple[start: datetime, end: datetime]
        """
        raise _attr_error(Performance, "periods")

    def quote(self, date: datetime) -> float:
        """
        Returns the backtest's total value on a given date

        Returns this performance's total value as of the given datetime.

        This function is only available after finalization.

        Parameters:
            date:
                The date to valuate on

        Returns:
            value:
                The total value of this performance's tracked portfolio
                as of the given datetime
        """
        raise _attr_error(Performance, "quote")

    def range(self, start: datetime, end: datetime) -> pd.DataFrame:
        """
        Safely returns an interval of the data corresponding to that range

        Given a starting and ending datetime, returns all of the data in
        this Performance's data which exists inside the corresponding
        interval.

        This function is only available after finalization.

        Parameters:
            start:
                The beginning of the interval

            end:
                The end of the interval

        Returns:
            The interval of this performance's data corresponding to the
            range defined by the inputs
        """
        raise _attr_error(Performance, "range")

    def roi(
        self, date: Optional[datetime] = None, days: Union[Max, float] = 365
    ) -> float:
        """
        Returns the backtest's return on investment over the previous period

        Given a date and a duration to define a date interval, returns the
        difference in total performance value from the beginning of the
        period to the end of the period as a proportion of the total value
        at the beginning.

        For safety purposes, a division by zero (total value at the
        beginning of the period is 0), will result in a return value of 0,
        rather than raising an error.

        This function is only available after finalization.

        Parameters:
            date:
                The end of the date interval

            days:
                The duration of the date interval

        Returns:
            The performance's return on investment (as a percentage) for
            the given date interval
        """
        raise _attr_error(Performance, "roi")

    def vol(
        self, date: Optional[datetime] = None, days: Union[Max, float] = 365
    ) -> float:
        """
        Returns the backtest's volatility over the specified period

        Returns a calculation of this perfomance's historical volatility
        across the interval defined by the inputs. Historical volatility
        for assets have a multiplier defined by the number of business
        days in a year

        This function is only available after finalization.

        Parameters:
            date:
                The historical volatility as of this date

            days:
                The duration of the interval to use for the volatility
                calculation

        Returns:
            The historical volatility for the given period
        """
        raise _attr_error(Performance, "vol")


class Backtest:
    """
    The statistical profile of an Algorithm's Backtest performance

    A Backtest stores data about a single 'run' or backtest of an algorithm.
    An Algorithm generates and returns a backtest object when it is called.

    During algorithm runtime, Backtests have access to portfolio data inside
    the algorithm's environment, as well as access to their own level of
    completion.

    When an algorithm runtime ends, it's respective Backtest is 'finalized', and
    advanced stastical measures are performed at once so that they may be
    parallelized.

    Backtests are not intended to be instantiated directly by end users, but
    end users will often interact with them as they are the object returned by
    performing an algorithm backtest.

    When accessing an attribute or method that exists on a performance,
    a Backtest will produce a dictionary of that attribute accessed
    on each Performance, where keys are the name of the portfolio tracked
    by each performance respectively. If the attribute being accessed in a
    call to a Performance method, the same arguments will be passed to a call
    for each Performance, and outputs will be used to construct a similarly
    structured dictionary.

    Examples:
        .. code-block:: python

            backtest = MyAlgorithm(end=3650)
            performance = backtest.performances["MAIN"]

        .. code-block:: pycon

            >>> backtest.alpha()
            {'MAIN': 1.34892}

            >>> performance.alpha()
            1.34892

    Parameters:
        env:
            The environment to track

        start:
            The Backtest's starting datetime

        end:
            The Backtest's ending datetime

        algo_name:
            The name of the algorithm being backtested

    Attributes:
        algo_name (str):
            The name of the Algorithm to which this Backtest belongs

        complete (bool):
            A boolean value indicating whether or not this Backtest's runtime is
            complete. If complete=True, this Backtest has been finalized.

        duration (timedelta):
            The duration of time covered by the backtest

        end (datetime):
            The end of the Backtest's runtime

        main (Environment):
            The environment object within the algorithm performing the backtest.
            This environment is what allows backtests to track performance data.

        performances (dict[str, Performance]):
            A dictionary of Performance objects, where keys are the name of the
            name of the portfolio being tracked by the Performance.

        start (datetime):
            The beginning of the Backtest's runtime

        timesteps (int):
            The number of timesteps (valuation periods) that this Backtest has
            iterated through

        timesteps_today (int):
            The number of timesteps that have been taken as of the day of the
            most recent valuation period.
    """

    def __init__(
        self, env: Environment, start: datetime, end: datetime, algo_name: str
    ) -> None:
        self._previous = self.date
        self.start = start
        self.end = end
        self.duration = self.end - self.start
        self.remaining = self.duration
        self.timesteps = 0
        self.timesteps_today = 0
        self.complete = False
        self.performances: dict[str, Performance] = {
            name: Performance(pf) for name, pf in env.portfolios.items()
        }
        self.main = self.performances[env.main.name]
        self.algo_name = algo_name

        # If the backtest's environment has been finalized, we can get a more
        # precise measurement of the completeness of the backtest by using
        # timesteps over days
        if env.times:
            self._total = len(env.times) * self.duration.days
            step_getter = (lambda self: (self.timesteps, self._total)).__get__(
                self, Backtest
            )
            completion_getter = (
                lambda self: round(((self.timesteps / self._total) * 100), 2)
            ).__get__(self, Backtest)
            self._step = step_getter
            self._completion = completion_getter

        # Calling __get__ with self and backtest turns the func into a bound
        # method of this backtest instance

        else:
            self._total = self.duration.days
            step_getter = (lambda self: (self.day, self._total)).__get__(self, Backtest)
            completion_getter = (
                lambda self: round(((self.day / self._total) * 100), 2)
            ).__get__(self, Backtest)
            self._step = step_getter
            self._completion = completion_getter
            # CAN PROPERTY TAKE IN AN OBJECT TO BIND TO?

    def __bool__(self) -> Literal[True]:
        """Backtests are always evaluated as True"""
        return True

    def __getattr__(self, attr: str) -> Any:
        """Allows Backtests to access Performance attributes/methods"""

        # Only reroute if performances is not empty
        if self.performances:

            # Getting one of the performances to see if it has the attr
            perf = list(self.performances.values())[0]
            if getattr(perf, attr, False):

                # If its a method, we want to call it with proper arguments
                # for each Performance
                if utils.is_func(getattr(perf, attr)):
                    return self._aggregate(attr)

                # Otherwise, just access it on each performance
                return {
                    pname: getattr(p, attr) for pname, p in self.performances.items()
                }

        raise AttributeError(f"AlphaGradient Backtest object has no attribute '{attr}'")

    def __str__(self) -> str:
        return f"<{self.algo_name} Algorithm Backtest: {self.duration}>"

    def __repr__(self) -> str:
        return self.__str__()

    @property
    def completion(self) -> float:
        """This Backtest's current level of completion"""
        if self.complete:
            return 100
        return self._completion()

    @property
    def day(self) -> int:
        """The number of days of difference between the runtime start and the
        current date"""
        return (self.date - self.start).days

    def _aggregate(self, func_name: str) -> Callable:
        """
        Private method for accessing Performance method on a Backtest

        When accessing an attribute or method that exists on a performance,
        returns a function that produces a dictionary with the attribute
        calculated for each performance contained within this backtest

        Examples:
        .. code:: python
             backtest = MyAlgorithm(end=3650)
             performance = backtest.performances["MAIN"]

             main_alpha = performance.alpha()
             aggregate_alpha = backtest.alpha()

        .. code::
             >>>main_alpha
             1.34892

             >>>aggregate_alpha
             {'MAIN': 1.34892}

        Parameters:
            func_name:
                The Performance method being accessed

        Returns:
            A function which accepts any arguments, so that it may be used for
            any Performance method call. This returned function performs the
            method call with the same arguments for every Performance, and
            returns a dictionary with which associates Performance names with
            the result of the method call. See examples for use cases
        """

        def aggregated(
            *args: Any, aggregate: bool = False, **kwargs: Any
        ) -> dict[str, Any]:
            attr_dict = {
                k: getattr(v, func_name)(*args, **kwargs)
                for k, v in self.performances.items()
            }
            if aggregate:
                attr_dict = {k: v for k, v in attr_dict.items() if not math.isnan(v)}
                adj_weights = {k: v for k, v in self.weights.items() if k in attr_dict}
                new_total = sum(adj_weights.values())
                adj_weights = {k: v / new_total for k, v in adj_weights.items()}
                values = np.array(list(attr_dict.values()))
                weights = np.array(list(adj_weights.values()))
                return (values * weights).sum()
            return attr_dict

        return aggregated

    def completion_report(self) -> str:
        """Returns a small report of the state of completion of the current backtest

        This report is formatted as such:
            * DAY | The number of days since the start of the runtime
            * STEP | The number of valuation periods as of today
            * COMPLETION | The current level of completion
            * REMAINING | The amount of time remaining until completion
        """
        counter, total = self._step()
        return (
            f"DAY {self.day} | "
            f"STEP {self.timesteps_today} | "
            f"COMPLETION: {self.completion}% ({counter} / {total}) | "
            f"REMAINING: {self.remaining}"
        )

    def finalize(self) -> None:
        """Updates the status of the Backtest to complete and performs parallelized
        performance metric calculation

        This is automatically called when a Backtest's runtime is completed,
        which subsequently calls finalize() on each of a Backtest's
        Performances.

        Updates the status of the Backtest to complete, and performs aggregate
        statistical analytics.
        """
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
        self.weights = {
            k: v / self.history.iloc[0, :]["TOTAL"] for k, v in self.initial.items()
        }
        self.history["CHANGE"] = self.history["TOTAL"].pct_change()

    def profile(self) -> dict[str, Any]:
        """Creates an HTML report of this backtest's performance

        TODO:
            Currently just shows the total profit and roi. Should accept a path
            for an HTML or .prof output which can be viewed in a browser,
            complete with graphics and data visualization.
        """
        profit = self.history["TOTAL"][-1] - self.history["TOTAL"][0]
        roi = round(self.roi(days="max", aggregate=True) * 100, 2)
        return {
            "PROFIT": f"${round(profit, 2)}",
            "ROI": f"{roi}% | {round(roi / 100, 2) + 1}x",
        }

    def update(self) -> None:
        """Updates the current backtest at each timestep"""

        # Only perform when the backtest is ongoing
        if not self.complete:

            # Incrementing steps
            self.timesteps += 1

            # Determining if the timesteps_today counter should be reset
            previous = utils.set_time(self._previous + timedelta(days=1), "00:00:00")
            if self.date >= previous:
                self.timesteps_today = 1
            else:
                self.timesteps_today += 1

            # Updating stored date
            self._previous = self.date
            self.remaining = self.end - self.date


class Stats:
    """
    Handles automatic Backtest/Performance generation for algorithms

    Parameters:
        algo: The algorithm that this stats instance belongs to

    Attributes:
        current (Optional[Backtest]):
            The current backtest being performed by the algorithm
    """

    def __init__(self, algo: Algorithm) -> None:
        self._runs: list[Backtest] = []
        self.current: Optional[Backtest] = None
        self._algoname: str = algo.__class__.__name__

    def __bool__(self) -> Literal[True]:
        return True

    @property
    def completion(self) -> float:
        """The level of completion of the current backtest"""
        if self.current is not None:
            return self.current.completion
        return 0

    @property
    def runs(self) -> list[Backtest]:
        """A list of all Backtests completed or currently underway by the
        algorithm attached to this Stats instance"""
        if self.current is not None and self.current.complete:
            return self._runs + [self.current]
        return self._runs

    def _finish(self) -> Backtest:
        """At algorithm runtime completion, finalizes the current backtest
        and resets the currently tracked backtest

        Returns:
            The previously current (now complete) Backtest object
        """
        if self.current is None:
            raise ValueError("No current backtest to finalize")
        self.current.finalize()
        self._runs.append(self.current)
        run = self.current
        self.current = None
        return run

    def _reset(self, env: Environment, start: datetime, end: datetime) -> None:
        """Creates a new backtest (when an algorithm is called) and sets it as
        the current backtest

        All of the environments given to _reset are simply passed to a new
        Backtest object, which is then set as the current backtest

        Parameters:
            env:
                The environment of the backtest

            start:
                The start of the backtest

            end:
                The end of the backtest
        """
        self.current = Backtest(env, start, end, self._algoname)

    def _update(self) -> None:
        """Updates the current backtest at each time step. This function is
        called by both global and local step() methods"""
        # Only perform is a backtest is currently progressing
        if self.current:
            self.current.update()

    def change_report(
        self, date: datetime = None, *, steps: int = None, delta: timedelta = None
    ) -> str:
        """Returns a report of changes to the main portfolio resulting from the
        most recent timestep

        Specifying any of date, steps, or delta will change the output
        accordingly to reflect changes from the different valuation period
        specified

        Of steps and delta, only ONE may be specified

        Parameters:
            date:
                The ending date of the report

            steps:
                How many steps to backtrack

            delta:
                How much time to backtrack

        Returns:
            The formatted report of the changes for the given period
        """
        if self.current is None:
            raise ValueError(f"No current backtest to report changes of.")
        comp_report = self.current.completion_report()
        change_report = self.current.main.change_report(
            date=date, steps=steps, delta=delta, header=True, border=True
        )
        return f"{comp_report}\n{change_report}"


class Algorithm(ABC):
    """
    A base class that provides the functionality for creating backtestable algorithms

    The AlphaGradient algorithm framework which allows for creation of automatically
    backtestable algorithms. Algorithms in AlphaGradient must implement two
    functions: setup and cycle.

    setup is a function responsible for creating an environment which the
    algorithm will use when backtesting. This environment is recreated from
    scratch every time a new backtest is created. setup must be capable of
    accepting start= and end= keyword arguments, so either add them explicitly
    or accept ** kwargs. It should always return an Environment object and
    nothing else

    Docstrings for setup functions should aim to explain what Assets the
    environment uses and why, as well as the goals of diferrent portfolios
    if there are more than one.

    cycle determines the behavior the algorithm performs at each valuation
    period. When the algorithm takes a step forward in time, It will valuate
    all of this asset's in its environment at the (new) current date, and then
    cycle will be called, which will allow the user to make changes in the
    algorithms environment (such as buying stocks in the environment's
    portfolios). Cycle must also be capable of accepting start= and end= keyword
    arguments, so add them explicitly or accept ** kwargs. It should return
    None; the main goal of cycle is to modify the algorithms environment.

    Docstrings for cycle functions should aim to explain what the goals of the
    algorithm are at every timestep, and what actions it takes to accomplish
    those goals.

    Any changes made by cycle are automatically recorded in the Backtest,
    handled by this algorithm's stats module.

    By default, Algorithms iterate through time using a 'default_run', which
    controls how time progresses and how cycle is called. If this time
    control is insufficient for an algorithm, the user can define their own
    'run' function will be used instead of the default run. Note, however, that
    a cycle function must still be implemented even if the new run function does
    not use it.

    If defining a custom run function (simply call it 'run', all lowercase), it
    also must be capable of accepting start= and end= keyword arguments and
    return None. It should be responsible for controlling how time steps are
    taken as the backtest progresses, as well as logging these events using
    either custom logging systems or those already implemented (self.progress).

    Docstrings for custom run functions should identify the reasons that
    specific time control is necessary for the function of the algorithm, as
    as well as what specifically that time control mechanism is. It should also
    describe logging operations/settings

    Note that for all of these user-defined functions, start= and end= keyword
    arguments will always be passed as a native python datetime object.
    Algorithm.__call__ will automatically handle conversion of non-python
    datetime objects to their native python equivalents, and those will be
    passed instead.

    Parameters:
        start:
            The default starting period of any backtest if none is provided

        end:
            The default ending period of any backtest if none is provided

        resolution:
            The default resolution (used in timesteps) if none is provided

        verbose:
            When True, will output a report of the changes from the previous
            valuation period at every new period.

        progress:
            Displays a progress bar of backtest completion when True. Displays
            nothing if verbose is True, regardless of value.

    Attributes:
        env (Environment):
            The algorithm's runtime environment. Contains all alphagradient
            assets that the algorithm has access to. Should be manipulated in
            self.cycle to perform the functions of the algorithm

        print (Union[built-in-function, NullClass]):
            Is print when verbose=True, otherwise does nothing

        progress (Union[tqdm, NullClass]):
            When progress=True, self.progress is a modified version of tqdm with
            partialized inputs relative to this algorithm. When progress=False,
            does nothing.

        resolution (timedelta):
            The default time resolution of this algorithm when backtesting.
            Calls to self.env.step will use this default resolution if a valid
            timedelta is not provided when called.

        stats (Stats):
            This algorithm's stats instance, which handles automatic creation
            of backtests and performance metrics during algorithm runtime

        verbose (bool):
            Whether or not to log changes to the console at each timestep

    .. automethod:: __call__
    """

    def __init__(
        self,
        resolution: Optional[timedelta] = None,
        verbose: bool = False,
        progress: bool = True,
    ) -> None:
        self.resolution = (
            self._global_res  # type: ignore[attr-defined]
            if resolution is None
            else self.validate_resolution(resolution)
        )
        self._environment = self.setup(start=self.start, end=self.end)
        self.verbose = verbose
        self.print: Any = print if verbose else utils.NullClass()
        self.stats = Stats(self)
        self.type.instances[self._generate_name()] = self  # type: ignore[attr-defined]

        # Reroutes some traditional tqdm functionality to work automatically with Backtest objects
        # without requiring user input. Essentially a partial class version of tqdm
        class tqdm_partial(tqdm):
            def __init__(this, *args, total=None, position=None, leave=None, **kwargs):
                total = self.stats.current._total if total is None else total
                position = 0 if position is None else position
                leave = True if leave is None else leave
                this.ncols = 150
                super().__init__(
                    *args,
                    total=total,
                    position=position,
                    leave=leave,
                    ncols=this.ncols,
                    **kwargs,
                )
                this.set_description(str(self.stats.current))

            def update(this, n=None):
                if n is None:
                    n = self.stats.current._step()[0] - this.n
                super().update(n=n)

            def close(this, *args, **kwargs):
                # this.update(this.total - this.n)
                super().close(*args, **kwargs)

        self.progress = (
            tqdm_partial if (progress and not verbose) else utils.NullClass()
        )

    def __call__(
        self,
        *args: Any,
        start: Optional[DatetimeLike] = None,
        end: Optional[DatetimeLike] = None,
        **kwargs: Any,
    ) -> Backtest:
        """
        Performs an algorithm backtest

        When an algorithm is called, creates a new Backtest object that tracks
        the newly created environment. Any start and end inputs are converted to
        appropriate python datetime objects. If start or end are None,
        appropriate datetime objects are determined based on the algorithm's
        defaults.

        Parameters:
            start (Optional[DatetimeLike]):
                The beginning datetime of the backtest

            end (Optional[DatetimeLike]):
                The ending datetime of the backtest

        Returns:
            A Backtest object tracking this algorithm's current environment
        """

        # Initializing start and end
        start = (
            self.optimal_start()
            if start is None
            else max(self.start, utils.to_datetime(start))
        )
        end = (
            self.end
            if end is None
            else min(self.end, self.validate_end(end, start=start))
        )

        print(start, end)

        # Reset stats for new backtest
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

    @property
    def _run(self) -> Callable:
        """Returns user defined run function if defined, else default_run"""
        if getattr(self, "run", False):
            return self.run  # type: ignore[attr-defined]
        return self.default_run

    @property
    def active(self) -> bool:
        """Whether or not the algorithm's environment can currently undergo changes"""
        current_complete = (
            self.stats.current.complete if self.stats.current is not None else False
        )
        return self.env.open and not current_complete

    @property
    def start(self) -> datetime:
        try:
            return self.env.start
        except AttributeError:
            return self._global_start

    @property
    def end(self) -> datetime:
        try:
            return self.env.end
        except AttributeError:
            return self._global_end

    @property
    def date(self) -> datetime:
        """The current date of the algorithm"""
        return self.env.date

    @property
    def env(self) -> Environment:
        """The algorithms environment"""
        return self._environment

    @env.setter
    def env(self, environment):
        if not isinstance(environment, Environment):
            raise TypeError(
                f"environment must be a Environment object, received {environment=}"
            )
        self._environment = environment

    @no_type_check
    @abstractmethod
    def setup(
        self, *args: Any, start: datetime, end: datetime, **kwargs: Any
    ) -> Environment:
        """
        Sets up the local environment for the algorithm to operate on

        A required implementation of all new algorithms, setup is an abstract
        method that returns a 'fresh' environment object that will be operated
        on over the course of an algorithm backtest. This method is called to
        create a new environment appropriate for the algorithm whenever a new
        backtest is created.

        Parameters:
            start (datetime):
                This method will ALWAYS receive a 'start' (datetime) keyword
                argument equal to the start of the backtest, so any user
                defined implementation must be capable of accepting it.

            end (datetime):
                This method will ALWAYS receive an 'end' (datetime) keyword
                argument equal to the end of the backtest, so any user defined
                implementation must be capable of accepting it.

        Returns:
            Any implementation of this function must ALWAYS return an
            AlphaGradient Environment object.
        """
        return Environment()

    @no_type_check
    @abstractmethod
    def cycle(self, *args: Any, start: datetime, end: datetime, **kwargs: Any) -> None:
        """
        Defines the algorithm's behavior at each new timestep

        A required implementation of all new algorithms, cycle is an abstract
        method that determines the algorithm's behavior at each timestep during
        backtesting. This method is called whenever this algorithm's environment
        steps forward in time. It should return None; its goal is to modify
        the self.env object (created by setup)

        Parameters:
            start (datetime):
                This method will ALWAYS receive a 'start' (datetime) keyword
                argument equal to the start of the backtest, so any user
                defined implementation must be capable of accepting it.

            end (datetime):
                This method will ALWAYS receive an 'end' (datetime) keyword
                argument equal to the end of the backtest, so any user defined
                implementation must be capable of accepting it.
        """
        # return Portfolio(0, date=self.start)
        return None

    @classmethod
    def _generate_name(cls) -> int:
        """
        Generated a name for this algorithm based on the number of algorithms
        that have already been instantiated.
        """
        return len(cls.type.instances)  # type: ignore[attr-defined]

    def default_run(
        self, *args: Any, start: datetime, end: datetime, **kwargs: Any
    ) -> None:
        """
        The default run cycle when none is implemented by an algorithm

        By default, this algorithm controls time iteration by simply iterating
        to the next datetime (determined by self.env.next()) until it has
        reached the end.

        The default implementation uses a tqdm progress bar to display
        algorithm backtest progress.

        Parameters:
            start:
                The beginning of the backtest

            end:
                The end of the backtest
        """

        # Manual control of tqdm progress bar
        with self.progress() as progress:  # type: ignore[operator]
            while self.date < end:
                if self.env.open:
                    self.cycle(*args, start=start, end=end, **kwargs)
                self.env.next()
                progress.update()

    def intitialize_inputs(self, start: datetime = None, end: datetime = None) -> None:
        """
        Ensures that start and end parameter inputs are viable

        TODO: This needs a complete overhaul. it doesnt even return anything??

        Parameters:
            start:
                The backtest start datetime

            end:
                The backtest end datetime
        """
        start = self.start if start is None else utils.to_datetime(start)

    def optimal_start(self, end: Optional[DatetimeLike] = None) -> datetime:
        """
        Based on the current algorithm environment, returns an optimal datetime
        for starting a backtest

        Returns a backtest starting datetime that:
            * Is guaranteed to be within the date range of all intantiated assets
            * | Is guaranteed to have ample time for calculations of historical
              | volatility, beta, percent change etc. BEFORE the start date
            * Automatically adjusts to accomodate shorter ending periods
        """
        max_start_data = [dataset.first for dataset in self.env.data()]
        if not max_start_data:
            return self.start

        max_start = max(max_start_data)
        min_end = min([dataset.last for dataset in self.env.data()])
        end = min_end if end is None else utils.to_datetime(end)

        # If the maximum start date is before the minimum end date, there is
        # no valid 'optimal start', because there is no date range that allows
        # backtesting of all available data.
        if max_start >= end:
            return self.start

        # Determining the optimal start period. To avoid errors, we will not sync to the beginning
        optimal_delta = (end - max_start) / 2
        optimal_date = max_start + optimal_delta

        # Setting the optimal date's time to market open unless specified otherwise
        t = "09:30:00"
        utils.set_time(optimal_date, t)

        # Bounding the date to acceptable minimums and maximums
        lower_bound = utils.set_time(max_start + timedelta(days=1), t)
        upper_bound = utils.set_time(max_start + timedelta(days=365), t)
        optimal_start = utils.bounded(
            optimal_date, lower=lower_bound, upper=upper_bound
        )

        return optimal_start

    def validate_end(
        self,
        end: Union[str, pd.Timestamp, datetime, float, timedelta],
        start: datetime = None,
    ) -> datetime:
        """Given a date-like object, returns an algorithm-viable datetime to
        act as the ending date of an algorithm backtest, based on some start
        input

        TODO: COMPLETE OVERHAUL

        Parameters:
            end:
                The end to validate

            start:
                Validating the end relative to a different start input

        Returns:
            The validates/converted end datetime.
        """
        start = self.optimal_start() if start is None else start

        # See if the datetime can be validated normally
        try:
            return utils.to_datetime(end)

        # For convenience, end inputs have many more acceptable / coherent
        # input types (in reference to some starting point)
        except TypeError as e:
            if isinstance(end, (float, int)):
                end = start + timedelta(days=end)

            elif isinstance(end, timedelta):
                end = start + end

            # No conversion possible
            if not isinstance(end, datetime):
                raise e

            return end

    def validate_resolution(self, delta: Union[timedelta, float]) -> timedelta:
        """Given a timedelta-like object, converts it to a algorithm-viable
        timedelta object

        TODO: COMPLETE OVERHAUL

        Parameters:
            delta:
                The time delta representing the resolution of the backtest

        Returns:
            The object converted to a valid python timedelta
        """
        if isinstance(delta, (float, int)):
            return timedelta(days=delta)

        elif isinstance(delta, timedelta):
            return delta

        else:
            raise TypeError(
                f"Invalid resolution type {delta=}. "
                "Resolution must be a number or timedelta"
            )


# Attaching algorithm class to the types enum
setattr(Algorithm, "type", types.algorithm)
setattr(types.algorithm, "c", Algorithm)
