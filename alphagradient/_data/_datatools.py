# -*- coding: utf-8 -*-
"""AG module containing tools for creation/manipulation of asset data

This module contains tools for accessing, creating, and modifying data
and datasets for use in AlphaGradient objects.

TODO:
    * AUTOMATICALLY DETERMINE RESOLUTION OF DATASETS
    * Implement dtype coercion on column validation
    * Type Hints
"""

# Standard imports
from __future__ import annotations

from datetime import datetime, timedelta, time
from numbers import Number
from pathlib import Path
import os

# Third party imports
import pandas as pd
import numpy as np

# Local imports
from .. import utils

# Typing
from typing import (
    TYPE_CHECKING,
    Any,
    Optional,
    Type,
    Union,
)
from typing_extensions import TypeAlias

from ..utils import DatetimeLike

if TYPE_CHECKING:
    from .._finance import Asset

_AssetData: TypeAlias = "AssetData"
"""Type Alias for AssetData"""

ValidData = Union[Path, str, _AssetData, pd.DataFrame, np.ndarray, float]
"""Valid inputs for AssetData conversion"""


def currency_info(base: Optional[str] = None, save: bool = False) -> pd.DataFrame:
    """Returns a DataFrame with all currency values updated relative
    to the base

    Parameters:
        base:
            Currency code on which to base all currency valuations

        save:
            Whether or not the dataframe should be saved as a pickle to the
            local directory

    Returns:
        The updated currency info
    """
    base = "USD" if base is None else base

    def mapper(code):
        try:
            ticker = yf.Ticker(f"{code}{base}=X")
            history = ticker.history(period="1d", interval="1d")
            return history["Close"].item()
        except:
            return -1

    raw_path = Path(__file__).parent.joinpath("currency_info_raw.p")
    pickle_path = Path(__file__).parents[1].joinpath("finance/currency_info.p")
    info = pd.read_pickle(raw_path)
    info["VALUE"] = info["CODE"].map(mapper)
    info = info[info["VALUE"] > 0]

    if save:
        with open(pickle_path, "wb") as p:
            info.to_pickle(p)

    return info


def get_data(asset: Asset) -> Optional[AssetData]:
    """
    Accesses locally stored data relevant to this asset

    This function safely accesses locally stored data for this asset, either in
    the persistent storage or a locally identified path (using CSVs, or other
    forms of storage for tabular data). It will return None if nothing is found

    Parameters:
        asset: The asset to retrieve data for

    Returns:
        Stored dataset relevant to the asset. None if nothing is found
    """
    key = asset.key
    data = None

    # Getting the global persistent path
    base_path = utils._global_persistent_path.fget(None)

    # Returning if the global persistent path has not been set
    if base_path is None:
        return None

    # Trying to get the data from persistent pickles
    path = base_path.joinpath(f"{key}.p")
    try:
        with open(path, "rb") as data:
            return AssetData(asset.__class__, pd.read_pickle(data), preinitialized=True)
    except EOFError:
        os.remove(path)
    except FileNotFoundError:
        pass

    # Getting the data from raw files
    path = base_path.joinpath(f"{key}.csv")
    try:
        return AssetData(asset.__class__, path)
    except Exception:
        return None


class AssetData:
    """Datetime-indexed datesets that store financial data for assets.

    All AlphaGradient assets seeking to use tabular data must use
    AssetData datasets.

    AssetData accepts any of the following inputs:
        * numbers (for assets with constant prices, or unit prices)
        * os.path-like objects (eg. pathlib.Path)
        * file-object-like objects
        * array-like objects (lists, ndarray, etc.)
        * strings that represent paths
        * pandas DataFrames

    AssetDatasets take tabular data and clean/validate it to make it
    usable for alphagradient assets. They check for the presence of required
    data, remove unnecessary data, ensure that dataframes are datetime-indexed,
    coerce data dtypes (TODO), and ensure that formatting is consistent between
    all assets.

    Parameters:
        asset_type:
            The asset type that defines the constraints for the formatting of
            this dataset. Different asset types allow different kinds of data.

        data (ValidData):
            A valid dataset to format (see docstring for acceptable types)

        columns:
            A list of strings to be used as columns. Optional

        preinitialized:
            Whether or not the data being passed in is already an initialized
            AssetData dataset. This requires that the data input is a
            pd.DataFrame object that has been loaded from a persistent
            AlphaGradient AssetData file (located in persistent storage)

    Attributes:
        open_value (str):
            The column to associate with market open

        close_value (str):
            The column to associate with market close

        single_valued (bool):
            True if this AssetData defines only one of close_value and open_value,
            or if they are the same. False if both are defined and different.
    """

    _set_time_vectorized = np.vectorize(utils.set_time, excluded=["t"])

    def __init__(
        self,
        asset_type: Type[Asset],
        data: ValidData,
        columns: Optional[list[str]] = None,
        preinitialized: bool = False,
    ):

        # When the data is already initialized
        if preinitialized:
            assert type(data) is pd.DataFrame
            self._data = data
            self._init_firstlast()
            self.resolution = self._global_res
            if len(self._data) > 1:
                self.resolution = (
                    self._data["_time_resolution_"]
                    .value_counts()
                    .index[0]
                    .to_pytimedelta()
                )
            return

        # Unpacking necessary values from the asset type
        _, _, _, required, optional, close_value, open_value = asset_type.get_settings(
            unpack=True
        )

        # Formatting required columns
        required = required if required else []
        required = [self.column_format(column) for column in required]

        def numinit(n):
            close_value = close_value if close_value else "CLOSE"
            additions = ["DATE"]
            if close_value not in required:
                additions.append(close_value)
            required = additions + required
            frame = [[datetime.today().date()] + ([data] * (len(required) - 1))]
            return pd.DataFrame(frame, columns=required)

        # null case
        if data is None:
            return None

        # Handle numeric inputs that default to static dataframes
        elif isinstance(data, Number):
            data = numinit(data)

        # Handle list inputs, np.ndarray inputs
        elif isinstance(data, (list, np.ndarray)):
            if not columns:
                raise ValueError(
                    f"{type(data).__name__} input "
                    "requires explicit column names "
                    "during initialization"
                )
            data = pd.DataFrame(data, columns=columns)

        elif isinstance(data, AssetData):
            data = data.data

        # Handle inputs that can be processed by pd.read_table
        elif not isinstance(data, pd.DataFrame):
            try:
                data = pd.read_table(data, sep=",")
            except (TypeError, ValueError) as e:
                pass

        # Final check that we have valid data prior to formatting
        if isinstance(data, pd.DataFrame):
            if (
                isinstance(data.index, pd.core.indexes.datetimes.DatetimeIndex)
                or isinstance(data.index.name, str)
                and data.index.name.lower() == "date"
            ):
                data.index.name = "DATE"
                data["DATE"] = data.index

                # This converts the values to a numpy array, automatically removing tzinfo
                data["DATE"] = data["DATE"].values

                # OLD METHOD OF REMOVING TZINFO
                # data["DATE"] = data["DATE"].map(lambda date: date.replace(tzinfo=None))

        else:
            raise ValueError(f"Unable to create valid asset dataset from {data}")

        # Formatting columns
        data.columns = [self.column_format(column) for column in data.columns]

        # Grabbing "OPEN" and "CLOSE" by defauly if not specified
        open_value = (
            "OPEN" if ("OPEN" in data.columns and not open_value) else open_value
        )
        close_value = (
            "CLOSE" if ("CLOSE" in data.columns and not close_value) else close_value
        )

        # Broadcasting open to close or close to open in case only one is provided
        if close_value and not open_value:
            close_value = self.column_format(close_value)
            open_value = close_value
            self.single_valued = True

        elif open_value and not close_value:
            open_value = self.column_format(open_value)
            close_value = open_value
            self.single_valued = True

        # By this point both should be present if even one was provided
        elif not all([close_value, open_value]):
            raise ValueError(
                "Must specify at least one opening or "
                "closing value name present in the data"
            )

        # Both an open and close have been explicitly provided
        else:
            open_value = self.column_format(open_value)
            close_value = self.column_format(close_value)
            self.single_valued = False

        # Attribute initialization
        self.open_value = open_value
        self.close_value = close_value

        # Adding default required columns (open, close, date)
        if close_value:
            required = [close_value] + required

        if open_value:
            required = [open_value] + required

        required = ["DATE"] + required

        # Removing duplicates
        required = list(set(required))

        # Both of the values (open and close) must be in required
        if not all([value in required for value in [open_value, close_value]]):
            raise ValueError(
                "Must specify at least one opening or "
                "closing value name present in the data"
            )

        # Final formatting requirements
        data = self._init_columns(data, required, optional)  # type: ignore[arg-type]

        # Setting date column as DatetimeIndex
        data = data.set_index(pd.DatetimeIndex(data["DATE"]))
        data.drop("DATE", axis=1, inplace=True)

        self._data = data
        self.resolution, success = self._init_resolution()
        if success:
            self._init_periods(asset_type)

        # Single value dataframe
        elif not data.empty:
            if self._data.index[0].time() == time():
                first = self._data.index[0]
                period_open = utils.set_time(first, asset_type.market_open)
                period_close = utils.set_time(first, asset_type.market_close)
                self._data["_period_close_"] = [period_close]
                self._data["_period_open_"] = [period_open]
            else:
                self._data["_period_close_"] = [self._data.index[0]]
                self._data["_period_open_"] = [self._data.index[0]]
            self._data["_time_resolution_"] = [self.resolution]

        # Dataframe is empty, add new columns
        else:
            new_cols = [
                "_time_resolution_",
                "_period_open_",
                "_period_close_",
                "CHANGE",
            ]
            self._data = self._data.reindex(self._data.columns.union(new_cols), axis=1)
            print(self._data)

        if not data.empty:
            self._init_firstlast()
            self._init_stats(asset_type)

        else:
            t = utils.set_time(datetime.today(), "0:0:0")
            self._first, self._last = t, t

    def __getattr__(self, attr: str) -> Any:
        try:
            return self.data.__getattr__(attr)
        except AttributeError:
            try:
                return self.data[attr]
            except KeyError as e:
                raise AttributeError(f"'AssetData' object has no attribute {e}")

    def __getitem__(self, item: Any) -> Any:
        return self.data[item]

    def __str__(self) -> str:
        return self.data.__str__()

    def __bool__(self) -> bool:
        return not self._data.empty

    def __getstate__(self) -> dict[str, Any]:
        return self.__dict__

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__ = state

    def __len__(self) -> int:
        return len(self._data)

    @property
    def first(self) -> datetime:
        """The first date available in the data"""
        return self._first

    @property
    def last(self) -> datetime:
        """The last date available in the data"""
        return self._last

    @property
    def data(self) -> pd.DataFrame:
        """The data associated with this Asset Dataset"""
        return self._data.drop(
            ["_time_resolution_", "_period_open_", "_period_close_"], axis=1
        )

    def _init_columns(
        self,
        data: pd.DataFrame,
        required: Union[list[str], dict[str, str]],
        optional: Union[list[str], dict[str, str]],
    ) -> pd.DataFrame:
        """Ensures that the input data meets the formatting
        requirements to be a valid asset dataset

        To be a valid asset dataset, input tabular data must have every
        column listed in 'required'. This method ensures that all of
        the required columns are present, as well as removes columns
        that don't show up in either the required or optional lists.

        TODO:
            * allow dictionaries to be passed in to enforce specific column dtypes.
            * | Implement more checks to ensure that data is error-free, and
              | lacks missing elements (or implement measures to be safe in the
              | presence of missing data)

        Parameters:
            data:
                The data being validated

            required:
                A list of strings representing column names. All of the columns
                in this list must be present to produce a viable dataset.

            optional:
                A list of strings representing column names. Columns in the
                data that are not required will still be kept in the data if
                they are present in this list. Otherwise, they will not be
                included.

        Returns:
            A verified (and potentially coerced) asset dataset

        Raises:
            ValueError:
                raised when the input dataset does not satisfy the requirements
        """
        # Check for column requirements
        required = required if required else {}
        optional = optional if optional else {}

        # Converting required and optional to dicts
        def to_dict(columns):
            if isinstance(columns, list):
                columns = [self.column_format(column) for column in columns]
                columns = {column: "float" for column in columns}

            elif isinstance(columns, dict):
                columns = {
                    self.column_format(column): dtype
                    for column, dtype in columns.items()
                }

            return columns

        required = to_dict(required)
        optional = to_dict(optional)

        # Checking whether all of the required columns have been satisfied
        satisfied = {column: (column in data.columns) for column in required}
        unsatisfied: Any = [
            column for column, present in satisfied.items() if not present
        ]
        if unsatisfied:
            unsatisfied = str(unsatisfied)[1:-1]
            raise ValueError(f"AssetData missing required columns: {unsatisfied}")

        # Coercing dtypes to those specified in required and optional
        # CURRENTLY NOT IMPLENENTED, REQUIRED PASSED IN AS LIST
        """
        for column_dict in [required, optional]:
            for column, dtype in column_dict:
                self.data[column] = self.data[column].astype(dtype)
        """

        # Dropping columns that are not present in optional or required
        for column in data.columns:
            if column not in list(required) + list(optional):
                data.drop(column, axis=1, inplace=True)

        return data

    def _init_firstlast(self) -> None:
        """Saves this dataset's first and last available dates inplace"""
        self._first = self._data["_period_open_"][0]
        self._last = self._data["_period_close_"][-1]

    def _init_periods(self, cls) -> None:
        """Defines opening and closing periods at each index in the dataset"""

        @np.vectorize
        def close_map(close, time_res, open_date):
            """Vectorized mapping function set datetimes to market close if they should be evaluated at market close"""
            if time_res >= np.timedelta64(1, "D"):
                return utils.set_time(pd.to_datetime(open_date), cls._market_close)
            return close

        index = self._data.index.to_series()
        self._data["_period_open_"] = index
        self._data["_period_close_"] = index + self._data["_time_resolution_"]

        # This time remapping only occurs when timedelta >= 1
        if self.resolution == timedelta(days=1):

            # Setting the opening time at each index to the market open time
            self._data["_period_open_"] = self._set_time_vectorized(
                self._data["_period_open_"], cls._market_open
            )

            # Setting the closing time at each index to the market close time, only if the delta is >= 1
            self._data["_period_close_"] = close_map(
                self._data["_period_close_"],
                self._data["_time_resolution_"],
                self._data["_period_open_"],
            )

    def _init_resolution(self) -> tuple[timedelta, bool]:
        """Automatically determines the time resolution (granularity)
        of the dataset's datetime index."""
        default = self._global_res
        resolution = None

        if self._data is not None and len(self._data) > 1:
            res_data = self._data.index.to_series()
            res_data = res_data.shift(-1) - res_data
            res_data[-1] = res_data[-2]
            resolution = res_data.value_counts().index[0].to_pytimedelta()
            self._data["_time_resolution_"] = res_data

        return (resolution, True) if resolution is not None else (default, False)

    def _init_stats(self, asset_type: Type[Asset]) -> None:
        """Adds a new "CHANGE" column to the dataset which corresponds to the
        percentage change in closing price compared to the previous period

        TODO:
            lmao, I didnt realize that pandas had a builtin pct_change func.
            woops... look to replace later.
        """
        if len(self._data) > 1:
            shifted = self._data[asset_type._close_value].shift(1)
            shifted.name = "A"
            shifted[0] = shifted[1]
            shifted = shifted.to_frame()
            shifted["B"] = self._data[asset_type._close_value]
            shifted = (shifted["B"] / shifted["A"]) - 1
        else:
            shifted = self._data[asset_type._close_value]

        self._data["CHANGE"] = shifted

    @staticmethod
    def column_format(column: str) -> str:
        """
        The standard string format for columns

        Takes a column name string and returns it in uppercase, with
        spaces replaced with underscores

        This is the column format used for all AlphaGradient AssetData columns,
        it publicly accessible in case users need it to properly format inputs
        in a dynamic fashion

        Parameters:
            column (str): The column name to be altered

        Returns:
            The altered column name
        """
        return column.replace(" ", "_").upper()

    def get_index(self, date: DatetimeLike) -> int:
        """
        Given a DatetimeLike object, returns the corresponding integer index

        Given any DatetimeLike object, returns an integer value corresponding
        to the index value which is nearest that datetime. If the date is before
        the beginning of the data, returns index 0. Otherwise, uses asof for
        determination of 'nearest'

        Parameters:
            date (DatetimeLike):
                The datetimelike object to find the index of

        Returns:
            The index value as an integer
        """
        date = max(utils.to_datetime(date), self._first)
        if date <= self.first:
            return 0
        return self._data.index.get_loc(self._data.index.asof(date))

    def get_times(self) -> list[time]:
        """
        Returns a list of unique times present in this data

        For all available points in this AssetData's time-series index, converts
        to a python time object, and then returns the set of all unique times as
        a list.

        Returns:
            A set of unique times for this dataset (as a list)
        """

        def uniquetimes(series):
            return pd.Series(series.map(lambda t: t.time()).unique()).to_list()

        return list(
            set(
                uniquetimes(self._data["_period_open_"])
                + uniquetimes(self._data["_period_close_"])
            )
        )

    def next(self, date: DatetimeLike) -> datetime:
        """
        Given a DatetimeLike object, returns the next available datetime in the
        dataset

        Given an object which can be coerced into a python datetime, returns
        a datetime corresponding to the next available datetime in the index.
        The next datetime MUST be GREATER than the given datetime, so if the
        input datetime falls on an index value directly, it will return the next
        index value.

        Parameters:
            date (DatetimeLike):
                The date for which the next available date will be determined

        Returns:
            A datetime corresponding to the next available date in the index
        """
        date = max(utils.to_datetime(date), self._first)
        index = self._data.index
        value = index.asof(date)
        ind = index.get_loc(value)
        openn = self._data.asof(date)["_period_open_"]
        close = self._data.asof(date)["_period_close_"]

        if date < openn:
            return openn

        elif date < close:
            return close

        elif ind < len(index) - 1:
            ind += 1

        return self._data.iloc[ind]["_period_open_"]

    def prev(self, date: DatetimeLike) -> datetime:
        """
        Given a DatetimeLike object, returns the previous datetime in the
        dataset

        Given an object which can be coerced into a python datetime, returns
        a datetime corresponding to the previous datetime in the index.
        The previous datetime MUST be LESS than the given datetime, so if the
        input datetime falls on an index value directly, it will return the
        previous index value.

        Parameters:
            date (DatetimeLike):
                The date for which the previous date will be determined

        Returns:
            A datetime corresponding to the previous date in the index
        """
        date = utils.to_datetime(date)
        index = self._data.index
        value = index.asof(date)
        ind = index.get_loc(value)
        openn = self._data.asof(date)["_period_open_"]
        close = self._data.asof(date)["_period_close_"]

        if ind > 0:
            prev_close = self._data.iloc[ind - 1]["_period_close_"]
            prev_open = self._data.iloc[ind - 1]["_period_open_"]

            if date >= close:
                return openn
            elif date >= openn:
                return prev_close
            elif date >= prev_close:
                return prev_open

        return openn

    def range(
        self, start: Union[DatetimeLike, float], end: Union[DatetimeLike, float]
    ) -> pd.DataFrame:
        """
        Given a start and an end, indexes the the dataframe after coercing to datetimes

        Expands regular indexing on an interval on a time-series dataframe by
        allowing coercion of not only DatetimeLike objects, but numbers as well.

        When numbers are passed in, uses the other input as a context to identify
        a period of time from that point, where the number respresents the number
        of days difference. For this reason, only one of the two inputs may be
        numbers.

        TODO:
            Is this safe?? It is dramatically more performant, but I'm not entirely sure how/why it works.
            For example, indexing using a scalar value that is not present in the index but which is
            contained in index interval will raise a KeyError. Suddently, this behavior becomes acceptable
            when indexing using a slice??? No clue whats happening here.

        Parameters:
            start (DatetimeLike or float):
                The beginning of the range to return

            end (DatetimeLike | Number):
                The end of the range to return

        Returns:
            A range of the available data corresponding to the period defined
            by start and end.
        """
        start_number = isinstance(start, (int, float))
        end_number = isinstance(end, (int, float))
        if start_number or end_number:
            if start_number and end_number:
                raise ValueError(f"Only one parameter may be defined as number")
            elif start_number:
                assert isinstance(start, (int, float))
                end = utils.to_datetime(end)
                start = end - timedelta(days=start)
            elif end_number:
                assert isinstance(end, (int, float))
                start = utils.to_datetime(start)
                end = start + timedelta(days=end)
        else:
            start = utils.to_datetime(start)
            end = utils.to_datetime(end)
        return self._data.loc[start:end]  # type: ignore[misc]

    def valuate(self, date: DatetimeLike, asset: Asset) -> float:
        """
        Given a DatetimeLike object, returns the data value (representing an
        asset value) associated with that datetime.

        Parameters:
            date (DatetimeLike):
                The datetime whose associated price information is requested

            asset:
                The asset (or asset type) that is being valuated. Different
                potentially have different column names that they associate
                with open and closing prices.

        Returns:
            The price information at the given datetime
        """
        date = max(self.first, utils.to_datetime(date))
        # Calling asof on the index is WAY faster than calling it directly on the frame
        data = self._data.loc[self._data.index.asof(date)]
        if date >= data["_period_close_"]:
            return data[asset.close_value]
        elif date >= data["_period_open_"]:
            return data[asset.open_value]
        else:
            return self.valuate(self.prev(date), asset)


# NOTE: THE NOTE BELOW IS ONLY RELEVANT FOR COLUMN ENUM DATASETS.
# THE NEW IMPLEMENTATION DOES NOT REQUIRE THIS.

"""
This is a little bit hacky, but this this needs to be defined outside
of the scope of AssetData even though it is only intended to be used
in that class. This is because the COLUMNS enum defined within will
not allow the use of subclasses as values for enum members. By
defining it outside, we can use Value within the COLUMNS enum scope,
allowing us to bypass the requirement that all values be the same.
Ordinarily, we could just use 'settings=NoAlias', but it imposes too
many restrictions when loading saved asset datasets from pickles.
"""

# Value = namedtuple('Value', 'type name')


# Below is the implementation of asset datasets that use enumerations
# for columns names. This may be revisited in the future
"""
class AssetData(pd.DataFrame):

    class COLUMNS(Enum):

        DATE = Value('datetime64', 'DATE')
        OPEN = Value('float', 'OPEN')
        CLOSE = Value('float', 'CLOSE')

        def __str__(self):
            return self.name

        def __repr__(self):
            return f'\'{self.name}\''

        @property
        def type(self):
            return self.value[0]

    static = False

    def __init__(self, data=None, required=None, optional=None):

        # Check for column requirements
        required = required if required else {}
        optional = optional if optional else {}

        def column_format(column):
            return column.replace(' ', '_').upper()

        # Converting required and optional to dicts
        def to_dict(columns):
            if isinstance(columns, list):
                columns = [column_format(column) for column in columns]
                columns = {column: Value('float', column)
                           for column in columns}

            elif isinstance(columns, dict):
                columns = {
                    column_format(column): Value(
                        columns[column],
                        column_format(column)) for column in columns}

            return columns

        required = to_dict(required)
        optional = to_dict(optional)

        # Update COLUMNS enum to accomodate requirements
        columns = [self.DATE, self.OPEN, self.CLOSE]

        def extend_columns(columns):
            enums = [column.name for column in self.COLUMNS]
            columns = {
                k.replace(
                    ' ',
                    '_').upper(): v for k,
                v in columns.items()}
            for name in columns:
                if name not in enums:
                    extend_enum(self.COLUMNS, name, columns[name])

        extend_columns(required)
        extend_columns(optional)

        # Updating columns to contain requirements
        columns += [self.COLUMNS[name] for name in required]

        # Handling AssetData inputs
        if isinstance(data, AssetData):
            if all([column in data.columns for column in required]):
                super().__init__(data)
                return
            else:
                raise ValueError('Dataset missing required columns')

        # Handling NoneType inputs
        data = 0 if data is None else data

        # Handling inputs that will result in single row dataframes
        if is_numeric(data):
            self.static = True
            data = pd.DataFrame(
                [[datetime.today()] + [data] * (len(columns) - 1)], columns=columns, dtype=float)

        # Handling non DataFrame inputs, checking for required columns
        else:
            if isinstance(data, str):
                data = pd.read_csv(data)
            elif isinstance(data, np.ndarray):
                data = pd.DataFrame(data)

            # Converting all columns to enums, dropping all columns which do
            # not exist in the enumeration
            to_convert = []
            to_drop = []
            data.columns = [column.replace(' ', '_').upper()
                            for column in data.columns]
            available = [column.name for column in self.COLUMNS]
            for column in data.columns:
                if column in available:
                    to_convert.append(self.COLUMNS[column])
                else:
                    to_drop.append(column)
            data.drop(to_drop, axis=1, inplace=True)
            data.columns = to_convert

            # Verifying existence of all required columns
            if not all([column in data.columns for column in columns]):
                raise ValueError(f'Dataset missing required columns')

            # Verifying column-specific dtypes
            data = data.astype(
                {column: column.type for column in data.columns})

        # Making the dataset datetime indexed, removing duplicate column
        if self.static or not data.empty:
            data = data.set_index(pd.DatetimeIndex(data[self.DATE]))
            data.drop(self.DATE, axis=1, inplace=True)

        super().__init__(data=data)

    # The limited scope of asset datasets allows the explicit definition of
    # boolean conversions
    def __bool__(self):
        return not (self.empty or self.static)

    # Allows the user to get COLUMN enum members as though they were
    # attributes of the dataset
    def __getattr__(self, attr):
        try:
            return self.COLUMNS[attr]
        except KeyError:
            return super().__getattr__(attr)

    # Allows the user to use strings to dynamically access the enum column
    # names
    def __getitem__(self, item):
        try:
            item = self.COLUMNS[item]
        except KeyError:
            pass
        return super().__getitem__(item)

    # Allows the user to dynamically access column enum members
    def get_column(self, column):
        try:
            return self.COLUMNS[column.replace(' ', '_').upper()]
        except KeyError:
            return None
"""

# OLD LEDGER SYSTEM

"""

class Ledger(pd.DataFrame):

    def instance(): return None

    def __new__(cls, *args):
        if cls.instance() is not None:
            return cls.instance()
        return super().__new__(cls)

    def __init__(self, data=None):
        # Try to access a ledger if one already exists
        try:
            data = data if data is not None else pd.read_pickle(
                'AlphaGradient/data/ledger')

            if not isinstance(data, (Ledger, pd.DataFrame)):
                raise TypeError('Invalid input for Ledger')

            super().__init__(data)

        # Otherwise, create a new one
        except FileNotFoundError:
            data = pd.DataFrame(
                columns=[
                    'ID',
                    'TYPE',
                    'NAME',
                    'STATUS',
                    'DATE'])
            super().__init__(data)
            self.auto_update()

        Ledger.instance = lambda: self

    def append(self, *args):
        data = super().append(*args)
        return Ledger(data)

    def to_pickle(self):
        pd.to_pickle(self, 'AlphaGradient/data/ledger')

    def auto_update(self):

        for raw_file in scandir('AlphaGradient/data/raw'):
            name = raw_file.name.split('.')[0]
            data = self.loc[self['ID'] == name]
            if data.empty:
                asset_type, asset_name = self.id_info(name)
                entry = pd.DataFrame([[name, asset_type, asset_name, 1, datetime.today()]], columns=[
                                     'ID', 'TYPE', 'NAME', 'STATUS', 'DATE'])
                self = self.append(entry)
            else:
                index = data.index.item()
                self.at[index, 'STATUS'] = 1
                self.at[index, 'DATE'] = datetime.today()

        for pickle in scandir('AlphaGradient/data/pickles'):
            data = self.loc[self['ID'] == pickle.name]
            if data.empty:
                asset_type, asset_name = self.id_info(pickle.name)
                entry = pd.DataFrame([[pickle.name, asset_type, asset_name, 2, datetime.today(
                )]], columns=['ID', 'TYPE', 'NAME', 'STATUS', 'DATE'])
                self = self.append(entry)
            else:
                index = data.index.item()
                self.at[index, 'STATUS'] = 2
                self.at[index, 'DATE'] = datetime.today()

        self.to_pickle()

    def update(self, data):
        pass

    def update_entry(self):
        pass

    def add_entry(self):
        pass

    @staticmethod
    def id(asset_type, asset_name):
        return f'{asset_type}_{asset_name}'.strip().upper()

    @staticmethod
    def id_info(ledger_id):
        # Decompose the ID
        info = ledger_id.split('_', 1)

        # Split into relevant information
        asset_type = info[0]
        asset_name = info[1]

        return asset_type, asset_name

    def get_status(self, ledger_id, asset_name=None):
        if asset_name is not None:
            ledger_id = self.id(ledger_id, asset_name)

        entry = self.loc[self['ID'] == ledger_id]

        if not entry.empty:
            return entry['STATUS'].item()

        return 0


def get_data_ledger(asset_type, asset_name, ledger=None):
    if not isinstance(ledger, Ledger):
        ledger = Ledger()

    ledger_id = ledger.id(asset_type, asset_name)
    status = ledger.get_status(ledger_id)

    data = None

    if status > 1:
        data = from_pickle(asset_type, asset_name)
        status = status - 1 if data is None else status

    if status <= 1:
        data = from_raw(asset_type, asset_name)

    return data

def from_pickle_ledger(asset_type, asset_name, ledger=None):
    # Check if an acceptable ledger is passed in
    if not isinstance(ledger, Ledger):
        ledger = Ledger()

    # Accessing the entry for this asset
    ledger_id = ledger.id(asset_type, asset_name)
    status = ledger.get_status(ledger_id)

    if status > 1:
        try:
            return pd.read_pickle(f'AlphaGradient/data/pickles/{ledger_id}')

        except FileNotFoundError:
            if status == 2:
                print('update ledger!')
            return None

    return None


def from_raw_ledger(asset_type, asset_name, ledger=None):
    # Check if an acceptable ledger is passed in
    if not isinstance(ledger, Ledger):
        ledger = Ledger()

    # Accessing the entry for this asset
    ledger_id = ledger.id(asset_type, asset_name)
    status = ledger.get_status(ledger_id)

    if status > 0:
        try:
            return AssetData(
                pd.read_csv(f'AlphaGradient/data/raw/{ledger_id}'))

        except FileNotFoundError:
            if status == 1:
                print('update ledger!')
            return None

    return None
"""
