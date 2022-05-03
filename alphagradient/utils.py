# -*- coding: utf-8 -*-
"""Standard utility functions used throughout AlphaGradient"""

# Standard Imports
from __future__ import annotations

from abc import ABC, abstractmethod
import builtins
from datetime import (
    date,
    datetime,
    time,
    timedelta,
)
import math
from pathlib import Path

# Third Party Imports
import numpy as np
import pandas as pd

# Typing
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Generator,
    Generic,
    Iterable,
    Optional,
    TypeVar,
    Union,
)

T = TypeVar("T")


class PropertyType(Generic[T]):
    """A Type class for property objects themselves, before being bound to a class instance"""

    def fget(self, *args: Any) -> T:
        ...


Property = builtins.property
"""A Type for builtin properties that have been bound to a class instance"""

PyNumber = Union[int, float]
"""Numeric type that does not include complex numbers (only native python types)"""

Number = Union[PyNumber, np.number, pd.core.arrays.numeric.NumericDtype]
"""Numeric type that does not include complex numbers"""

DatetimeLike = Union[pd.Timestamp, np.datetime64, date, datetime, str]
"""Objects convertable to python datetimes"""

TimeLike = Union[time, str]
"""Objects convertable to python time objects"""

DateOrTime = Union[DatetimeLike, time]
"""Objects that are either DatetimeLike or TimeLike in nature"""

if TYPE_CHECKING:
    from typeshed import SupportsLessThanT as SLTT

_global_persistent_path: PropertyType[Path]


def auto_batch(iterable: Iterable) -> Generator:
    """
    Returns a generator which yields automatically sized batches

    Given a sized iterable, determines an optimal batch size to be used for
    multiprocessing purposes. Using this batch size, returns a generator which
    yields batches of the iterable with the optimal size

    Parameters:
        iterable: An iterable from which to create a batch generator

    Returns:
        The batch generator of the iterable input
    """
    return get_batches(iterable, auto_batch_size(iterable))


def auto_batch_size(iterable: Iterable) -> int:
    """
    Returns a multiprocessing-optimal batch size for an iterable

    Given an iterable, returns an integer value representing an optimal batch
    size for use in python's multiprocessing library

    Parameters:
        iterable (Iterable): Sized iterable to determine optimal batch size for

    Returns:
        The optimal batch size for multiprocessing
    """
    # Converting to a sized iterable to guarantee __len__ functionality
    iterable = list(iterable)

    # Output Parameters
    horizontal_offset = 10000
    horizontal_stretch = 70 / 100_000_000
    vertical_offset = 100

    # Building the quadratic
    output: Number
    output = len(iterable) - horizontal_offset
    output = output**2
    output *= -1
    output *= horizontal_stretch
    output += vertical_offset

    # Output bounded between 30 and 100
    return bounded(int(output), lower=30, upper=100)


def bounded(
    to_bound: SLTT, lower: Optional[SLTT] = None, upper: Optional[SLTT] = None
) -> SLTT:
    """
    Bounds an object between a lower and upper bound

    Given an object that defines behavior for comparison (__lt__, __gt__),
    returns the object bounded between the lower and upper bounds. Boundaries
    will be ommited if they are not provided (None). If lower and upper are not
    None, they must be of the same type as to_bound.

    Type Explanation:
        SLTT (SupportsLessThanT): A TypeVar which implements the __lt__ method.

    Parameters:
        to_bound (SLTT): the object to be bounded
        lower (Optional[SLTT]): the lower boundary of the operation
        upper (Optional[SLTT]): the upper boundary of the operation

    Returns:
       The bounded object
    """
    if lower is None and upper is None:
        raise ValueError(
            "Of the parameters 'lower' and 'upper', at least one must be" "specified"
        )
    if lower:
        to_bound = max(to_bound, lower)
    if upper:
        to_bound = min(to_bound, upper)

    return to_bound


def deconstruct_dt(dt: DateOrTime) -> dict[str, float]:
    """
    Returns a dictionary of datetime attribute values on object 'dt'

    Given a DatetimeLike object, returns a dictionary where keys are the
    object's date and time related attribute names, and values are the object's
    associated attribute values.

    Parameters:
        dt (DateOrTime): the dt to deconstruct

    Returns:
        A dictionary of attributes and their associated values on dt

    Raises:
        TypeError: Raised if dt is not a datetime-like object, as it wont have
        the proper attributes.
    """
    # The potential attributes to be accessed
    d = ["year", "month", "day"]
    t = ["hour", "minute", "second", "microsecond"]
    attrs = []

    # Accept string arguments to convert to datetime
    if isinstance(dt, str):
        dt = read_timestring(dt)

    # Determine which elements should be accessed on the dt
    if isinstance(dt, datetime):
        attrs = d + t
    elif isinstance(dt, time):
        attrs = t
    elif isinstance(dt, date):
        attrs = d
    else:
        raise TypeError(f"{dt=} is not a valid datetime object")

    # Collecting the attributes
    dtdict = {}
    for attr in attrs:
        dtdict[attr] = getattr(dt, attr)

    return dtdict


def get_batches(iterable: Iterable, size: int = 100) -> Generator:
    """
    Returns a generator of the iterable which yields batches of the given size

    Given an iterable, uses the size parameter to create a generator which
    yields batches of the iterable of the given size.

    Parameter:
        iterable: The iterable to yield batches of
        size: The batch size of the returned generator

    Returns:
        A generator which yields batches of size 'size' of the iterable
    """
    # Because we will be indexing the iterable, we must instantiate the entire
    # thing in memory in case it isnt (ie generators)
    iterable = list(iterable)
    last = len(iterable)
    for i in range(math.ceil(last / size)):
        start = i * size
        end = start + size
        end = end if end < last else last
        yield iterable[start:end]


def get_time(t: DateOrTime) -> time:
    """
    Given a timestring or datetime-like object, returns a datetime.time object

    Given an object t which represents a time or a datetime, returns a native
    python datetime.time object of the appropriate time. t can be an isoformat
    time string or datetime string, or a datetime-like object

    Parameters:
        dt (DateOrTime): The time object to convert

    Returns:
        The converted datetime.time object
    """
    if isinstance(t, (time, str)):
        return to_time(t)
    return to_datetime(t).time()


def get_weekday(dt: DatetimeLike) -> str:
    """
    Returns the day of the week on which a DatetimeLike object falls

    Parameters:
        dt (DatetimeLike): The object whose weekday is determined

    Returns:
        String of the day of the week on which the DatetimeLike object falls
    """
    weekdays = {
        0: "Monday",
        1: "Tuesday",
        2: "Wednesday",
        3: "Thursday",
        4: "Friday",
        5: "Saturday",
        6: "Sunday",
    }

    return weekdays[to_datetime(dt).weekday()]


def is_func(f: Any) -> bool:
    """
    Returns a boolean value indicating whether or not f is a kind of function

    Given an object f, returns a boolean value indicating whether or not the
    object is a function. Idenfities all python objects whose sole or primary
    purpose is to be called directly, rather than objects that simply support
    an implementation of __call__.

    Behavior is slightly different than the inspect module's isfunction(), as it
    includes methods (bound and unbound), as well as abstract, static, and class
    methods.

    A 'function' is an instance of any of the following:
        * function
        * method (bound or unbound)
        * staticmethod
        * classmethod
        * abstractmethod
        * lambda
        * built-in-function

    Parameters:
        f: The object who's status as a function is being determined

    Returns:
        True if f is a method, function, builtin-method-or-function, or lambda,
        else False
    """

    # Fake class to access type 'method' and 'classmethod'
    class C:
        def method(self):
            pass

    # Getting abstract base methods
    class ABCC(ABC):
        @abstractmethod
        def amethod(self):
            pass

    # Fake function to access type 'function'
    def func():
        pass

    # Getting classic and static methods
    cmethod = classmethod(func)
    smethod = staticmethod(func)

    # Fake lambda to access type 'lambda'
    lamb = lambda: None

    # Fake instance to access type 'bound method'
    c = C()

    # Gathering all callable types
    functype = type(func)
    methodtype = type(C.method)
    classmethodtype = type(cmethod)
    staticmethodtype = type(smethod)
    abstractmethodtype = type(ABCC.amethod)
    boundmethodtype = type(c.method)
    lambdatype = type(lamb)
    builtintype = type(print)

    return isinstance(
        f,
        (
            functype,
            methodtype,
            boundmethodtype,
            lambdatype,
            builtintype,
            abstractmethodtype,
            classmethodtype,
            staticmethodtype,
        ),
    )


def nearest_expiry(
    expiry: DatetimeLike, method: Literal["after", "before", "both"] = "after"
) -> datetime:
    """
    Returns the nearest valid expiry to the input datetime object

    Determining expiries for options contracts can be difficult, because they
    must fall on a business day, and their expiry time must be the market close.
    Given an expiry whose validity is unknown, this function returns the
    nearest expiry that is guaranteed to be valid. If the given expiry is
    valid, it will be unchanged when it is returned.

    The method argument is used to determine how the 'nearest' is defined. It
    has three options: "after", "before", and "both"

    Method must be one of the following string literals:
        * "after": returns the nearest expiry that is AFTER the input expiry
        * "before": returns the nearest expiry that is BEFORE the input expiry.
        * | "both": compares the distances of the nearest before and after, and
          | return the smaller of the two. In the case that they are equal, the
          | date determined by "after" will be used.

    The default argument is "after" because using "before" or "both" can
    potentially lead to dangerous behavior for algorithms, as it can return an
    expiry which is before the current date of the algorithm. This can cause
    options contracts to initialize as expired. Only change the method
    argument if you are positive that the returned expiry will be greater
    than the algorithm's current date.

    Parameters:
        expiry (DatetimeLike):
            The expiry who's closest valid expiry will be determined

        method:
            One of "after", "before", or "both"

    Returns:
       The nearest valid expiry
    """

    # Ensuring expiry is a pydatetime
    expiry = to_datetime(expiry)

    # All expiries must expire at market close (4PM)
    expiry = set_time(expiry, "4:00 PM")

    # Change the expiry day if it is not a weekday
    if expiry.weekday() > 4:

        # Closest AFTER
        if method == "after":
            dist = 7 - expiry.weekday()
            expiry += timedelta(days=dist)

        # Closest BEFORE
        elif method == "before":
            dist = expiry.weekday() - 4
            expiry -= timedelta(days=dist)

        # Comparing both
        elif method == "both":
            bdist = expiry.weekday() - 4
            adist = 7 - expiry.weekday()
            if bdist < adist:
                expiry -= timedelta(days=bdist)
            else:
                expiry += timedelta(days=adist)

    return expiry


def progress_print(to_print: Any, last: list[int] = [0]) -> None:
    """Prints, but returns the carriage to the front of the last print"""
    print("\r" + (" " * last[0]), end="\r", flush=True)  # type: ignore[operator]
    print(to_print, end="", flush=True)
    last[0] = len(str(to_print))


def read_timestring(timestring: str) -> time:
    """
    Given a timestring, returns a datetime.time object representative of the time

    This function reads in 'timestrings', which are one of two things:
        #. | Isoformat times as strings, using 24 hours
           | (eg 04:00:00, 18:30, 02:59:59.99, etc)

        #. | Strings based on 12 hour clocks
           | (see ag.utils.read_twelve_hour_timestring docs)

    Using this timestring, returns a python datetime.time object corresponding
    to the time in the timestring. if dtype is set to dict, a deconstructed
    datetime attr dictionary will instead be returned. For more info on
    dtdicts, read the docs for ag.utils.deconstruct_dt

    Parameters:
        timestring:
            string representing the time

        dtype:
            The type of data to return

    Returns:
        The time or dict object corresponding to the time in the timestring
    """
    try:
        return read_twelve_hour_timestring(timestring)
    except (TypeError, ValueError) as e:
        return time.fromisoformat(timestring)


def read_twelve_hour_timestring(timestring: str) -> time:
    """Reads a timestring based on a 12 hour clock and returns a time

    Given a timestring representing a time on a 12 hour clock, returns the
    appropriate time object

    Must be formatted as follows:
        * hour   | This is the only required value, integer
        * minute | separated from hour by a colon, optional, integer
        * second | separated from minute by a colon, optional, float
        * AM/PM  | string 'AM' or 'PM', separated from second by a space

    When AM or PM is not provided in the timestring, AM will be assumed.

    Valid Examples:
        * '4:30 PM'
        * '4:30 AM'
        * '1 PM'
        * '1'
        * '11:59:59.999 PM'
        * '12:00:00 AM'

    Invalid Examples:
        * '0:00'
        * '13:30'
        * '103 PM'
        * '0'
        * '22'
        * '4:30:99 PM'
        * '3:99 PM'

    Parameters:
        timestring: The string containing the time to convert to a time object

    Returns:
        The corresponding time object

    Raises:
        TypeError:
            When timestring is not a string. Only str objects can be parsed

        ValueError:
            When the timetring is invalid / improperly formatted.
    """
    # Timestrings must be strs
    if not isinstance(timestring, str):
        raise TypeError(f"timestring must be a string, got {type(timestring)}")

    # Variable Initialization
    ampm = "AM"
    info = []
    timestring = timestring.split(" ")  # type: ignore[assignment]

    # Getting AM/PM component
    if len(timestring) > 1:
        ampm = timestring[1]

    # Getting individual time components
    info = timestring[0].split(":")

    # isoformat is 00:00:00.00, max 3 colons
    if len(info) > 4:
        raise ValueError(f"Failed to parse timestring {timestring}")

    # collecting the attributes necessary to create a time object
    tdict = {}
    attrs = ["hour", "minute", "second", "microsecond"]
    for attr, value in zip(attrs, info):
        tdict[attr] = int(value)

    # Setting missing components to 0
    for attr in attrs:
        if not tdict.get(attr):
            tdict[attr] = 0

    # hours less and 1 and more than 12 are off limits in 12 hour clocks
    if not 1 <= tdict["hour"] <= 12:
        raise ValueError(f"Failed to parse timestring {timestring}")

    # 12:30 AM is 00:30 isoformat
    if ampm == "AM" and tdict["hour"] == 12:
        tdict["hour"] == 0

    # 12:30 PM is 12:30 isoformat, 1:30 PM is 13:30 isoformat
    elif ampm == "PM" and tdict["hour"] < 12:
        tdict["hour"] += 12

    # Building and returning a time object
    return time(**tdict)  # type: ignore[arg-type]


def set_time(dt: DatetimeLike, t: DateOrTime) -> datetime:
    """Sets the given datetime-like object to the given time

    Given a DatetimeLike object 'dt' and a time-like object 't', returns a
    datetime like object that shares the date of dt and the time of t.

    Very similar to datetime.combine, but accepts datetime objects for both
    inputs.

    Parameters:
        dt (DatetimeLike): Datetime to convert
        t (DateOrTime): Time to convert to

    Returns:
        python datetime.datetime object with converted time
    """
    # Initializing the new time that will be set
    newtime: dict[str, float] = {}

    # Reading the necessary time attributes
    if isinstance(t, str):
        t = read_timestring(t)
        newtime = deconstruct_dt(t)
    elif isinstance(t, time):
        newtime = deconstruct_dt(t)
    else:
        newtime = deconstruct_dt(to_datetime(t).time())

    # Creating the new datetime with t=t
    return to_datetime(dt).replace(**newtime)  # type: ignore [arg-type]


def timestring(t: DateOrTime) -> str:
    """Converts a time-like object to a 12-hour-clock timestring

    Given a time-like object t, returns a timestring represented by the
    12-hour-clock (eg. 4:30 PM).

    Parameters:
        t (DateOrTime):
            date or time object to read into a 12-hour-clock-based timestring

    Returns:
        A string representing the time on a 12-hour-clock
    """
    # Ensuring that t is a time object
    if not isinstance(t, time):
        t = to_datetime(t).time()

    # Deconstructing components to create a time string
    ampm = "AM"
    hour = t.hour
    minute = t.minute if t.minute > 9 else f"0{t.minute}"
    if hour > 12:
        ampm = "PM"
        hour -= 12
    return f"{hour}:{minute} {ampm}"


def to_datetime(dtlike: DatetimeLike) -> datetime:
    """
    Given a datetime-like object, converts it to a python standard datetime

    Parameters:
        dtlike (DatetimeLike):
            The Datetime-convertable object

    Returns:
        The converted python datetime

    Raises:
        TypeError: Only accepts python-datetime-convertable objects
    """
    if isinstance(dtlike, datetime):
        return dtlike
    elif isinstance(dtlike, pd.Timestamp):
        return dtlike.to_pydatetime()
    elif isinstance(dtlike, np.datetime64):
        return pd.Timestamp(dtlike).to_pydatetime()
    elif isinstance(dtlike, date):
        return datetime.combine(dtlike, datetime.min.time())
    elif isinstance(dtlike, str):
        return datetime.fromisoformat(dtlike)

    raise TypeError(f"Can not convert passed object {dtlike} to python datetime")


def to_step(current: datetime, delta: Union[DateOrTime, timedelta, float]) -> timedelta:
    """
    Converts an ambiguous delta object to a python timedelta

    Given an amiguous object which can in some way be interpreted as a timedelta
    relative to some 'current' time, converts that object to an appropriate
    timedelta object, or 'step' in time.

    Parameters:
        current:
            The 'current' time, which determines how to interpret the delta

        delta (Union[DateOrTime, timedelta, float]);
            The object being passed that may represent a 'step' in time

    Returns:
        the appropriate timedelta 'step'

    Raises:
        TypeError:
            When passed a type that can not be coerced/interpreted

        ValueError:
            When a type-appropriate object can not be coerced, or is in some way
            invalid (eg. the step in time is BEFORE the current time)
    """
    # Multiple parses must be made on strings to successfully coerce all of them
    if isinstance(delta, str):
        try:
            delta = set_time(current, read_timestring(delta))
        except ValueError:
            delta = datetime.fromisoformat(delta)  # type: ignore[arg-type]

    elif isinstance(delta, time):
        delta = set_time(current, delta)

    elif isinstance(delta, (float, int)):
        delta = current + timedelta(days=delta)

    elif isinstance(delta, timedelta):
        delta = current + delta

    # if isinstance(delta, DatetimeLike):
    else:
        delta = to_datetime(delta)

    if delta > current:
        return delta - current

    raise ValueError(
        f"Passed delta {delta} is prior to current time {current}. Please "
        "choose a time AFTER the current date."
    )


def to_time(tlike: TimeLike) -> time:
    """
    Given a TimeLike object, converts it to a python standard time object

    Parameters:
        tlike (TimeLike):
            The time-convertable object

    Returns:
        The converted python time object

    Raises:
        TypeError: Only accepts python-time-convertable objects
    """
    if isinstance(tlike, str):
        return read_timestring(tlike)
    elif isinstance(tlike, time):
        return tlike

    raise TypeError(f"Can not convert passed object {tlike} to python time")


class NullClass:
    """
    A class designed to take the place of other functions, modules, or classes

    This class stands in place of a function, class, or module attached to
    another class as an attribute. When an attribute is initialized as a
    NullClass, one can safely access it as an attribute, call it, and access
    attributes on it. These actions can also be performed recursively; any of
    these operations performed on the nullclass will simply return itself,
    allowing them to be chained infinitely.

    Use this class in place of another function or class in order to safely
    use an attribute without making constant checks.

    This is most useful in place of functions/classes that perform
    logging/printing, but also makes sense in place of functions that modify
    things in place or always return None.

    Examples:
        .. highlight:: python
        .. code-block:: python

            class MyClass:
                def __init__(self, data, verbose=False):
                    # This is cleaner and more pythonic than...
                    self.print = print if verbose else NullClass()
                    self.print("Initialized as Verbose!")

                    # Alternative 1
                    self.print = print if verbose else lambda *args, **kwargs: None
                    self.print("Initialized as Verbose!")

                    # Alternative 2
                    self.print = print if print is verbose else None
                    if self.print is not None:
                        self.print("Initialized as Verbose!")

                    # Alternative 3
                    self.verbose = verbose
                    if self.verbose:
                        print("Initialized as Verbose!")

                    # etc etc etc...

                    # This is cleaner and more pythonic than...
                    self.tqdm = tqdm.progress_bar if verbose else NullClass()
                    with self.tqdm(total=1000) as pbar:
                        while condition:
                            self.do_something()
                            pbar.update(1) # Safe!

                    # Alternative
                    self.verbose = verbose
                    if verbose:
                        with tqdm.progress_bar(total=1000) as pbar:
                            while condition:
                                self.do_something()
                                pbar.update(1)
                    else:
                        while condition:
                            self.do_something() # gross.
    """

    def __call__(self, *args: Any, **kwargs: Any) -> NullClass:
        return self

    def __getattr__(self, attr: str) -> NullClass:
        return self

    def __enter__(self, *args, **kwargs) -> NullClass:
        return self

    def __exit__(self, *args, **kwargs) -> None:
        pass

    def __bool__(self) -> bool:
        return False
