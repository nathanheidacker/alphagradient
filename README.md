# AlphaGradient
![Tests](https://github.com/nathanheidacker/alphagradient/actions/workflows/tests.yml/badge.svg)

## Introduction

**AlphaGradient** is a package for creating and backtesting financial algorithms in native python. AlphaGradient implements asset and portfolio-analagous datastructures that interact in intuitive ways, and provides a framework for developing algorithms that utilize these objects in highly parallelized backtests. AlphaGradient is built on top of the industry's most widely adopted libraries, and requires (comparatively) little technical knowhow and time investment to pick up.

To be able to use AlphaGradient effectively, you must know...
- basic finance
- **really** basic python

Where other libraries might require you to be a programmer first and and a certified quantitative financial analyst second, AlphaGradient is truly geared towards hobbyist algorithmic traders who want to play around with ideas for financial algorithms. Within minutes, you can have a fully backtestable algorithm using nothing but basic python.

AlphaGradient is to the algorithmic trader as console gaming is to the gamer; it gets the job done and comes at a bargain--batteries included. If you have even a passing interest in finance or algorithmic trading, this package makes it as easy as possible to get involved without having a background in computer science.

## Installation
The source code is currently hosted on GitHub at:
https://github.com/nathanheidacker/alphagradient

Binary installers for the latest released version are available at the [Python
Package Index (PyPI)](https://pypi.org/project/alphagradient)

```
# using pip
pip install alphagradient
```

## Dependencies
- [pandas](https://github.com/pandas-dev/pandas)
- [numpy](https://github.com/numpy/numpy)
- [aenum](https://github.com/ethanfurman/aenum)
- [yfinance](https://github.com/ranaroussi/yfinance)
- [tqdm](https://github.com/tqdm/tqdm)
- [pathos](https://github.com/uqfoundation/pathos)

## License
AlphaGradient is licensed under the [Apache License](LICENSE)

## Background
As it stands currently, AlphaGradient is still just a personal project; it is not intended for professional use, and does not guarantee the accuracy of its backtesting results. AlphaGradient is a tool intended for hobbyist algorithmic traders to quickly iterate on ideas for algorithms, as well as get a preliminary understanding of an algorithm's success.

This is a very early pre-alpha release of this package. It is not stable, and needs a massive amount of additional testing. However, most functionality should be operable for the time being.

## Documentation
More in-depth documentation is hosted on [TBD]

## Quick-Start Guide

### Assets

#### Explanation
In AlphaGradient, assets are the fundamental objects that underlie all financial algorithms, representing a unit measure of a single financial asset. For example, an asset may represent a single share of stock, a single bond, or a single call option contract.

Assets themselves do not constitute any kind of financial position, and only a single instance of an asset may exist at one time. For example, there may never at once be two assets that represent the same type of stock of the same company, or two assets that represent options contracts of exactly the same strike price, maturity, and underlying security.

AlphaGradient assets should instead be conceived of as a more conceptually abstract representation of some unit measurement of an asset, from which financial positions can be created.

#### Examples

```python
import alphagradient as ag
from datetime import datetime, timedelta
from math import floor

# Date as of writing this documentation
today = datetime.fromisoformat(f"2022-04-06 16:00:00")

# Setting the global date to today
ag.globals.sync(today)

# This works works even with two different data arguments, or none at all
spy = ag.finance.Stock("SPY", data="path/to/your/data.csv")
other_spy = ag.finance.Stock("SPY")

# Creating a SPY call that expires thursday next week
strike = math.floor(spy.value) + 5
expiry = datetime.fromisoformat("2022-04-14 16:00:00")
call = ag.finance.Call(spy, strike=strike, expiry=expiry)
```
```
>>>spy
<STOCK SPY: $446.52 /share> // as of April 6th, 2022 EOD

>>>spy is other_spy
True

>>>call
<CALL SPY442C2022-04-22: $271.73 /contract>
```
### Portfolios

#### Explanation

Portfolios are the means by which one can trade assets within a financial algorithm. Portfolios provide functionality for entering and exiting both long and short positions, and automatically handle valuation of the assets in which they have a financial position. For example, a portfolio with a position in an asset with a terminal maturity will automatically handle the expiry of the position upon reaching the maturity.

Portfolios store a detailed history of their positions at every valuation period. The class provides a multitude of ways of presenting and using this data for further analysis.

#### Positions
A position represents a financial stake in an asset. Portfolios create positions in assets when trading them. Like a portfolio, a position also keeps track of its own value history at every valuation.

Positions in an asset may either be long or short--that is, owned or owed. Only certain assets provide functionality for entering a short position, and attempting to short assets that do not explicitly support it will raise an exception.

Positions are typically not interacted with directly, but are modulated by using their respective portfolio as an interface.

#### Examples
```python
# Create a portfolio with $100,000 USD
initial = 100_000
portfolio = ag.finance.Portfolio(initial)

# Selling two covered calls
portfolio.buy(spy, 200)
portfolio.short(call, 2)

spy_position = portfolio.get_position(spy)
call_position = portfolio.get_position(call, short=True)
```

Note that relative to the portfolio that holds them, short positions are always negative as they represent a debt that must be repaid.
```
>>>portfolio.positions
{'CASH': <USD $11451.91>, 'STOCK_SPY_LONG': <200 shares @ $446.52 | RETURN: 0.0%>, 'SPY451C2022-04-14_SHORT': <2 contracts @ $-377.96 | RETURN: 0.0%>}

>>>portfolio.cash
<USD $11451.91>

>>>spy_position
<200 STOCK_SPY shares @ $446.52 /share | VALUE: $89304.0 | RETURN: 0.0%>

>>>call_position
<2 SPY451C2022-04-14 contracts @ $-377.96 /contract | VALUE: $-755.91 | RETURN: 0.0%>
```
```python
# Stepping ahead in time by one week
today += timedelta(days=7)
ag.globals.step(today)
```
```
>>>portfolio.positions
{'CASH': <USD $11451.91>, 'STOCK_SPY_LONG': <200 shares @ $446.52 | RETURN: -0.72%>, 'SPY451C2022-04-14_SHORT': <2 contracts @ $-377.96 | RETURN: 96.03%>}

>>>spy
<STOCK SPY: $443.31 /share>

>>>call
<CALL SPY451C2022-04-14: $14.99 /contract>

>>>spy_position
<200 STOCK_SPY shares @ $446.52 /share | VALUE: $88662.0 | RETURN: -0.72%>

>>>call_position
<2 SPY451C2022-04-14 contracts @ $-377.96 /contract | VALUE: $-29.98 | RETURN: 96.03%>
```
```python
# Liquidating all positions
portfolio.liquidate()
profit = portfolio.liquid - initial
roi = round(profit / initial * 100, 3)
roi = f"{roi}%"
```
```
>>>portfolio.positions
{'CASH': <USD $100083.93>}

>>>profit
83.92999999999302

>>>roi
0.084%
```
### Universes

#### Explanation
Universes are AlphaGradient's stock selection and filtering mechanism. Universes are (typically large) dictionaries of instantiated AlphaGradient stock objects that provide mechanisms for easy filtering by selection criteria.

#### Persistent Datasets
All AlphaGradient assets have the ability to persist their formatted datasets in memory, such that they can be easily reloaded to dramatically decrease initialization times. Universes are a particularly good place to introduce this functionality because they typically initialize many hundreds or even **thousands** of stock objects at once, which may constitute as much as 6-7 GB of formatted price history data. The first time around, this initialization process may take as long as 15 minutes, depending on one's internet and processor speed. With persistent memory, one can reinitialize over 10k Stock objects in just a few seconds.

Calling alphagradient.global.persist() will create a directory in the current working directory (or wherever you want if you pass in a path argument) that stores formatted AlphaGradient assets for reinitialization. If this directory already exists in your current working directory, it will be found automatically when AlphaGradient is imported. Basically, its a good idea to work inside of some kind of isolated project directory that stores persistent versions of all of the assets you use frequently.

It should be noted that this will work for all instantiated AlphaGradient assets, not just those that are created by Universe objects.

#### Universe Initialization

By default, initializing a universe object with no arguments will only instantiate stock objects that are stored in the persistent directory. This means that, if none has been set or found, a Universe will initialize completely empty. One can pass in an array-like or iterable of Stock objects if they wish to initialize it with those specifically. If these objects have not yet been instantiated, you can also pass in an array-like or iterable of stock tickers (uppercase strings, eg ["SPY", "DIA", "TSLA", "APPL"]).

Lastly, one can pass in "all", which will initialize the Universe with all available securities listed on a host of US exchanges, including the NYSE, NYSE ARCA, NASDAQ and many more. This is a large undertaking, including over 12,000 publicly traded companies, and will likely take the average computer over 15 minutes to complete. For this reason, Universe objects are initialized with verbose=True by default, which will display a progress bar of their intitialization progress. It is for this reason that we **strongly** recommend you define a directory for persistent storage.

#### Examples

```python
# Persisting our asset data locally
ag.globals.persist() // creates folder called 'alphagradient.persistent' in current directory
universe = ag.finance.Universe("all")

```
Heres the performance of a little script I made to demonstrate initializing from 0. This was peformed on a M1 Macbook Air with a modest internet connection (~100Mbps)
```
$ python3 make_universe.py

Beginning Script...

Initializing Universe: 12067 Stocks
-----------------------------------
[1]: adding 1 stock that is already instantiated
[2]: initializing 12066 stocks to be added to Universe
[3]: no local data detected, moving on to online initialization
[4]: initializing 12066 stocks from online data (121 batches)
100%|█████████████████████████████████████████████████████████████| 121/121 [13:08<00:00,  6.52s/it]
Successfully added 10419 of 12066 stocks (937 failures, 710 timeouts, 1647 total errors))

[5]: retrying 710 timeouts
[4]: initializing 710 stocks from online data (19 batches)
100%|███████████████████████████████████████████████████████████████| 19/19 [00:56<00:00,  2.97s/it]
576 / 710 successful timeout reattempts (63 failures, 71 timeouts, 134 total errors)

Elapsed Time:  847.0219991250001 seconds
```
Here's performance on the second run
```
$ python3 make_universe.py

Beginning Script...

Initializing Universe: 10996 Stocks
-----------------------------------
[1]: adding 1 stock that is already instantiated
[2]: initializing 10995 stocks to be added to Universe
[3]: initializing 10995 stocks from local data
100%|████████████████████████████████████████████████████████| 10995/10995 [00:15<00:00, 702.14it/s]

Successfully added 10995 of 10995 stocks (0 failures, 0 timeouts, 0 total errors))

Elapsed Time:  17.487481792 seconds
```

Notice that the second time, the stocks that have been identified as failures (rather than timeouts) are not included in the second initialization. These are stocks that may be delisted, mistyped, or non-existent altogether.

#### Filtering

Universe objects are fundamentally just dictionaries; they associate a stock's ticker with its instantiated asset. They provide all of the same functionality that a dictionary does when used normally. However, they also provide the ability to filter universe objects by accessing stock attributes on on the universe object. These filtering operations can also be joined to be performed in rapid succession.

Filtering operations return a UniverseView, which is a 'view' of an initialized universe with filters applied in order. These filters can be rearranged, added to, or removed in any order to produce a different view or return to the original object.

#### Examples

```python
# getting all stocks with value above $1000 USD
filtered = universe.filter[universe.value > 1000]
```

```
>>>len(universe)
10995

>>>len(filtered)
16

>>>filtered
{'AMZN': <STOCK AMZN: $3110.82 /share>, 'AVGOP': <STOCK AVGOP: $1855.54 /share>, 'AZO': <STOCK AZO: $2171.43 /share>, 'BKNG': <STOCK BKNG: $2260.11 /share>, 'CABO': <STOCK CABO: $1439.85 /share>, 'CMG': <STOCK CMG: $1590.0 /share>, 'GOOG': <STOCK GOOG: $2605.72 /share>, 'GOOGL': <STOCK GOOGL: $2597.88 /share>, 'MELI': <STOCK MELI: $1135.75 /share>, 'MKL': <STOCK MKL: $1479.23 /share>, 'MTD': <STOCK MTD: $1342.59 /share>, 'NVR': <STOCK NVR: $4419.99 /share>, 'SEB': <STOCK SEB: $4155.0 /share>, 'TPL': <STOCK TPL: $1458.55 /share>, 'TSLA': <STOCK TSLA: $1022.37 /share>, 'WTM': <STOCK WTM: $1075.63 /share>}
```

```python
# stocks above $1000 who have positive alpha (using SPY as a benchmark)
filtered = filtered.filter[universe.alpha() > 0]

# the above operation is identical to this (chaining filtering operations)
filtered = universe.filter[universe.value > 100, universe.alpha() > 0]
```

```
>>>len(filtered)
8

>>>filtered
{'AVGOP': <STOCK AVGOP: $1855.54 /share>, 'AZO': <STOCK AZO: $2171.43 /share>, 'GOOG': <STOCK GOOG: $2605.72 /share>, 'GOOGL': <STOCK GOOGL: $2597.88 /share>, 'MKL': <STOCK MKL: $1479.23 /share>, 'MTD': <STOCK MTD: $1342.59 /share>, 'SEB': <STOCK SEB: $4155.0 /share>, 'TSLA': <STOCK TSLA: $1022.37 /share>}

>>>filtered.history
Universe: 10996 stocks
----------------------
stock.value > 1000: 16 results | 99.854% (10980) removed | 99.854% (10980) total removed
stock.alpha() > 0: 8 results | 50.0% (8) removed | 99.927% (10988) total removed
```

### Environments

#### Explanation

In much the same way that a python programmer uses virtual environments to manage dependencies among disparate workflows, so too should an AlphaGradient user make explicit the distinct environment of assets that they intend to use. Valuating assets at each time step is computationally expensive, and if thousands of assets are instantiated but only a few are being used in an algorithm, runtimes can unnecessarily increase by orders of magnitude.

AlphaGradient Environments are the solution to this problem. They provide what is essentially an isolated, separate instance of AlphaGradient which tracks only those assets it is directed to track, and can iterate the valuation of those assets in isolation.

Environments provide essentially a clone of the AlphaGradient API, with the exception that they only operate on assets created within the environment. Assets created outside the environment can also be added into it retrospectively.

#### Initializing an Environment Object

Environment objects can be passed an iterable of assets when initialized, or nothing at all. However, all environments must have at least one portfolio. Environments accept a portfolio argument at initialization, which can be either a single portfolio or iterable of portfolios. The environment will determine a 'primary' portfolio, which will then be bound to environment.main. When passing an iterable for portfolios, be careful that the iterable is ordered, lest the primary portfolio be determined randomly.

After an environment has been created, one can use it in the same way they would use the AlphaGradient base module to intantiate new objects, as well as access them. Where AlphaGradient has truly global access to all assets, Environments have 'global' access to all of the assets that they contain, which are accessed in exactly the same way.

#### Examples

```python
# Creating SPY stock
spy = ag.finance.Stock("SPY")

# Initially this env will only contain spy
env = ag.finance.Environment(assets=spy)

# Creating a new asset within the environment
env.stock("DIA")

# Adding an asset from outside of the environment
qqq = ag.finance.Stock("QQQ")
env.track(qqq)
```
```
>>>env.assets
[<STOCK SPY: $443.31 /share>, <STOCK DIA: $345.76 /share>, <STOCK QQQ: $346.35 /share>]

>>>env.stock
{'SPY': <STOCK SPY: $443.31 /share>, 'DIA': <STOCK DIA: $345.76 /share>, 'QQQ': <STOCK QQQ: $346.35 /share>}

>>>env.stock.spy is ag.stock.spy
True

>>>env.date
2022-04-13 16:00:00

>>>env.stock.spy
<STOCK SPY: $443.31 /share>

>>>env.step(timedelta(days=1))
>>>env.date
2022-04-14 16:00:00

>>>env.stock.spy
<STOCK SPY: $437.79 /share>
```

### Algorithms

#### Explanation

Finally, AlphaGradient Algorithms are a framework for organizing all of these components to create backtestable financial algorithms. To create a new algorithm, simply subclass from ag.Algorithm and define two new functions, setup and cycle, which are discussed below. Anything passed to an instantiated algorithm object when called will be passed to both of these functions.

Both of these functions share an identical minimal function header. That is, if you don't define any additional arguments that you need for your algorithm, the minimum that both will always be passed (and should therefore be capable of accepting) is: (*args, start, end, ** kwargs). start and end will **always** be passed to both of these functions as explicit keyword arguments, so you must accept them, even if you don't use them. This means that the minimum function header for either setup OR cycle is as such:
```python
def setup(self, **kwargs):
    ...

def cycle(self, **kwargs):
    ...
```

If you need to accept additional arguments for either your setup or cycle, you can just add them:
```python
def setup(self, additional_positional_parameter, *args, **kwargs):
    ...

def cycle(self, *args, additional_keyword_parameter=None, **kwargs):
    ...
```

#### Defining Setup

Setup is the first of two functions that must be defined in any new algorithm. Setup is a function that creates/resets and returns an Environment object. The setup function is responsible for the creation of this object, and is called prior to every backtest. The environment returned by setup will automatically be bound to the algorithm, so it can be referred to in cycle.

#### Defining Cycle

Cycle is the second of two functions that must be defined in any new algorithm. Cycle defines the algorithm's behavior at each time step. cycle is run at every new step in time. Cycle should return None; it modifies the environment in place.

Setup defines the algorithm's initial environment, and cycle's job is to alter it over the course of the algorithm's runtime. Changes to the environment across the duration of the algorithms runtime are automatically recorded by the algorithm, and can be used to generate statistical performance measures after the backtest is complete.

#### Basic Example

Here's a super simple example to show the barebones minimum of an algorithm. It simply buys and holds SPY, starting with $1,000,000 USD.

```python
class IndexHold(ag.Algorithm):
    """A tutorial algorithm! Buy and Hold!"""

    def setup(self, *args, start, **kwargs):
        # Our initial balance
        initial = 1_000_000
        spy = ag.finance.Stock("SPY")

        # Creating an environment object
        env = ag.finance.Environment(start=start, assets=spy)

        # identical to env.main.invest(initial)
        env.invest(initial)

        # Calling this drastically improves runtime performance
        env.finalize()

        return env

    def cycle(self, *args, start, end, **kwargs):
        # Buying at the start...
        if self.date <= start:

            # Determining how much we can afford
            to_buy = math.floor(self.env.liquid / self.env.stock.spy.value)

            # This buys the asset on the main portfolio
            self.env.buy(self.env.stock.spy, to_buy)

        # And holding till the end!
        elif self.date >= end:

            # Selling everything
            self.env.liquidate()

```

#### Advanced Example

Here is a more advanced example that demonstates some common design patterns and emulates typical usage.

```python
class ThetaGang(ag.Algorithm):
  """An example algorithm in the algorithm library used to demonstrate some of
  AlphaGradient's standard features and best practices

  This is a tutorial algorithm that seeks to demonstrate some of AlphaGradient's
  features and standard design practices. This algorithm sells the maximum number
  of covered calls on SPY that it can, with a bounded strike price to prevent from
  selling calls that could lose money when assigned

  Here's a breakdown:

    1) At the beginning of the day, buy as many shares of SPY as we can to the
    nearest multiple of 100

    2) Using SPY shares as collateral, sells 1 DTE covered calls on SPY where
    the strike is determined by SPY's current value. The algorithm will never
    sell a call with a strike below it's average cost for the shares it owns.
    This prevents it from losing money in the case of call assignment.

    3) The strike bounding component of 2) is toggle-able by instantiating with
    bounded=False
  """

  def __init__(self, *args, bounded=True, **kwargs):
    super().__init__(*args, **kwargs)

    # Determines whether or not a lower bound should be placed on the strike
    self.bounded = bounded


  def setup(self, *args, start, end, **kwargs):
    # Creating a basket with the given start parameter
    env = ag.finance.Environment(start=start)

    # Creating SPY stock, attaching it to self (will be referenced frequently)
    # This call to the stock() method both instantiates the stock within the environment,
    # AND returns it, allowing us to set it as an attribute
    self.spy = env.stock("SPY")

    # Initial investment into the primary portfolio
    env.invest(self.spy.value * 150)

    # We only want the algorithm to evaluate at market open and close of each day
    # Finalizing will dramatically increase execution time, but is not necessary
    env.finalize(manual=["9:30 AM", "4:00 PM"])

    return env

  def cycle(self, *args, start, end, **kwargs):
    """The actions to perform at every valuation point"""

    # Selling as many covered calls on SPY as we can
    self.env.covered_call(self.generate_call())

    # The above line of code is a shortcut for:
    # self.env.main.covered_call(self.generate_call())

    # Showing the changes at every time step
    self.print(self.stats.change_report())

  def generate_call(self, delta=1):
    """Generates the ideal SPY call to be sold based on current circumstances"""

    # Getting our current position in the Asset <STOCK SPY>
    spy_position = self.env.get_position(self.spy)

    # Determining our optimal strike price
    optimal = math.floor(self.spy.value) + delta

    # Determining a lower bound for our strike price (the ceiling of our basis)
    lower_bound = optimal
    if spy_position and self.bounded:
      lower_bound = math.ceil(spy_position.average_cost)

    # Determining our strike price
    strike = max(optimal, lower_bound)

    # Determining the call expiry date (1 DTE)
    expiry = self.env.date + timedelta(days=1)

    # We can't sell calls with expiries on weekends or outside of market hours
    expiry = ag.utils.nearest_expiry(expiry)

    # Creating the call using the environment so that it doesnt have to be added retroactively
    return self.env.call(self.spy, strike, expiry)
```

This is a relatively complex algorithm, but here's what it looks like without the comments. It's less than 30 lines of code.
```python
class ThetaGang(ag.Algorithm):
  def __init__(self, *args, bounded=True, **kwargs):
    super().__init__(*args, **kwargs)
    self.bounded = bounded

  def setup(self, *args, start, end, **kwargs):
    env = ag.finance.Environment(start=start)
    self.spy = env.stock("SPY")
    env.invest(self.spy.value * 150)
    env.finalize(manual=["9:30 AM", "4:00 PM"])
    return env

  def cycle(self, *args, start, end, **kwargs):
    self.env.covered_call(self.generate_call())
    self.print(self.stats.change_report())

  def generate_call(self, delta=1):
    spy_position = self.env.get_position(self.spy)
    optimal = math.floor(self.spy.value) + delta
    lower_bound = optimal
    if spy_position and self.bounded:
      lower_bound = math.ceil(spy_position.average_cost)
    strike = max(optimal, lower_bound)
    expiry = self.env.date + timedelta(days=1)
    expiry = ag.utils.nearest_expiry(expiry)
    return self.env.call(self.spy, strike, expiry)
```

### Backtesting

Ok, so we've made a few algorithms, but how do we know if they work? Assume that the script below is a continuation of everything we've built up to this point, and is called backtest.py
```python
# pretend the algorithms have been defined above this point

def main():

    # Timing total runtime
    from time import perf_counter
    time = perf_counter()

    # Instantiating our algos
    index_algo = IndexHold()
    theta_algo = ThetaGang(bounded=True)

    # This start date will give us some backtesting space
    start = datetime.fromisoformat("2010-01-03")

    # This is the number of years we want to run for
    runtime = 10 # num years here
    runtime *= 365

    # Running the algorithms will return a Run object, which contain backtest results
    index_backtest = index_algo(start=start, end=runtime)
    theta_backtest = theta_algo(start=start, end=runtime)

    # Some output fluff
    print(f"Backtest performance from {start} to {start + timedelta(days=runtime)}")

    # Getting their performance metrics
    tests = [index_backtest, theta_backtest]
    for test in tests:
        stats = test.profile()
        print(f"{test.algo_name} Backtest: Profit: {stats["PROFIT"]} | ROI: {stats["ROI"]}")

    # Printing total runtime
    print(f"Elapsed time: {perf_counter() - time} seconds")

if __name__ == "__main__":
    main()
```
Here's the output:
```
$ python3 backtest.py
<IndexHold Algorithm Backtest: 3650 days, 0:00:00>: 100%|███████████████████████████████████████████████████████| 7300/7300 [00:04<00:00, 1591.71it/s]
<ThetaGang Algorithm Backtest: 3650 days, 0:00:00>: 100%|████████████████████████████████████████████████████████| 7300/7300 [00:16<00:00, 448.67it/s]

Backtest performance from 2010-01-03 00:00:00 to 2020-01-01 00:00:00
IndexHold Backtest: Profit: $1852415.81 | ROI: 185.24% | 2.85x
ThetaGang Backtest: Profit: $1195490.83 | ROI: 5427.5% | 55.28x

Elapsed Time:  20.984690541000003 seconds
```
Both of these algorithms are currently availble for anyone to test out under AlphaGradient's algolibrary. Access them from the algolib module such as in the example below.
```python
algo = ag.algolib.IndexHold()

# or

algo = ag.algolib.ThetaGang()
```

## About Me and Contact Info
Im a student from Northwestern University in Evanston Illinois studying cognitive science and artificial intelligence. I'm currently seeking employment. If you're interested in my work, feel free to contact me at nathanheidacker@gmail.com
