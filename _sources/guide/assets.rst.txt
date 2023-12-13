.. _api.assets:

=========================
Assets and Financial Data
=========================

.. currentmodule:: alphagradient

Assets
------
In AlphaGradient, :class:`Assets <Asset>` are the fundamental objects that underlie all financial algorithms, representing a unit measure of a single financial asset. For example, an asset may represent a single share of stock, a single bond, or a single call option contract. 

In the case that an asset is not a financial instrument or otherwise lacks a unit measurement, it typically represents a single instance of the lowest purchasable (typically non-divisible) quantity of that asset. An example of assets of this type may be unique physical items like houses or cars.

To illustrate the necessity for this unit quantity representation of assets, consider a commodity that can be purchased in bulk at a reduced price. This commodity may be represented by several AlphaGradient assets representing different unit quantities, allowing bundled items to be purchased at a different price point than the sum of their constituents.

Assets themselves do not constitute any kind of financial position, and only a single instance of an asset may exist in any given :class:`Environment`. For example, there may never at once be two assets that represent stock of the same company, or two assets that represent options contracts of exactly the same strike price, maturity, and underlying security (within the same environment).

AlphaGradient assets should instead be conceived of as a more conceptually abstract representation of some unit measurement of an asset, from which financial positions can be created. For more information about financial positions in an asset, please refer to the documentation on :class:`portfolios <Portfolio>`.

Creating an Asset
+++++++++++++++++
Creating an asset in AlphaGradient is intuitive and highly flexible. AlphaGradient can automatically infer details from previous instantiations of an asset, allowing assets that have been created in the past to be instantiated using only a name. Take for example the creation of an asset representing stock of the index fund SPY. There are many different methods of instantiating such an asset.

.. code:: python

    import alphagradient as ag
    import datetime

    # Filling all parameters
    spy = ag.Stock(
        ticker = 'SPY', 
        data = 'path/to/your/data.csv'
    )

    # The easiest method
    spy_copy = ag.finance.Stock('SPY')

.. code:: pycon

    >>> spy
    <STOCK SPY: $446.52 /share>

The less data you provide when creating an asset, the more AlphaGradient will try to infer. When instantiated without data, assets will look for data available from previous instantiations, or data files in expected locations that match the asset's identifying information.

Of course, not all parameters are optional, and there will be different required parameters for different classes of assets. In the case of Stock, only the ticker is a required parameter. For more information about the distinctions between initialization of different asset classes, please refer to the documentation on :ref:`standard asset classes <api.standard>`.

When a duplicate of an asset is created such as in the case above, AlphaGradient will simply return the instance that already exists in memory. In the above example, both spy and spy_copy point to the same object in memory, essentially just acting as aliases. Attempting to copy or deepcopy an asset will raise an AssetDuplicationError, unless special settings in the class definition have specified otherwise.

.. code:: pycon

    >>> spy is spy_copy
    True

    >>> from copy import deepcopy
    >>> deepcopy(spy)
    AssetDuplicationError: Attempted duplication of SPY STOCK. Multiple instances of this asset are not within a single environment

Some special asset classes like :class:`Stock` have a large amount of data available online, and offer the ability to create an asset instance even in the absence of any data provided by the user, past or present. It should be noted that this function is only available to privileged asset classes that have large online availability (such as stock), and that there are no guarantees as to the data quality. Data pulled from online sources may have limited time resolution, accuracy, and precision.

Using an Asset
++++++++++++++
Once an asset exists in memory, portfolio objects can create long or short financial positions in the asset by by buying or shorting the asset.

.. code:: python

    # Creating a portfolio with $10,000
    portfolio = ag.Portfolio(10000)

    # Buying 10 shares of spy
    portfolio.buy(spy, 10)

    # Shorting 10 shares of spy
    portfolio.short(spy, 10)

Asset prices are automatically updated as time steps are taken over the course of an algorithm. Assets also come built in with a range of functions for visualization and analysis.

Creating a New Asset Class
++++++++++++++++++++++++++
If one wishes to create a new asset class, they can do so by subclassing AlphaGradient's base Asset class. Note that Asset is an abstract base class, and therefore has some required functionality that must be provided to be instantiable. New Asset types must provide a method of valuation (:meth:`Asset.valuate`) so that the Asset may be valuated at every time step even in the absence of data.

Subclassing an asset comes with many optional settings for controlling how financial data is formatted during asset initialization. Please refer to the documentation on :class:`Asset sublass intialization <Asset>` for specific settings and their options.

The function signature for subclass intialization is defined as such:

.. code:: python

    def __init_subclass__(
        cls,
        *args: Any,
        hidden: bool = False,
        require_data: Optional[bool] = None,
        prohibit_data: Optional[bool] = None,
        required: Optional[list[str]] = None,
        optional: Optional[list[str]] = None,
        open_value: Optional[str] = None,
        close_value: Optional[str] = None,
        market_open: Optional[DateOrTime] = None,
        market_close: Optional[DateOrTime] = None,
        units: tuple[str, str] = None,
        settings: dict[str, Any] = None,
        **kwargs: Any,
    ) -> None:

When creating a new asset class, developers should consider what data is available for the asset, and what data is minimally required for the asset to function as expected. For example, some Assets such as BrownianStock and other synthetics do not require data, as they generate it themselves. In the case that external data is required, the required and optional data (column names in a CSV or DataFrame) for the asset can be specified. Any column present in the data but not present in the arguments of the initialization will be ignored. If a data file acting as an input to an asset does not satisfy all of the columns specified in 'required', the asset will fail to initialize and an error will be raised. For this reason, be sure to include only those columns whose presence is **absolutely** necessary for the functioning of the asset.

Most assets will be valuated with data as time steps are taken. However, sometimes assets can be reasonably valuated without data. For example, one might use the Black Scholes equation to estimate options prices for which they lack the real price data. For this reason, all assets must define a _valuate method, which should return a floating point value representing the price of the asset. The default behavior provided by the Asset base class will be appropriate for most asset classes.

As an exercise, consider now the creation of an asset representing a virtual item for the competitive esport Counter Strike: Global Offensive:

.. code:: python

    import alphagradient as ag
    import datetime

    # Creating settings for the new asset type
    settings = {
        "require_data" = True
        "required" = ["max_bid"],
        "optional" = ["min_ask", "sell_orders", "buy_orders"],
        "open_value" = "max_bid"
        "units" = ("item", "items")
    }

    class CSGOItem(ag.finance.Asset, settings=settings):
        """An asset class representing a virtual item in CSGO"""

        def __init__(*args, item_condition, souvenier, **kwargs):
            self.item_condition = item_condition
            self.souvenier = souvenier
            super().__init__(*args, **kwargs)

        def valuate(self):
            """
            Valuate method does nothing, because the asset is data-dependent
            """
            return self.value

    dragon_lore = CSGOItem(
        name = 'dragon lore', 
        item_condition = 'factory new', 
        data = 'steam/marketplace/data.csv',
        souvenier = True
    )

.. code:: pycon

    >>> dragon_lore
    <CSGOITEM DRAGON_LORE: $68,760.70 /item>

When an asset is valuated through data, the standard implementation looks to use a column labeled 'CLOSE' to identify where to get price information for that time interval. The column responsible for data-based valuation can be changed by specifying the 'open_value' setting in the subclass definition.

CSGO items are bought and sold on the Steam Marketplace, where large bid ask spreads exist and markets are typically highly illiquid. In this context, it makes more sense to represent the 'price' of an asset as the highest available bid, as there may not be sales of the item recent enough to accurately reflect its market value. Finding the average of the max bid and min ask is another option, which would likely be more accurate but would require more data ('min_ask' would instead be in the required columns rather than the optional columns). The specifics of the implementation are at the discretion of the developer, of course, but this is merely an example of how one might accomplish that.

Asset Access
++++++++++++
All subclasses of asset, as well as some other AlphaGradient primitives, are registered with the global environment (as well as the environment they are instantiated in) when created. This allows for easy access to locally and globally instantiated assets, and provides a hook for controlling assets when they might otherwise be inaccessible.

.. code:: pycon

    >>> ag.csgoitem.dragon_lore
    <CSGOITEM DRAGON_LORE: $68,760.70 /item>

    >>> ag.csgoitem.dragon_lore is dragon_lore
    True

    >>> ag.stock.spy
    <STOCK SPY: $446.52 /share>

    >>> ag.stock.spy is spy
    True

Using the class name itself, one can also access a weakly referential dictionary of all of the instances of that asset type that have been instantiated in that environment. Accessing it on the top-level module itself (the global environment) will access all assets of that type instantiated in **any** environment.

One can also access all instantiated assets in an environment by accessing 'assets' on the environment.

.. code:: pycon

    >>> ag.stock
    {'SPY': <STOCK SPY: $446.52 /share>}

    >>> ag.csgoitem
    {'DRAGON_LORE': <CSGOITEM DRAGON_LORE: $68,760.70 /item>}

    >>> ag.assets
    [<STOCK SPY: $446.52 /share>, <CSGOITEM DRAGON_LORE: $68,760.70 /item>]


Financial Data for Assets
-------------------------
Financial data underlies the functioning of the majority of assets within AlphaGradient. Most assets will primarily be valuated by checking for price data within a specified time interval. With very limited exception, all of the financial data to be used in financial algorithms is to be provided by the user. When available, AlphaGradient is able to provide very limited data for assets like stocks, but the quality of this data is not guaranteed, and it should never be relied upon in a professional setting.

Data to be used in an AlphaGradient asset must follow some specific guidelines to be admissible. Most importantly, AlphaGradient datasets are those which associate a series of prices with a series of of time intervals. At minimum, one must have a column denoting the beginning of the period and one column denoting the associated price data for that interval in order for the table to be acceptable. Time intervals need not be consistent or follow any kind of pattern; even tick level data is acceptable. 

What Data can be Used?
++++++++++++++++++++++
AlphaGradient :class:`AssetData <dtypes.AssetData>` objects exist at any Asset's 'data' attribute. These are rigorously formatted pandas DataFrame objects which have met the specific formatting requirements of the particular asset class they were made for. Asset Datasets are extremely flexible in the type of inputs they consider to be :class:`ValidData`. Asset Datasets automatically interpret inputs of all different types and coerce them into highly formatted, AlphaGradient-compliant pandas DataFrames. However, the actual tabular data used as an input must meet some minimum criteria to be considered acceptable.

Criteria:
    * The data must have a column corresponding to the date/datetime of each valuation period, either as the first column or labeled explicitly as "DATE". This will later be transformed into into the datetime index of the asset dataset. Alternatively, passing in a DataFrame object that is already datetime indexed is also acceptable.
    * At least one column must refer to the price of the asset at some point during the valuation period, be it beginning, middle or end. This name of this column must either be 'CLOSE' or the name specified in the 'open_value' setting. This is a required column by default, and is the only required column if none are additionally provided by the user.

If these criteria are met, the tabular data is viable for conversion to an Asset Dataset.

Examples of Acceptable Data
+++++++++++++++++++++++++++
Suppose we have the following CSV, which contains trading data for SPY for the dates 2021-06-21 through 2021-07-01

.. list-table:: SPY Data
    :header-rows: 1

    * - DATA
      - OPEN
      - HIGH
      - LOW
      - CLOSE
      - ADJ VOLUME
      - VOLUME
    * - 2021-06-21
      - 416.799988
      - 421.059998
      - 415.929993
      - 420.859985
      - 420.859985
      - 72822000
    * - 2021-06-22
      - 420.850006
      - 424.000000
      - 420.079987
      - 423.109985
      - 423.109985
      - 57700300
    * - 2021-06-23
      - 423.190002
      - 424.049988
      - 422.510010
      - 422.600006
      - 422.600006
      - 49445400
    * - 2021-06-24
      - 424.890015
      - 425.549988
      - 424.619995
      - 425.100006
      - 425.100006
      - 45110300
    * - 2021-06-25
      - 425.899994
      - 427.089996
      - 425.549988
      - 426.609985
      - 426.609985
      - 58129500
    * - 2021-06-29
      - 427.880005
      - 428.559998
      - 427.130005
      - 427.700012
      - 427.700012
      - 35970500
    * - 2021-06-30
      - 427.209991
      - 428.779999
      - 427.179993
      - 428.059998
      - 428.059998
      - 64827900
    * - 2021-07-01
      - 428.869995
      - 430.600006
      - 428.799988
      - 430.429993
      - 430.429993
      - 53365900
    
There are several ways that this data can be used to create an asset within AlphaGradient.

.. code:: python

    import alphagradient as ag
    import pandas as pd
    import numpy as np

    path = 'path/to/spy.csv'
    date = '2021-06-23'
    columns = ['date', 'open', 'high', 'low', 'close', 'adj close', 'volume']

    # Method 1: Reading in from a file path
    spy = ag.Stock(name='SPY', data=path)

    # Method 2: Passing in a dataframe
    data = pd.read_csv(path)
    spy = ag.Stock('SPY', data)

    # Method 3: Passing in an ndarray with columns
    data = np.genfromtxt(path, delimiter=',')
    spy = ag.Stock('SPY', data, columns=columns)

    # Method 4: Passing in a nested list
    data = data.tolist()
    spy = ag.Stock('SPY', data, columns=columns)

    # Making sure the data is valuated on 'date'
    ag.globals.sync(date)

All of the methods above produce a viable asset <SPY STOCK $422.60 /share> evaluated on 2021/06/23, with data available from the twenty first of June to first of August, 2021. Whatever data you may have available, it can be used in AlphaGradient assets as long as it can be transposed into one of these formats prior to asset instantiation.

Non Data-Oriented Valuation of Assets
+++++++++++++++++++++++++++++++++++++
Some Assets can be valuated without time indexed data at each step, and some are even preferentially valuated without data at each time step. In some assets, non data-based valuation make take precedence depending on settings in the class definition. For example, one could create an asset that merely serves the purpose of emulating another asset but without any need for data, such that many hundreds or even thousands of them could be instantiated at once without any need for expensive, high level data. Consider a synthetic stock that undergoes valuation through geometric brownian motion:

.. code:: python
    import alphagradient as ag

    # Generate 100 random, simulated stock assets in a list
    simulated = [ag.BrownianStock for _ in range(100)]

Using methods such as this, one can quickly test the efficacy of their algorithms before moving on to real backtesting. They can also test algorithms that operate on time resolutions for which they don't have access to real, reliable data, because simulated assets can be valuated at arbitrarily precise time resolutions.