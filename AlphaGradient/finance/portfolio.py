# -*- coding: utf-8 -*-
"""AG module containing portfolio class

Todo:
    * Improve portfolio analysis tools
"""

# Standard imports
from datetime import datetime
from numbers import Number

# Third party imports
import pandas as pd
import numpy as np

class Position:
    """Object representing a position in a financial asset

    An object representing a financial stake in some underlying asset, to be used in portfolios to track holdings. Automatically updates value using the value of the underlying asset, and keeps track of average cost to enter the position.

    Attributes:
        asset (Asset): the underyling asset
        quantity (Number): the quantity of the asset represented by the
            position
        short (bool): True if short, False if long
        cost (Number): The total cost to enter this position
        value (Number): The current market value of this position
        average_cost (Number): The cost of this position per unit
        key (str): A key representing this position
    """

    def __init__(self, asset, quantity, short=False):
        self.asset = asset
        self.quantity = quantity
        self.short = short
        self.cost = self.value
        self.history = self._history()

    def __add__(self, other):
        # Positions can only be added to other positions
        if not isinstance(other, Position):
            return NotImplemented

        # Updating quantity
        self.quantity += other.quantity

        # Updating position cost
        short = -1 if self.short else 1
        self.cost += self.asset.price * other.quantity * short

        # Updating the position history
        self.update_history()

        return self

    def __sub__(self, other):
        # Positions can only be subtracted from other positions
        if not isinstance(other, Position):
            return NotImplemented

        # Updating quantity
        self.quantity -= other.quantity

        # Updating position cost
        short = -1 if self.short else 1
        self.cost -= self.asset.price * other.quantity * short

        # Updating position history
        self.update_history()

        return self

    def __eq__(self, other):
        return self.asset is other.asset and self.short is other.short

    def __str__(self):
        return f"{self.quantity} units @ ${self.average_cost} | MV: ${self.value}"

    def __repr__(self):
        return self.__str__()

    @property
    def key(self):
        short = "SHORT" if self.short else "LONG"
        return f"{self.asset.key}_{short}"

    @property
    def value(self):
        return round(self.asset.price * self.quantity * (-1 if self.short else 1), 2)

    @property
    def average_cost(self):
        return round(self.cost / self.quantity, 2)

    def view(self):
        """A memory efficient copy of the position

        Returns a 'view' of the position in its current state for use use in portfolio histories.

        Returns:
            view: memory efficient copy of the position
        """
        class View:
            def __init__(self, position):
                self.asset = position.asset.key
                self.quantity = position.quantity
                self.short = position.short
                self.cost = position.cost
                self.value = position.value
            def __str__(self):
                short = "SHORT" if self.short else "LONG"
                return f"{self.asset}_{short}: {self.quantity} @ ${self.value / self.quantity}"
            def __repr__(self):
                return self.__str__()

        return View(self)

    def _history(self):
        """A pandas DataFrame of this position's value history

        Returns a datetime indexed dataframe of this positions market value history and cost, which is automatically updates when changes in the position or the underlying asset occur

        Returns:
            history (pd.DataFrame): position history
        """
        history = pd.DataFrame(
                               [[self.value, self.cost]], 
                               columns=["VALUE", "COST"], 
                               index=pd.DatetimeIndex([self.asset.date], name="DATE")
                               )

        return history

    def update_history(self):
        """updates the positions history to reflect changes

        Updates this positions history whenever changes in the position occur, either in the size of the position itself or the price of the underlying asset

        Returns:
            updates the history inplace, no return value
        """
        self.history.loc[self.asset.date] = [self.value, self.cost]
    



class Portfolio:
    """An object representing a portfolio of financial assets

    AlphaGradient portfolios are designed to interact natively with AlphaGradient assets, and provide all of the functionality one would expect with a normal financial portfolio. 

    Attributes:
        cash (float): How much of the base currency the Portfolio holds
        date (datetime): the last time this position was altered or
            valuated
        positions (dict): A dictionary of all of this Portfolio's
            current positions
        longs (dict): A dictionary of all long positions
        shorts (dict): A dictionary of all short positions

    """
    def __init__(self, initial, date=None):
        self.cash = initial
        self.date = date if isinstance(date, datetime) else datetime.today()
        self.positions = {}
        self.history = self._history()

    def __str__(self):
         return f"<AlphaGradient Portfolio at {hex(id(self))}>"

    def __repr__(self):
        return self.__str__()

    @property
    def longs(self):
        """A dictionary which associates all long positions with their asset keys"""
        return {v.asset.key:v for k, v in self.positions.items() if not v.short}

    @property
    def shorts(self):
        """A dictionary which assocaites all short positions with their asset keys"""
        return {v.asset.key:v for k, v in self.positions.items() if v.short}

    @property
    def value(self):
        """This portfolio's net market value"""
        return self.cash + sum([position.value for position in self.positions.values()])

    def buy(self, asset, quantity):
        """Buys an asset using this portfolio

        Creates a long position in the given asset with a purchase volume given by 'quantity'.

        Args:
            asset (Asset): The asset in which to create a long position
            quantity (Number): The purchase quantity

        Returns:
            Modifies this portfolio's positions inplace. No return value
        """

        # Creating the position to be entered into
        position = Position(asset, quantity)

        if position.value > self.cash:
            raise ValueError(f"Purchasing {quantity} units of {asset.name} {asset.type} requires ${position.value}, but this portfolio only has ${self.cash} in reserve")

        # Updating the position if one already exists
        try:
            self.positions[position.key] += position
        except KeyError as e:
            self.positions[position.key] = position

        # Updating the potfolio's cash reserves
        self.cash -= position.value

        # Updating the portfolio's history to reflect purchase
        self.update_history()

    def sell(self, asset, quantity):
        """Sells a long position in this portfolio

        Decrements a long position in the given asset by 'quantity'. Maximum sale quantity is the amount owned by the portfolio.

        Args:
            asset (Asset): The asset of the corresponding decremented
                position
            quantity (Number): The sale quantity

        Returns:
            Modifies this portfolio's positions inplace. No return value
        """

        # Creating the position to be sold
        position = Position(asset, quantity)

        # Selling the position if the current position is large enough to satisfy the sale quantity
        try:
            current = self.positions[position.key]
            if current.quantity >= quantity:
                self.cash += position.value
                self.positions[position.key] -= position
            else:
                raise ValueError(f"Portfolio has insufficient long position in asset {asset.name} {asset.type} to sell {quantity} units. Only {current} units available")

        # We can only sell positions that we own
        except KeyError as e:
            raise ValueError(f"Portfolio has no long position in asset {asset.name} {asset.type}")

        # Updating the portfolio's history to reflect sale
        self.update_history()

    def short(self, asset, quantity):
        """Shorts an asset using this portfolio

        Creates a short position in the given asset with a short volume given by 'quantity'.

        Args:
            asset (Asset): The asset in which to create a short position
            quantity (Number): The short sale quantity

        Returns:
            Modifies this portfolio's positions inplace. No return value
        """

        # Creating the position to be shorted
        position = Position(asset, quantity, short=True)

        # Updating the position if one already exists
        try:
            self.positions[position.key] += position
        except KeyError as e:
            self.positions[position.key] = position

        # Updating the potfolio's cash reserves
        self.cash -= position.value

        # Updating the portfolio's history to reflect short sale
        self.update_history()

    def cover(self, asset, quantity):
        """Covers a short position in this portfolio

        Creates a short position in the given asset with a short volume given by 'quantity'.

        Args:
            asset (Asset): The asset in which to create a short position
            quantity (Number): The short sale quantity

        Returns:
            Modifies this portfolio's positions inplace. No return value
        """

        # Creating the short position to be covered
        position = Position(asset, quantity, short=True)

        required = -1 * position.value
        if required > self.cash:
            raise ValueError(f"Covering {quantity} short sold units of {asset.name} {asset.type} requires ${required}, but this portfolio only has ${self.cash} in reserve")

        # Covering the position if the current short position is large enough to satisfy the quantity to cover
        try:
            current = self.positions[position.key]
            if current.quantity >= quantity:
                self.cash += position.value
                self.positions[position.key] -= position
            else:
                raise ValueError(f"Portfolio has insufficient short position in asset {asset.name} {asset.type} to cover {quantity} units. Only {current.quantity} units have been sold short")

        # We can only cover positions that we have shorts in
        except KeyError as e:
            raise ValueError(f"Portfolio has no short position in asset {asset.name} {asset.type}")

        # Updating the portfolio's history to reflect short cover
        self.update_history()

    def _history(self):
        """A history of this portfolio's positions and value

        Returns a datetime indexed pandas dataframe of this portfolio's positions and total market value. This function is only used to initialize it

        Returns:
            history (pd.DataFrame): portfolio position/value history
        """
        positions = [position.view() for position in self.positions.values()]
        return pd.DataFrame(
                            [[positions, self.value]],
                            columns=["POSITIONS", "VALUE"], 
                            index=pd.DatetimeIndex([self.date], name="DATE")
                            )

    def update_history(self):
        """updates this portfolio's history

        Returns:
            Modifies history in place. Returns nothing
        """
        positions = [position.view() for position in self.positions.values()]
        self.history.loc[self.date] = np.array([positions, self.value], dtype="object")
        
