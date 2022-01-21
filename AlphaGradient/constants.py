from aenum import Enum


# Determines whether a datatype is fundamentally numeric
def is_numeric(obj):
    try:
        obj + 1
        obj - 1
        obj * 1
        obj / 1
        float(obj)
        int(obj)
        try:
            obj[0]
            return False
        except BaseException:
            return True
    except BaseException:
        return False

# System for console logging when performing algorithms


class LoggingSystem:
    class Switch:
        def __init__(self, name, b=False):
            self.bool = b if isinstance(b, bool) else False
            self.name = name

        def SET(self, b):
            if isinstance(b, bool):
                self.bool = b

        def TRUE(self):
            self.bool = True

        def FALSE(self):
            self.bool = False

        def SWITCH(self):
            self.bool = not self.bool

        def PRINT(self, *args, **kwargs):
            if self.bool:
                print(*args, **kwargs)

        def __str__(self):
            return f'{self.name}: {self.bool}'

        def __repr__(self):
            return self.__str__()

    def __init__(self):
        self.BASIC = self.Switch('BASIC')
        self.ADVANCED = self.Switch('ADVANCED')
        self.DEBUG = self.Switch('DEBUG')
        self.EXCEPTIONS = self.Switch('EXCEPTIONS')
        self._SWITCHES = [
            self.BASIC,
            self.ADVANCED,
            self.DEBUG,
            self.EXCEPTIONS]

    @property
    def SWITCHES(self):
        return self._SWITCHES

    @property
    def ACTIVE(self):
        return [switch for switch in self._SWITCHES if switch.bool is True]

    def ANY(self, content, *args):
        conditions = [
            arg for arg in args] if args else [
            switch.name for switch in self._SWITCHES]
        if any(
                [condition == switch.name for switch in self.ACTIVE for condition in conditions]):
            print(content)

    def ALL(self, content, *args):
        condition = [arg for arg in args] if args else [
            switch.name for switch in self._SWITCHES]
        if all(
                [condition == switch.name for switch in self.ACTIVE for condition in conditions]):
            print(content)

    def PRINT(self, content, *args, mode="ANY"):
        if mode.upper() == "ANY":
            self.ANY(content, *args)
        elif mode.upper() == "ALL":
            self.ALL(content, *args)

    def SETALL(self, boolean=None):
        value = None
        if isinstance(boolean, str):
            if boolean.upper() == 'TRUE':
                value = True
            elif boolean.upper() == 'FALSE':
                value = False
            elif boolean.upper() == 'SWITCH':
                for switch in self._SWITCHES:
                    switch.SWITCH()
        elif isinstance(boolean, bool):
            value = boolean

        if value is not None:
            for switch in self._SWITCHES:
                switch.SET(value)

    def TRUE(self):
        for switch in self._SWITCHES:
            switch.TRUE()

    def FALSE(self):
        for switch in self._SWITCHES:
            switch.FALSE()

    def __str__(self):
        result = f'LOGGING CONFIG: ' '\n---------------\n'
        for switch in self._SWITCHES:
            spaces = ' ' * (11 - len(switch.name))
            result += f'    {switch.name}:{spaces}{switch.bool}' + '\n'
        return result[:-1]

    def __repr__(self):
        return "<AlphaGradient Logging System>"


# Logging System
VERBOSE = LoggingSystem()

try:
    NoneType
except NameError:
    NoneType = type(None)


# Package-wide Constants
class CONSTANTS(Enum):

    # Package Information
    VERSION = 'v1.0.0'
    AUTHOR = 'Nathan Heidacker'

    # Numerical Constants
    INDEX_ANNUAL_RETURN = 0.1
    TBILL_YIELD_1YR = 0.0007
