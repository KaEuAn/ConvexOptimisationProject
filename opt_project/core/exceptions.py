# here custom exceptions will be defined

# exceptions for methods

class InitialPositionError(Exception):
    pass

class GessianMatrixReversibilityError(Exception):
    pass

class ConditionsError(Exception):
    pass