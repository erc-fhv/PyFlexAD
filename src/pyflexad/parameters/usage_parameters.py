from abc import ABCMeta


class UsageParameters(metaclass=ABCMeta):
    """
    d : int
        The number of time intervals.
    dt : float
        The time interval resolution in hours.
    """
    d: int
    dt: float
