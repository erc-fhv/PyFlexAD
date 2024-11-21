from dataclasses import dataclass

import numpy as np


@dataclass(kw_only=True)
class GeneralParameters:
    """
    x_lower: np.ndarray
        The lower power bound for each time interval in kWh.
    x_upper: np.ndarray
        The upper power bound for each time interval in kWh.
    s_lower: np.ndarray
        The lower energy bound for each time interval in kWh.
    s_upper: np.ndarray
        The upper energy bound for each time interval in kWh.
    s_initial: float
        The initial energy in kWh.
    alpha: float
        The self discharge factor.
    d: int
        The number of time intervals.
    dt: float
        The time interval resolution in hours.
    x0: float
        The power offset in kW (default = 0.0).
    """
    x_lower: np.ndarray
    x_upper: np.ndarray
    s_lower: np.ndarray
    s_upper: np.ndarray
    s_initial: float
    alpha: float
    d: int
    dt: float
    x0: float = 0.0
