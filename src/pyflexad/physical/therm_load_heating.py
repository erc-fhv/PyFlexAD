from dataclasses import dataclass
from typing import Self

import numpy as np

from pyflexad.parameters.general_parameters import GeneralParameters
from pyflexad.parameters.hardware_parameters import HardwareParameters
from pyflexad.parameters.usage_parameters import UsageParameters
from pyflexad.physical.energy_storage import EnergyStorage


@dataclass(kw_only=True)
class TCLHUsage(UsageParameters):
    """
    d : int
        The number of time intervals.
    dt : float
        The time interval resolution in hours.
    theta_r: float
        Set point temperature in K
    theta_a: float
        Ambient temperature in K
    theta_0: float
        Initial temperature in K
    delta: float
        Temperature dead band in K
    demand: np.ndarray
        The inflexible power demand in kW
    """
    d: int
    dt: float
    theta_r: float
    theta_a: float
    theta_0: float
    delta: float
    demand: np.ndarray

    @classmethod
    def from_celsius(cls, d: int, dt: float,
                     theta_r_deg_c: float, theta_a_deg_c: float, theta_0_deg_c: float, delta: float,
                     demand: np.ndarray) -> Self:
        offset = 273.15
        return cls(d=d, dt=dt,
                   theta_r=theta_r_deg_c - offset, theta_a=theta_a_deg_c - offset,
                   theta_0=theta_0_deg_c - offset, delta=delta, demand=demand)


@dataclass(kw_only=True)
class TCLHHardware(HardwareParameters):
    """
    name: str
        The name of the hardware model.
    C: float
        The heat capacity of the hardware in kWh/K.
    R: float
        The thermal resistance of the hardware in K/W.
    p_max: float
        The maximum power of the hardware in kW.
    cop: float
        The coefficient of performance of the hardware.
    """
    name: str
    C: float
    R: float
    p: float
    cop: float


class ThermostaticLoadHeating(EnergyStorage):
    """
    TCLs, e.g. refrigerators, air conditioners, water heaters, heat pumps, etc.,
    are appliances designed to maintain a desired temperature range.
    The temperature dynamics in a TCL is usually described by Newton’s law of cooling together with an additional term
    for the power consumption.
    During operation, the temperature is expected to remain within a certain range around the set point temperature.
    However, the power supply can be shifted in time, which offers operational flexibility.
    """

    @classmethod
    def new(cls, hardware: TCLHHardware, usage: TCLHUsage, id: str = None) -> Self:
        """
        Create a new instance of the ThermostaticLoadHeating class.

        Parameters
        ----------
        hardware : TCLHHardware
            The hardware parameters of the thermostatic load heating.
        usage : TCLHUsage
            The usage parameters of the thermostatic load heating.
        id : str, optional
            The ID of the thermostatic load heating. If not provided, a new ID will be generated.

        Returns
        -------
        ThermostaticLoadHeating
            The newly created instance of the ThermostaticLoadHeating class.
        """
        if id is None:
            id = cls.create_id()

        load_profile = cls.empty_load_profile(usage.d)

        """auxiliary parameters"""
        x0 = - (usage.theta_a - usage.theta_r) / (hardware.cop * hardware.R)

        """scalar parameters"""
        alpha = np.exp(-usage.dt / (hardware.C * hardware.R))
        s_initial = hardware.C / hardware.cop * (usage.theta_0 - usage.theta_r)

        """auxiliary variables"""
        v = np.power(alpha, np.arange(1, usage.d + 1))
        w = np.cumsum(v * usage.demand) * usage.dt

        """vector parameters"""
        x_lower = - x0 * np.ones(usage.d)
        x_upper = (hardware.p - x0) * np.ones(usage.d)
        s_lower = (w - hardware.C * usage.delta) / hardware.cop * np.ones(usage.d)
        s_upper = (w + hardware.C * usage.delta) / hardware.cop * np.ones(usage.d)

        gp = GeneralParameters(x_lower=x_lower, x_upper=x_upper,
                               s_lower=s_lower, s_upper=s_upper, s_initial=s_initial,
                               alpha=alpha, d=usage.d, dt=usage.dt, x0=x0)
        return cls(id=id, gp=gp, load_profile=load_profile)
