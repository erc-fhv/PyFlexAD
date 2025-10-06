from dataclasses import dataclass
from typing import Self

import numpy as np

from pyflexad.parameters.general_parameters import GeneralParameters
from pyflexad.parameters.hardware_parameters import HardwareParameters
from pyflexad.parameters.usage_parameters import UsageParameters
from pyflexad.physical.energy_storage import EnergyStorage


@dataclass(kw_only=True)
class PHESUsage(UsageParameters):
    """
    d : int
        The number of time intervals.
    dt : float
        The time interval resolution in hours.
    initial_volume: float
        Initial volume of the upper hydro storage reservoir in m^3
    final_volume: float
        Final volume of the upper hydro storage reservoir in m^3
    """
    dt: float
    d: int
    initial_volume: float
    final_volume: float


@dataclass(kw_only=True)
class PHESHardware(HardwareParameters):
    """
    name: str
        Name of the hydro storage
    max_discharging_power: float
        Maximum discharging power in kW
    max_charging_power: float
        Maximum charging power in kW
    max_volume: float
        Maximum volume of the upper hydro storage reservoir in m^3
    min_volume: float
        Minimum volume of the upper hydro storage reservoir in m^3
    delta_h: float
        The height difference between upper and lower reservoir in m
    """
    name: str
    max_discharging_power: float
    max_charging_power: float
    max_volume: float
    min_volume: float
    delta_h: float


class PumpedHydroEnergyStorage(EnergyStorage):
    """
    A pumped hydro energy storage (PHES) pumps water from a lower reservoir to an upper reservoir,
    which increases the potential energy of the water.
    This energy can then be released on demand to generate electricity.
    """

    @classmethod
    def new(cls, hardware: PHESHardware, usage: PHESUsage, id: str = None) -> Self:
        """
        Create a new instance of the PumpedHydroEnergyStorage class with the provided hardware and usage parameters.

        Parameters
        ----------
        hardware: PHESHardware
            The hardware parameters for the energy storage.
        usage: PHESUsage
            The usage parameters for the energy storage.
        id: str, optional
            The unique identifier for the energy storage. If not provided, a new one will be generated.

        Returns
        -------
        PumpedHydroEnergyStorage
            A new instance of the PumpedHydroEnergyStorage class.
        """
        if id is None:
            id = cls.create_id()

        load_profile = cls.empty_load_profile(usage.d)

        """auxiliary variables"""
        g = 9.81  # gravitational acceleration m/s^2
        rho = 1000  # density water kg/m^3
        eta_star = 3.6 * 10 ** 9 / (rho * g * hardware.delta_h)

        """scalar parameters"""
        s_initial = usage.initial_volume / eta_star
        alpha = 1

        """vector parameters"""
        x_upper = hardware.max_charging_power * np.ones(usage.d)
        x_lower = hardware.max_discharging_power * np.ones(usage.d)
        s_upper = hardware.max_volume / eta_star * np.ones(usage.d)
        s_lower = np.append(hardware.min_volume * np.ones(usage.d - 1), usage.final_volume) / eta_star

        gp = GeneralParameters(x_lower=x_lower, x_upper=x_upper,
                               s_lower=s_lower, s_upper=s_upper, s_initial=s_initial,
                               alpha=alpha, d=usage.d, dt=usage.dt)

        return cls(id=id, gp=gp, load_profile=load_profile)
