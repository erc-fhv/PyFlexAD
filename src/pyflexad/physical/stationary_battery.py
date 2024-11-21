from dataclasses import dataclass
from typing import Self

import numpy as np

from pyflexad.physical.energy_storage import EnergyStorage
from pyflexad.parameters.general_parameters import GeneralParameters
from pyflexad.parameters.hardware_parameters import HardwareParameters
from pyflexad.parameters.usage_parameters import UsageParameters


@dataclass(kw_only=True, frozen=True)
class BESSUsage(UsageParameters):
    """
    Parameters
    ----------
    d : int
        The number of time intervals.
    dt : float
        The time interval resolution in hours.
    initial_capacity: float
        Initial capacity of the battery in kWh.
    final_capacity: float
        Final capacity of the battery in kWh.
    """
    d: int
    dt: float
    initial_capacity: float
    final_capacity: float


@dataclass(kw_only=True)
class BESSHardware(HardwareParameters):
    """
    Parameters
    ----------
    name: str
        Name of the battery model
    max_discharging_power: float
        Maximum discharging power in kW
    max_charging_power: float
        Maximum charging power in kW
    max_capacity: float
        Maximum capacity in kWh
    min_capacity: float
        Minimum capacity in kWh
    self_discharge_factor: float
        Self-discharge factor
    """
    name: str
    max_discharging_power: float
    max_charging_power: float
    max_capacity: float
    min_capacity: float
    self_discharge_factor: float


class StationaryBattery(EnergyStorage):
    """
    A BESS, e.g. a stationary battery, is a device that can store energy and release the stored energy on demand,
    allowing operational flexibility.
    """

    @classmethod
    def new(cls, hardware: BESSHardware, usage: BESSUsage, id: str = None) -> Self:
        """
        Create a new instance of the class using the provided hardware and usage parameters.

        Parameters
        ----------
        hardware : BESSHardware
            The hardware parameters for the battery.
        usage : BESSUsage
            The usage parameters for the battery.
        id : str, optional
            The unique identifier for the battery. If not provided, a new ID will be generated.

        Returns
        -------
        StationaryBattery
            The newly created instance of the class.

        Raises
        ------
        ValueError
            If the final capacity is less than the minimum capacity or greater than the maximum capacity.

        """
        if id is None:
            id = cls.create_id()

        load_profile = cls.empty_load_profile(usage.d)

        if usage.final_capacity < hardware.min_capacity:
            raise ValueError(f"final_capacity < min_capacity")
        elif usage.final_capacity > hardware.max_capacity:
            raise ValueError(f"final_capacity > max_capacity")

        """scalar parameters"""
        s_initial = usage.initial_capacity
        alpha = hardware.self_discharge_factor

        """vector parameters"""
        x_upper = hardware.max_charging_power * np.ones(usage.d)
        x_lower = hardware.max_discharging_power * np.ones(usage.d)
        s_upper = hardware.max_capacity * np.ones(usage.d)
        s_lower = np.append(hardware.min_capacity * np.ones(usage.d - 1), usage.final_capacity)

        gp = GeneralParameters(x_lower=x_lower, x_upper=x_upper,
                               s_lower=s_lower, s_upper=s_upper, s_initial=s_initial,
                               alpha=alpha, d=usage.d, dt=usage.dt)
        return cls(id=id, gp=gp, load_profile=load_profile)

    @classmethod
    def random_usage(cls, hardware: BESSHardware, d: int, dt: float, id: str = None) -> Self:
        """
        Create a random usage for a stationary battery.

        Parameters
        ----------
        hardware : BESSHardware
            The hardware parameters for the battery.
        d : int
            The number of time intervals.
        dt : float
            The time interval resolution in hours.
        id : str, optional
            The unique identifier for the battery. If not provided, a new ID will be generated.

        Returns
        -------
        StationaryBattery
            The newly created instance of the class.

        """
        initial_capacity = np.random.uniform(0.5 * hardware.max_capacity, hardware.max_capacity)
        final_capacity = 1 / 2 * initial_capacity

        usage = BESSUsage(initial_capacity=initial_capacity, final_capacity=final_capacity, d=d, dt=dt)

        return cls.new(hardware=hardware, usage=usage, id=id)
