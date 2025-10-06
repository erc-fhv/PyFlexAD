from dataclasses import dataclass
from typing import Self

import numpy as np

from pyflexad.parameters.general_parameters import GeneralParameters
from pyflexad.parameters.hardware_parameters import HardwareParameters
from pyflexad.parameters.usage_parameters import UsageParameters
from pyflexad.physical.energy_storage import EnergyStorage


@dataclass(kw_only=True)
class EVUsage(UsageParameters):
    """
    d : int
        The number of time intervals.
    dt : float
        The time interval resolution in hours.
    initial_capacity: float
        Initial capacity of the battery in kWh
    final_capacity: float
        Final capacity of the battery in kWh
    demand: np.ndarray
        The inflexible power demand profile for each time interval in kW.
    availability: np.ndarray
        The availability for each time interval (0 for not available, 1 for available).
    """
    d: int
    dt: float
    initial_capacity: float
    final_capacity: float
    demand: np.ndarray
    availability: np.ndarray


@dataclass(kw_only=True)
class EVHardware(HardwareParameters):
    """
    Parameters
    ----------
    name: str
        Name of the ev model
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


class ElectricVehicle(EnergyStorage):
    """
    A Electric Vehicle (EV)  can store energy and release the stored energy on demand,
    allowing operational flexibility, with limited availability and additional trip consumption.
    """

    @classmethod
    def new(cls, hardware: EVHardware, usage: EVUsage, id: str = None) -> Self:
        """
        Create a new instance of the class using the provided hardware and usage parameters.

        Parameters
        ----------
        hardware : EVHardware
            The hardware parameters for the electric vehicle.
        usage : EVUsage
            The usage parameters for the electric vehicle.
        id : str, optional
            The unique identifier for the electric vehicle. If not provided, a new ID will be generated.

        Returns
        -------
        ElectricVehicle
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
            raise ValueError("final_capacity < min_capacity")
        elif usage.final_capacity > hardware.max_capacity:
            raise ValueError("final_capacity > max_capacity")

        """check demand for availability"""
        available_indices = usage.availability > 0.0
        if usage.demand[available_indices].sum() > 0:
            raise ValueError("Demand must be zero when availability is non-zero.")

        """scalar parameters"""
        s_initial = usage.initial_capacity
        alpha = hardware.self_discharge_factor

        """auxiliary variables"""
        v = np.power(alpha, np.arange(1, usage.d + 1))
        w = np.cumsum(v * usage.demand) * usage.dt

        """vector parameters"""
        x_upper = hardware.max_charging_power * np.ones(usage.d) * usage.availability
        x_lower = hardware.max_discharging_power * np.ones(usage.d) * usage.availability
        s_upper = w + hardware.max_capacity * np.ones(usage.d)
        s_lower = np.append(w[0:-1] + hardware.min_capacity * np.ones(usage.d - 1),
                            w[-1] + usage.final_capacity)

        gp = GeneralParameters(x_lower=x_lower, x_upper=x_upper,
                               s_lower=s_lower, s_upper=s_upper, s_initial=s_initial,
                               alpha=alpha, d=usage.d, dt=usage.dt)

        return cls(id=id, gp=gp, load_profile=load_profile)

    @classmethod
    def with_charging(cls, hardware: EVHardware,
                      initial_capacity: float,
                      availability: np.ndarray,
                      power_demand: np.ndarray,
                      charging: np.ndarray,
                      d: int, dt: float) -> Self:
        """
        Perform calculations based on the provided hardware,
        initial capacity, availability, power demand, charging data, time intervals, and time interval resolution.

        Parameters
        ----------
        hardware : EVHardware
            The hardware parameters for the electric vehicle.
        initial_capacity : float
            The initial capacity of the battery in kWh.
        availability : np.ndarray
            The availability for each time interval (0 for not available, 1 for available).
        power_demand : np.ndarray
            The inflexible power demand profile for each time interval in kW.
        charging : np.ndarray
            The charging data for each time interval.
        d : int
            The number of time intervals.
        dt : float
            The time interval resolution in hours.

        Returns
        -------
        ElectricVehicle
            A new instance of the class with updated usage parameters.

        Raises
        ------
        ValueError
            If there are inconsistencies in demand and availability or if the final capacity exceeds hardware limits.
        """

        if power_demand[availability > 0].sum() > 0:
            """check demand for availability"""
            raise ValueError("Demand must be zero when availability is non-zero.")

        if charging[availability < 1].sum() > 0:
            """check charging for availability"""
            raise ValueError("Charging must be zero when availability is zero.")

        c_rate = hardware.max_charging_power / hardware.max_capacity

        charging_power = charging * hardware.max_charging_power / c_rate / 100
        # Todo: fix workaround with very low charging rate

        p_s = sum([(charging_power[t] - power_demand[t]) * hardware.self_discharge_factor ** (d - (t + 1))
                   for t in range(d)])
        p_aux = sum([-power_demand[t] * hardware.self_discharge_factor ** (d - (t + 1)) for t in range(d)])

        final_capacity = hardware.self_discharge_factor ** d * initial_capacity + max(p_s * dt, p_aux * dt)

        if final_capacity < hardware.min_capacity:
            final_capacity = hardware.min_capacity
        elif final_capacity > hardware.max_capacity:
            final_capacity = hardware.max_capacity

        usage = EVUsage(dt=dt,
                        d=d,
                        initial_capacity=initial_capacity,
                        final_capacity=final_capacity,
                        demand=power_demand,
                        availability=availability,
                        )

        return cls.new(hardware=hardware, usage=usage)

    @classmethod
    def random_usage(cls, hardware: EVHardware, d: int, dt: float, availability: np.ndarray = None,
                     demand: np.ndarray = None) -> Self:
        """
        Create a random usage for an electric vehicle.

        Parameters
        ----------
        hardware : EVHardware
            The hardware parameters for the electric vehicle.
        d : int
            The number of time intervals.
        dt : float
            The time interval resolution in hours.
        availability : np.ndarray, optional
            The availability for each time interval (0 for not available, 1 for available).
            If not provided, a random availability will be created.
        demand : np.ndarray, optional
            The inflexible power demand profile for each time interval in kW.
            If not provided, a random demand will be created.

        Returns
        -------
        ElectricVehicle
            A new instance of the class with updated usage parameters.

        Raises
        ------
        ValueError
            If there are inconsistencies in demand and availability or if the final capacity exceeds hardware limits.
        """
        initial_capacity = np.random.uniform(0.5 * hardware.max_capacity, hardware.max_capacity)
        final_capacity = hardware.min_capacity

        if demand is None:
            """create a random demand"""
            demand = np.random.uniform(0.0, hardware.max_capacity / d, size=d)

        if availability is None:
            """create a random availability"""
            availability = np.random.choice([0, 1], size=d)

            """set a minimum availability"""
            if availability.sum() < 2:
                availability[0] = 1
                availability[-1] = 1

            """fix the demand for available time periods"""
            demand = np.where(availability == 0, demand, 0.0)

        """check demand for availability"""
        if demand[availability > 0].sum() > 0:
            raise ValueError("Demand must be zero when availability is non-zero.")

        if final_capacity < hardware.min_capacity:
            final_capacity = hardware.min_capacity
        elif final_capacity > hardware.max_capacity:
            final_capacity = hardware.max_capacity

        usage = EVUsage(d=d, dt=dt,
                        initial_capacity=initial_capacity,
                        final_capacity=final_capacity,
                        demand=demand,
                        availability=availability,
                        )

        return cls.new(hardware=hardware, usage=usage)
