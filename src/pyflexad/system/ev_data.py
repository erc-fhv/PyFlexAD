import os
import warnings
from typing import Self

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class EVData:

    def __init__(self, power_demand: np.ndarray, availability: np.ndarray, charging_status: np.ndarray) -> None:
        """

        Parameters
        ----------
        power_demand: np.ndarray
            EV trip consumption per time interval in kW
        availability: np.ndarray
            EV availability per time interval (0 = not available, 1 = available)
        charging_status: np.ndarray
            EV charging status per time interval (0 = not charging, 1 = charging)
        """
        self.__power_demand = power_demand
        self.__availability = availability
        self.__charging_status = charging_status

    @property
    def power_demand(self) -> np.ndarray:
        return self.__power_demand

    @property
    def availability(self) -> np.ndarray:
        return self.__availability

    @property
    def charging(self) -> np.ndarray:
        return self.__charging_status

    @classmethod
    def from_file(cls, path_ev: str, n_entities: int, n_time_periods: int, dt: float, selection: str = "random",
                  dates: pd.DatetimeIndex = None) -> Self:
        """
        Creates an instance of the class from data stored in files.

        Parameters
        ----------
        path_ev : str
            Path to the directory containing the EV data files.
        n_entities : int
            Number of entities to select from the available participants.
        n_time_periods : int
            Number of time periods for which data is available.
        dt : float
            Time interval between data points in hours.
        selection : str, optional
            Method to select participants. Default is "random".
            - "ordered": selects participants in order.
            - "random": randomly selects participants.
            - list[int]: selects participants from a list of indices.
        dates : pd.DatetimeIndex, optional
            Specific dates for which data is available. Default is None.

        Returns
        -------
        EVData
            An instance of the class with the selected EV data.

        Raises
        ------
        ValueError
            If dt is not 0.25 h.
            If n_entities is higher than the number of available participants.
            If Max selection is greater than or equal to the number of participants.
            If Dates length does not match n_time_periods.
            If Dates type is not pd.DatetimeIndex.
        NotImplementedError
            If selection is not "ordered", "random", or a list of indices.
        """

        if not np.isclose(dt, 0.25):
            raise ValueError(f"dt must be 0.25 h for EV data, got {dt:.2f} instead")

        participants_file = os.path.join(path_ev, "participant_IDs.pickle")
        participants = pd.read_pickle(participants_file)
        n_participants = len(participants)

        if selection == "ordered":
            """ordered participants choice"""
            if n_entities > n_participants:
                raise ValueError(f"Number of entities is higher than the number of available participants! "
                                 f"{n_entities} > {n_participants}")

            selection = np.arange(n_entities)

        elif selection == "random":
            """random participants choice"""

            if n_entities > n_participants:
                warnings.warn(
                    f"Number of entities is higher than "
                    f"the number of available participants! {n_entities} > {n_participants} "
                    "same participants will be used multiple times")

            selection = np.random.choice(n_participants, n_entities, replace=True)
        elif isinstance(selection, list):
            """selection from a list of participant indices"""
            if np.max(selection) >= n_participants:
                raise ValueError(f"Max selection {np.max(selection)} >= len(participants) {n_participants}")

            selection = np.array(selection)
        else:
            raise NotImplementedError(f"Selection {selection} not implemented")

        """get the demand of the chosen participants"""
        power_demand_list = []
        availability_list = []
        charging_list = []

        for k in selection:
            file = os.path.join(path_ev, f"ev_df_{participants[k]}.pickle")
            df = pd.read_pickle(file)

            if dates is None:
                demand_df = df[["loadable_bool", "power_kW", "charging_bool"]]
            elif isinstance(dates, pd.DatetimeIndex):

                if len(dates) != n_time_periods:
                    raise ValueError(f"Dates length {len(dates)} does not match n_time_periods {n_time_periods}")

                demand_df = df.loc[dates][["loadable_bool", "power_kW", "charging_bool"]]
            else:
                raise NotImplementedError(f"Dates type {type(dates)} not implemented")

            d = demand_df.values[:n_time_periods].astype(float)

            availability_list.append(d[:, 0])
            power_demand_list.append(d[:, 1])
            charging_list.append(d[:, 2])

        power_demand = np.array(power_demand_list)
        availability = np.array(availability_list)
        charging = np.array(charging_list)

        return cls(power_demand, availability, charging)

    @classmethod
    def from_random(cls, max_power_consumption: float, n_entities: int, n_time_periods: int) -> Self:
        """
        Creates random EV data based on specified parameters.

        Parameters
        ----------
        max_power_consumption : float
            Maximum power consumption for random data generation.
        n_entities : int
            Number of entities for which data is generated.
        n_time_periods : int
            Number of time periods for which data is generated.

        Returns
        -------
        EVData
            An instance of the class with randomly generated EV data.
        """
        availability = np.random.random_integers(low=0, high=1, size=(n_entities, n_time_periods))

        """set a minimum availability"""
        if availability.sum() < 2:
            availability[0] = 1
            availability[-1] = 1

        power_demand = np.random.uniform(low=0, high=max_power_consumption, size=(n_entities, n_time_periods))
        charging = np.zeros_like(availability)

        power_demand = np.where(availability == 0, power_demand, 0.0)
        charging = np.where(availability == 1, charging, 0.0)

        return cls(power_demand, availability, charging)

    def plot_power_demand(self, ax: plt.axes = None, method: str = "stack", use_legend: bool = True) -> None:
        """
        Plots the power demand data based on the specified method.

        Parameters
        ----------
        ax : plt.axes, optional
            The axes on which to plot the data. If None, a new figure and axes will be created.
        method : str, optional
            The method to use for plotting the data. Default is "stack".
        use_legend : bool, optional
            Whether to display the legend on the plot. Default is True.

        Returns
        -------
        None
        """
        fig, ax = plt.subplots() if ax is None else ax
        intervals = np.arange(self.power_demand.shape[1])

        if method == "stack":
            labels = [f"Participant[{i}]" for i in range(self.power_demand.shape[0])]
            ax.stackplot(intervals, self.power_demand, labels=labels)
        elif method == "line":
            for i in range(self.power_demand.shape[0]):
                ax.plot(intervals, self.power_demand[i, :], label=f"Participant[{i}]")
        elif method == "total":
            ax.plot(intervals, self.power_demand.sum(axis=0), label=f"Total Demand")
        elif method == "fill":
            ax.fill_between(intervals, np.zeros_like(intervals), self.power_demand.sum(axis=0), label=f"Total Demand")
        else:
            raise NotImplementedError(f"Method {method} not implemented")

        ax.set_xlabel("Time Interval")
        ax.set_ylabel("Power Demand / kW")
        if use_legend:
            ax.legend()
        ax.grid(True)
