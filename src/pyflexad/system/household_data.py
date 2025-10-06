import os
import warnings
from typing import Self

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class HouseholdsData:

    @classmethod
    def from_file(cls, path_hh: str, n_entities: int, n_time_periods: int,
                  selection: str | list[int] = "random", dates: pd.DatetimeIndex = None) -> Self:
        """
        Generate a HouseholdsData object from a file.

        Parameters
        ----------
        path_hh: str
            The path to the directory containing the household data files.
        n_entities :int
            The number of entities to be selected.
        n_time_periods: int
            The number of time periods to consider.
        selection: str | list[int]
            The selection method to use. Defaults to "random".
                - "random": Randomly select entities from the available participants.
                - list[int]: Select entities from a list of participant indices.
        dates: pd.DatetimeIndex
            The dates to consider. Defaults to None.

        Returns
        -------
        Self: The generated HouseholdsData object.

        Raises
        -----
        ValueError: If the maximum selection is greater than or equal to the number of participants.
        NotImplementedError: If the selection method is not implemented.
        ValueError: If the dates length does not match n_time_periods.
        NotImplementedError: If the dates type is not implemented.
        """
        participants_file = os.path.join(path_hh, "residential_IDs_file1.pickle")
        participants = pd.read_pickle(participants_file)

        if selection == "random":
            """random participants choice"""
            n_participants = len(participants)
            if n_entities > n_participants:
                warnings.warn(
                    f"Number of entities is higher than "
                    f"the number of available participants! {n_entities} > {n_participants} "
                    "same participants will be used multiple times")

            selection = np.random.choice(n_participants, n_entities, replace=True)
        elif isinstance(selection, list):
            """selection from a list of participant indices"""
            if np.max(selection) >= len(participants):
                raise ValueError(f"Max selection {np.max(selection)} >= len(participants) {len(participants)}")

            selection = np.array(selection)
        else:
            raise NotImplementedError(f"Selection {selection} not implemented")

        power_demand_list = []
        """get the demand of the chosen participants"""
        for k in selection:
            file = os.path.join(path_hh, f"hh_df_{participants[k]}.pickle")
            df = pd.read_pickle(file)

            if dates is None:
                demand_df = df.copy()
            elif isinstance(dates, pd.DatetimeIndex):

                if len(dates) != n_time_periods:
                    raise ValueError(f"Dates length {len(dates)} does not match n_time_periods {n_time_periods}")

                demand_df = df.loc[dates]
            else:
                raise NotImplementedError(f"Dates type {type(dates)} not implemented")

            d = demand_df.values[:n_time_periods].flatten()
            power_demand_list.append(d)

        power_demand = np.array(power_demand_list)
        return cls(power_demand)

    @classmethod
    def from_random(cls, min_power_demand: float, max_power_demand: float, n_households: int,
                    n_time_periods: int, random_seed: int = None) -> Self:
        """
        Generates random power demand data for multiple households over multiple time periods.

        Parameters
        ----------
        min_power_demand: float
            The minimum power demand value.
        max_power_demand: float
            The maximum power demand value.
        n_households: int
            The number of households for which to generate power demand.
        n_time_periods: int
            The number of time periods for which to generate power demand.
        random_seed: int, optional
            The random seed for reproducibility. Defaults to None.

        Returns
        -------
        Self: An instance of the class with the generated power demand data.
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        power_demand = np.random.uniform(low=min_power_demand, high=max_power_demand,
                                         size=(n_households, n_time_periods))
        return cls(power_demand)

    def __init__(self, power_demand: np.ndarray) -> None:
        """
        Initialize the power demand data.

        Parameters
        ----------
        power_demand : np.ndarray
            The inflexible power demand data for each time interval.
        """
        self.__power_demand = power_demand

    @property
    def power_demand(self) -> np.ndarray:
        return self.__power_demand

    def plot_power_demand(self, ax: plt.axes = None, method: str = "stack", use_legend: bool = True):
        """
        Plot the power demand data.

        Parameters
        ----------
        ax : plt.axes, optional
            The axes to plot on. If None, a new figure and axes will be created.
            Defaults to None.
        method : str, optional
            The method to plot the power demand data.
            Must be one of "stack", "line", "total", or "fill".
            Defaults to "stack".
        use_legend : bool, optional
            Whether to include a legend in the plot.
            Defaults to True.

        Raises
        ------
        NotImplementedError
            If the specified method is not implemented.

        Returns
        -------
        None

        Notes
        -----
        This function plots the power demand data on the specified axes.
        The power demand data is represented as a 2D numpy array with shape (n_households, n_time_periods).
        The x-axis represents the time intervals, and the y-axis represents the power demand in kW.
        The method parameter determines how the power demand data is plotted.
        - If method is "stack", the power demand data is plotted as a stacked area plot.
        - If method is "line", the power demand data is plotted as individual lines for each household.
        - If method is "total", the total power demand is plotted as a single line.
        - If method is "fill", the total power demand is filled under the x-axis.
        The legend is included in the plot if use_legend is True.
        The plot is displayed with a grid.
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
            ax.plot(intervals, self.power_demand.sum(axis=0), label="Total Demand")
        elif method == "fill":
            ax.fill_between(intervals, np.zeros_like(intervals), self.power_demand.sum(axis=0), label="Total Demand")
        else:
            raise NotImplementedError(f"Method {method} not implemented")

        ax.set_xlabel("Time Interval")
        ax.set_ylabel("Power Demand / kW")
        if use_legend:
            ax.legend()
        ax.grid(True)
