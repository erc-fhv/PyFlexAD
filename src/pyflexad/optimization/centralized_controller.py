from abc import ABC
from abc import abstractmethod

import numpy as np

from pyflexad.physical.energy_storage import EnergyStorage
from pyflexad.utils.upr import UPR


class CentralizedController(ABC, UPR):

    def __init__(self, power_demand: np.ndarray) -> None:
        """
        Initialize the CentralizedController with the given power demand.

        Parameters
        ----------
        power_demand: np.ndarray
            Array representing the power demand for each time interval.

        Returns
        -------
        None
        """
        self._d = power_demand.shape[1]
        self._power_demand = power_demand
        self._agg_power_demand = np.sum(self._power_demand, axis=0)

    def optimize(self, items: list[EnergyStorage], minimize: bool = True) -> np.ndarray:
        """
        Optimize the power distribution among the given EnergyStorage items.

        Parameters
        ----------
        items: list[EnergyStorage]
            List of EnergyStorage objects to optimize power distribution for.
        minimize: bool, optional
            If True, minimize the power distribution, else maximize. Defaults to True.

        Returns
        -------
        np.ndarray
            Aggregated power distribution after optimization.
        """
        individual_powers = self.solve(items=items, minimize=minimize)
        aggregated_power = np.sum(individual_powers, axis=1)
        return aggregated_power

    @abstractmethod
    def solve(self, items: list[EnergyStorage], minimize: bool = True) -> np.ndarray:
        """
        Perform optimization to distribute power among EnergyStorage items.

        Parameters
        ----------
        items: list[EnergyStorage]
            List of EnergyStorage objects to distribute power among.
        minimize: bool, optional
            If True, minimize power distribution; if False, maximize. Defaults to True.

        Returns
        -------
        np.ndarray
            Individual power distribution after optimization.
        """
        pass

    @staticmethod
    def constr_A_b(x, item: EnergyStorage, j: int):
        A, b = item.calc_A_b()
        return A @ x[:, j] <= b

    @abstractmethod
    def calc_upr(self, **kwargs) -> float:
        pass
