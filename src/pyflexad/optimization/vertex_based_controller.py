from abc import ABC
from abc import abstractmethod

import numpy as np

from pyflexad.utils.upr import UPR
from pyflexad.virtual.aggregator import Aggregator


class VertexBasedController(ABC, UPR):

    def __init__(self, power_demand: np.ndarray) -> None:
        """
        Initialize the VertexBasedController with the given power demand.

        Parameters
        ----------
        power_demand: np.ndarray
            Array representing the power demand.

        Returns
        -------
        None
        """
        self._d = power_demand.shape[1]
        self._power_demand = power_demand
        self._agg_power_demand = np.sum(self._power_demand, axis=0)

    def optimize(self, aggregator: Aggregator) -> np.ndarray:
        """
        Optimize the power distribution based on the given aggregator.

        Parameters
        ----------
        aggregator: Aggregator
            The aggregator object providing vertices for optimization.

        Returns
        -------
        np.ndarray
            The aggregated power distribution after optimization.
        """
        vertices = aggregator.get_vertices()
        aggregated_power, alphas = self.solve(vertices)
        aggregator.disaggregate(alphas)
        return aggregated_power

    @abstractmethod
    def solve(self, vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pass

    @abstractmethod
    def calc_upr(self, **kwargs) -> float:
        pass
