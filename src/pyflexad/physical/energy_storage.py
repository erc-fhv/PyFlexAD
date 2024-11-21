import collections
import uuid
from abc import ABC
from typing import Self

import matplotlib.pyplot as plt
import numpy as np

from pyflexad.math.iabvg import IABVG
from pyflexad.math.iabvg_jit import IABVG_JIT
from pyflexad.math.lpvg_gurobipy import LPVGGurobipy
from pyflexad.math.lpvg_pyomo import LPVGPyomo
from pyflexad.math.polytope import Polytope
from pyflexad.math.signal_vectors import SignalVectors
from pyflexad.parameters.general_parameters import GeneralParameters
from pyflexad.parameters.hardware_parameters import HardwareParameters
from pyflexad.parameters.usage_parameters import UsageParameters
from pyflexad.utils.algorithms import Algorithms
from pyflexad.virtual.virtual_energy_storage import VirtualEnergyStorage


class EnergyStorage(ABC):

    @staticmethod
    def create_id() -> str:
        """
        Generate a unique identifier string using the UUID version 4 format.

        Returns
        -------
        str: A unique identifier string.
        """
        return str(uuid.uuid4())

    @staticmethod
    def empty_load_profile(d: int) -> np.ndarray:
        """
        Generate an array filled with NaN values of size d.

        Parameters
        ----------
            
        d: int
            The size of the array.

        Returns
        -------
        np.ndarray: An array filled with NaN values.
        """
        return np.full(d, np.nan)

    @classmethod
    def new(cls, hardware: HardwareParameters, usage: UsageParameters) -> Self:
        """
        Create a new instance of the EnergyStorage class with specific hardware parameters and usage parameters.

        Parameters
        ----------
        hardware: HardwareParameters 
            The specific parameters related to the hardware of the energy storage.
        usage: UsageParameters
            The parameters related to the usage of the energy storage.

        Returns
        -------
        EnergyStorage: A new instance of the EnergyStorage class.
        """
        raise NotImplementedError

    def __init__(self, id: str, gp: GeneralParameters, load_profile: np.ndarray) -> None:
        """

        Parameters
        ----------
        id: str
            The unique identifier of the energy storage.
        gp : GeneralParameters
            The general parameters of the energy storage.
        load_profile : np.ndarray
            The load profile of the energy storage.
        """
        self.__id = id
        self.__load_profile = load_profile
        self.__gp = gp

    def get_id(self) -> str:
        return self.__id

    def get_load_profile(self) -> np.ndarray:
        return self.__load_profile

    def set_load_profile(self, load_profile: np.ndarray) -> None:
        self.__load_profile = load_profile

    def to_virtual(self, algorithm: Algorithms, signal_vectors: SignalVectors = None) -> VirtualEnergyStorage:
        """
        Converts the energy storage to a virtual representation based on the specified algorithm and signal vectors.

        Parameters
        ----------
        algorithm: Algorithms
            The algorithm to use for the conversion.
        signal_vectors: SignalVectors, optional
            The signal vectors used in the conversion process. Defaults to None.

        Returns
        -------
        VirtualEnergyStorage
            A virtual representation of the energy storage.

        Raises
        ------
        ValueError
            If the specified algorithm is not supported.
        """
        if algorithm == Algorithms.EXACT:
            vertices = Polytope.from_A_b(*self.calc_A_b()).V_sorted()
        elif algorithm == Algorithms.LPVG_GUROBIPY:
            vertices = (LPVGGurobipy.from_general_params(self.__gp)
                        .approx_vertices(*self.calc_A_b(), signal_vectors=signal_vectors))
        elif algorithm in Algorithms.get_lpvg_pyomo_algorithms():
            vertices = (LPVGPyomo.from_general_params(self.__gp, solver=Algorithms.get_solver(algorithm))
                        .approx_vertices(*self.calc_A_b(), signal_vectors=signal_vectors))
        elif algorithm == Algorithms.IABVG:
            vertices = IABVG.from_general_params(self.__gp).approx_vertices(signal_vectors)
        elif algorithm == Algorithms.IABVG_JIT:
            vertices = IABVG_JIT.from_general_params(self.__gp).approx_vertices(signal_vectors)
        else:
            raise ValueError(f"algorithm {algorithm} is not supported")

        return VirtualEnergyStorage(id=self.__id, vertices=vertices, load_profile=self.__load_profile)

    def calc_A_b(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculation of matrix A and vector b for the energy storage model
        """

        """calculate A matrix"""
        eye = np.identity(self.__gp.d)
        v = collections.deque([self.__gp.alpha ** i for i in range(self.__gp.d)])
        v_list = [list(v)]

        for i in range(self.__gp.d - 1):
            v.rotate(1)
            v_list.append(list(v))

        gamma = np.tril(np.array(v_list).T, 0)
        A = np.concatenate((-eye, eye, gamma, -gamma), axis=0)

        """calculate b vector"""
        v = np.array([self.__gp.alpha ** i for i in range(1, self.__gp.d + 1)])
        b_1 = -self.__gp.x_lower
        b_2 = self.__gp.x_upper
        b_3 = 1 / self.__gp.dt * (self.__gp.s_upper - self.__gp.s_initial * np.ones(self.__gp.d) * v)
        b_4 = 1 / self.__gp.dt * (self.__gp.s_initial * np.ones(self.__gp.d) * v - self.__gp.s_lower)
        b = np.concatenate((b_1, b_2, b_3, b_4))

        b = b + A @ (np.ones(self.__gp.d) * self.__gp.x0)

        return A, b

    def plot_load_profile_2d(self, ax: plt.axes = None, label: str = "", color: str = None, marker: str = "o",
                             edgecolors: str = "k",
                             s: int = 100, zorder: int = 3) -> None:
        """
        Plots a 2D load profile on the specified axes.

        Parameters
        ----------
        ax: plt.axes
            The axes to plot on. Defaults to None.
        label: str
            The label for the plot. Defaults to an empty string.
        color: str
            The color of the marker. Defaults to None.
        marker: str
            The marker style. Defaults to 'o'.
        edgecolors: str
            The color of the marker edges. Defaults to 'k'.
        s: int
            The size of the marker. Defaults to 100.
        zorder: int
            The z-order of the plot. Defaults to 3.

        Returns
        -------
        None
        """
        ax.scatter(self.__load_profile[0], self.__load_profile[1], color=color, marker=marker, label=label, s=s,
                   edgecolors=edgecolors, zorder=zorder)

    def __repr__(self) -> str:
        """
        Returns a string representation of the object.

        Returns
        -------
        str: The string representation of the object.
        """
        return f"{self.__class__.__name__}(id={self.__id}, load_profile={self.__load_profile})"


def virtualized(items: list[EnergyStorage], algorithm: Algorithms, signal_vectors: SignalVectors = None) \
        -> list[VirtualEnergyStorage]:
    """
    Converts a list of EnergyStorage objects into a list of VirtualEnergyStorage objects using the specified algorithm
    and signal vectors.

    Parameters
    ----------
    items: list[EnergyStorage]
        A list of EnergyStorage objects to convert.
    algorithm: Algorithms
        The algorithm to use for the conversion.
    signal_vectors: SignalVectors, optional
        The signal vectors to apply during the conversion. Defaults to None.

    Returns
    -------
    list[VirtualEnergyStorage]
        A list of VirtualEnergyStorage objects resulting from the conversion.
    """
    return [item.to_virtual(algorithm, signal_vectors) for item in items]
