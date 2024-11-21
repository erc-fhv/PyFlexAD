from abc import ABCMeta, abstractmethod
from typing import Self

import numpy as np

from pyflexad.math.signal_vectors import SignalVectors
from pyflexad.parameters.general_parameters import GeneralParameters


class LPVG(metaclass=ABCMeta):
    """Linear Programming Vertex Generation algorithm"""

    @classmethod
    def from_general_params(cls, general_params: GeneralParameters, solver: str) -> Self:
        return cls(**general_params.__dict__, solver=solver)

    def __init__(self, x_lower: np.ndarray, x_upper: np.ndarray,
                 s_lower: np.ndarray, s_upper: np.ndarray,
                 s_initial: float,
                 alpha: float,
                 d: int,
                 dt: float,
                 x0: float, solver: str) -> None:
        self.__x_lower = x_lower
        self.__x_upper = x_upper
        self.__s_lower = s_lower
        self.__s_upper = s_upper
        self.__s_initial = s_initial
        self.__alpha = alpha
        self.__d = d
        self.__dt = dt
        self.__x0 = x0
        self.__solver = solver.lower()

    @property
    def solver(self):
        return self.__solver

    @abstractmethod
    def approx_vertices(self, A: np.ndarray, b: np.ndarray, signal_vectors: SignalVectors) -> np.ndarray:
        """
        Approximate vertices based on input matrices and signal vectors.

        Parameters
        ----------
        A : np.ndarray
            The matrix A.
        b : np.ndarray
            The vector b.
        signal_vectors : SignalVectors
            An object containing signal vectors.

        Returns
        -------
        np.ndarray
            The vertices calculated based on the input.
        """
        pass
