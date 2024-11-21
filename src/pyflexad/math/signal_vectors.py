import itertools
from typing import Self

import numpy as np


class SignalVectors:

    @staticmethod
    def g_of_2_exp_d(d: int) -> int:
        return 2 ** d

    @staticmethod
    def g_of_2_d_10(d: int) -> int:
        return 2 * d * (d + 10)

    @staticmethod
    def g_of_d_exp_2(d: int) -> int:
        return d ** 2

    @staticmethod
    def generate(d: int, g: int = None) -> np.ndarray:
        """
        Generates a matrix of signals based on the input dimension and the number of signals.

        Parameters
        ----------
        d: int
            The dimension of the signal vectors.
        g: int
            The number of signals to generate. Defaults to 2**d.

        Returns
        -------
        np.ndarray: A matrix containing the generated signals.
        """
        if g is None:
            g = 2 ** d

        if d > 8:
            signals = np.unique(np.random.choice([-1, 1], [g, d]), axis=1)
            while len(signals) < g:
                signals = np.unique(
                    np.stack([signals, np.random.choice([-1, 1], size=[g - len(signals), d])]),
                    axis=1)
        else:
            signals = np.array(list(itertools.product([-1, 1], repeat=d)))
        return signals

    @classmethod
    def new(cls, d: int, g: int = None) -> Self:
        """
        Generates a new instance of the SignalVectors class based on the input dimension and the number of signals.

        Parameters
        ----------
        d: int
            The dimension of the signal vectors.
        g: int, optional
            The number of signals to generate. Defaults to None.

        Returns
        -------
        Self: A new instance of the SignalVectors class.
        """
        signals = cls.generate(d=d, g=g)
        return cls(signals=signals)

    def __init__(self, signals: np.ndarray) -> None:
        self.signals = signals

    @property
    def d(self) -> int:
        return self.signals.shape[1]

    @property
    def g(self) -> int:
        return self.signals.shape[0]
