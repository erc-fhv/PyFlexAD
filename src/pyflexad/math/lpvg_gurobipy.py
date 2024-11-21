from typing import Self

import gurobipy as gp
import numpy as np

from pyflexad.math.lpvg import LPVG
from pyflexad.math.signal_vectors import SignalVectors
from pyflexad.parameters.general_parameters import GeneralParameters


class LPVGGurobipy(LPVG):
    """Linear Programming Vertex Generation algorithm using GurobiPy as backend"""
    @classmethod
    def from_general_params(cls, general_params: GeneralParameters, solver: str = "gurobi") -> Self:
        return cls(**general_params.__dict__, solver="gurobi")

    @property
    def solver(self) -> str:
        return "gurobi"

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

        """calculate vertices for approximation"""
        vertices = np.zeros((signal_vectors.g, signal_vectors.d))

        with gp.Model("Vertex Generation") as model:
            model.Params.OutputFlag = 0
            x: gp.MVar = model.addMVar(shape=(A.shape[1],), lb=-gp.GRB.INFINITY)
            model.addConstr(A @ x <= b, name="x_constraint")

            for i, c in enumerate(signal_vectors.signals):
                model.setObjective(c @ x, gp.GRB.MAXIMIZE)
                model.optimize()

                if model.status == gp.GRB.Status.OPTIMAL:
                    vertices[i] = x.X
                else:
                    raise RuntimeError(f"GUROBIPY Status: {model.status}")

        return vertices
