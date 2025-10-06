import numpy as np
import pyomo.environ as pyo

from pyflexad.math.lpvg import LPVG
from pyflexad.math.signal_vectors import SignalVectors


class LPVGPyomo(LPVG):
    """Linear Programming Vertex Generation algorithm using Pyomo as backend"""

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

        with pyo.SolverFactory(self.solver) as opt:
            model = pyo.ConcreteModel(name="Vertex Generation")

            model.a0 = pyo.Set(initialize=range(A.shape[0]))
            model.a1 = pyo.Set(initialize=range(A.shape[1]))
            model.x = pyo.Var(model.a1, domain=pyo.Reals)

            @model.Constraint(model.a0, model.a1)
            def constraint(pyo_model: pyo.ConcreteModel, _i: int, _j: int):
                return A[_i, _j] * pyo_model.x[_j] <= b[_i]

            for i, c in enumerate(signal_vectors.signals):
                model.objective = pyo.Objective(expr=pyo.quicksum(c[j] * model.x[j] for j in model.a1),
                                                sense=pyo.maximize)

                results = opt.solve(model, tee=False)

                if results.solver.status == pyo.SolverStatus.ok:
                    vertices[i, :] = pyo.value(model.x[:])
                else:
                    raise RuntimeError(f"{model.name} = {results.solver.status}")

                model.del_component(model.objective)

        return vertices
