import logging

import numpy as np

from pyflexad.optimization.vertex_based_controller import VertexBasedController
from ._solvers import import_gurobi

logger = logging.getLogger(__name__)


class VertexBasedPowerController(VertexBasedController):

    def solve(self, vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Perform decentralized optimization for peak shaving using the given vertices.

        Parameters
        ----------
        vertices: np.ndarray
            Array of vertices for optimization.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Aggregated power distribution and alpha values after optimization.
        """
        gp = import_gurobi()
        vertices_T = vertices.T
        n_vertices = vertices_T.shape[1]

        """optimization"""
        logger.debug("Building Gurobi model for VertexBasedPowerController (d=%d, n_vertices=%d)", self._d, n_vertices)
        with gp.Model("Decentralized optimization, peak shaving") as model:
            model.Params.OutputFlag = 0
            x = model.addMVar(shape=self._d, lb=-gp.GRB.INFINITY, name="x")
            alpha = model.addMVar(shape=(n_vertices,), ub=1)
            t = model.addVar(lb=0.0)

            model.addConstrs(-t <= x[i] + self._agg_power_demand[i] for i in range(self._d))
            model.addConstrs(x[i] + self._agg_power_demand[i] <= t for i in range(self._d))
            model.addConstr(x == vertices_T @ alpha, name="x_constraint")
            model.addConstr(gp.quicksum(alpha) == 1, name="alpha_constraint")

            model.setObjective(t, gp.GRB.MINIMIZE)
            model.optimize()

            logger.debug("Gurobi solve finished with status=%s", model.status)
            if model.status != gp.GRB.Status.OPTIMAL:
                raise RuntimeError(
                    f"VertexBasedPowerController.solve failed (status={model.status})."
                )

            aggregated_power = x.X
            alphas = alpha.X

        return aggregated_power, alphas

    def calc_upr(self, **kwargs) -> float:
        return self._calc_power_upr(**kwargs)
