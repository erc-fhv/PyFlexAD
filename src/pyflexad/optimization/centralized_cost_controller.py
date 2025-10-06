import logging

import numpy as np

from pyflexad.optimization.centralized_controller import CentralizedController
from pyflexad.physical.energy_storage import EnergyStorage
from ._solvers import import_gurobi

logger = logging.getLogger(__name__)


class CentralizedCostController(CentralizedController):

    def __init__(self, power_demand: np.ndarray, energy_prices: np.ndarray, dt: float) -> None:
        super().__init__(power_demand)
        self.dt = dt
        self.energy_prices = energy_prices

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
        gp = import_gurobi()
        n = len(items)

        """optimization"""
        logger.debug("Building Gurobi model for CentralizedCostController (d=%d, n=%d)", self._d, n)
        with gp.Model("Centralized optimization, cost reduction") as model:
            model.Params.OutputFlag = 0
            x = model.addMVar(shape=(self._d, n), lb=-gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name="x")

            model.addConstrs(self.constr_A_b(x, item, j) for j, item in enumerate(items))

            objective_expression = gp.quicksum(self.energy_prices @ (x[:, i]) * self.dt
                                               + self.energy_prices @ self._agg_power_demand
                                               for i in range(n))

            model.setObjective(expr=objective_expression, sense=gp.GRB.MINIMIZE)
            model.optimize()

            logger.debug("Gurobi solve finished with status=%s", model.status)
            if model.status != gp.GRB.Status.OPTIMAL:
                raise RuntimeError(
                    f"CentralizedCostController.optimize failed (status={model.status}). "
                    f"Check cost function and constraints; infeasibility or unboundedness may be the cause."
                )

            if not minimize:
                """workaround maximization after minimization"""
                model.setObjective(expr=objective_expression, sense=gp.GRB.MAXIMIZE)
                model.optimize()

                if model.status not in [gp.GRB.Status.OPTIMAL,
                                        gp.GRB.Status.UNBOUNDED]:  # -> seams to work in some cases
                    raise RuntimeError(
                        f"CentralizedCostController maximize pass failed (status={model.status})."
                    )

            """save operation point power to flexibilities"""
            for j, item in enumerate(items):
                item.set_load_profile(x.X[:, j])

            individual_power = x.X

        return individual_power

    def calc_upr(self, **kwargs) -> float:
        return self._calc_cost_upr(**kwargs)
