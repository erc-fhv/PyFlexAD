import logging

import numpy as np

from pyflexad.optimization.centralized_controller import CentralizedController
from pyflexad.physical.energy_storage import EnergyStorage
from ._solvers import import_gurobi

logger = logging.getLogger(__name__)


class CentralizedPowerController(CentralizedController):

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
        logger.debug("Building Gurobi model for CentralizedPowerController (d=%d, n=%d)", self._d, n)
        with gp.Model("Centralized optimization, peak shaving") as model:
            model.Params.OutputFlag = 0
            x = model.addMVar(shape=(self._d, n), lb=-float("inf"))
            t = model.addVar(lb=0.0)

            for i in range(self._d):
                model.addConstr(-t <= self._agg_power_demand[i] + gp.quicksum(x[i, j] for j in range(n)))
                model.addConstr(self._agg_power_demand[i] + gp.quicksum(x[i, j] for j in range(n)) <= t)

            model.addConstrs(self.constr_A_b(x, item, j) for j, item in enumerate(items))

            model.setObjective(t, gp.GRB.MINIMIZE)
            model.optimize()

            logger.debug("Gurobi solve finished with status=%s", model.status)
            if model.status != gp.GRB.Status.OPTIMAL:
                raise RuntimeError(
                    f"CentralizedPowerController.optimize failed (status={model.status}). "
                    f"Check model formulation and constraints; infeasibility or unboundedness may be the cause."
                )

            if not minimize:
                """workaround maximization after minimization"""
                model.setObjective(t, gp.GRB.MAXIMIZE)
                model.optimize()

                if model.status not in [gp.GRB.Status.OPTIMAL, gp.GRB.Status.UNBOUNDED]:  # seams to work in some cases
                    raise RuntimeError(
                        f"CentralizedPowerController maximize pass failed (status={model.status})."
                    )

            """save operation point power to flexibilities"""
            for j, item in enumerate(items):
                item.set_load_profile(x.X[:, j])

            individual_powers = x.X
        return individual_powers

    def calc_upr(self, **kwargs) -> float:
        return self._calc_power_upr(**kwargs)
