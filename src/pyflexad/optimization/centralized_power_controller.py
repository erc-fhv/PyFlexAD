import gurobipy as gp
import numpy as np

from pyflexad.optimization.centralized_controller import CentralizedController
from pyflexad.physical.energy_storage import EnergyStorage


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
        n = len(items)

        """optimization"""
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

            if model.status != gp.GRB.Status.OPTIMAL:
                raise RuntimeError(f"GUROBIPY Status: {model.status}")

            if not minimize:
                """workaround maximization after minimization"""
                model.setObjective(t, gp.GRB.MAXIMIZE)
                model.optimize()

                if model.status not in [gp.GRB.Status.OPTIMAL, gp.GRB.Status.UNBOUNDED]:  # seams to work in some cases
                    raise RuntimeError(f"GUROBIPY Status: {model.status}")

            """save operation point power to flexibilities"""
            for j, item in enumerate(items):
                item.set_load_profile(x.X[:, j])

            individual_powers = x.X
        return individual_powers

    def calc_upr(self, **kwargs) -> float:
        return self._calc_power_upr(**kwargs)
