import gurobipy as gp
import numpy as np

from pyflexad.optimization.centralized_controller import CentralizedController
from pyflexad.physical.energy_storage import EnergyStorage


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
        n = len(items)

        """optimization"""
        with gp.Model("Centralized optimization, cost reduction") as model:
            model.Params.OutputFlag = 0
            x = model.addMVar(shape=(self._d, n), lb=-gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name="x")

            model.addConstrs(self.constr_A_b(x, item, j) for j, item in enumerate(items))

            objective_expression = gp.quicksum(self.energy_prices @ (x[:, i]) * self.dt
                                               + self.energy_prices @ self._agg_power_demand
                                               for i in range(n))

            model.setObjective(expr=objective_expression, sense=gp.GRB.MINIMIZE)
            model.optimize()

            if model.status != gp.GRB.Status.OPTIMAL:
                raise RuntimeError(f"{model.name} = {model.status}")

            if not minimize:
                """workaround maximization after minimization"""
                model.setObjective(expr=objective_expression, sense=gp.GRB.MAXIMIZE)
                model.optimize()

                if model.status not in [gp.GRB.Status.OPTIMAL,
                                        gp.GRB.Status.UNBOUNDED]:  # -> seams to work in some cases
                    raise RuntimeError(f"GUROBIPY Status: {model.status}")

            """save operation point power to flexibilities"""
            for j, item in enumerate(items):
                item.set_load_profile(x.X[:, j])

            individual_power = x.X

        return individual_power

    def calc_upr(self, **kwargs) -> float:
        return self._calc_cost_upr(**kwargs)
