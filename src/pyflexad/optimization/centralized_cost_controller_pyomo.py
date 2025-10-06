import logging
from typing import List

import numpy as np
import pyomo.environ as pyo

from pyflexad.physical.energy_storage import EnergyStorage
from .centralized_controller import CentralizedController

logger = logging.getLogger(__name__)


class CentralizedCostControllerPyomo(CentralizedController):
    """Pyomo-based centralized cost minimization controller.

    Uses an open-source LP solver (e.g., HiGHS, CBC, GLPK) through Pyomo.
    """

    def __init__(self, power_demand: np.ndarray, energy_prices: np.ndarray, dt: float,
                 solver: str = "appsi_highs") -> None:
        super().__init__(power_demand)
        self.dt = dt
        self.energy_prices = energy_prices
        self.solver = solver

    def _build_model(self, items: List[EnergyStorage]) -> pyo.ConcreteModel:
        d = self._d
        n = len(items)

        m = pyo.ConcreteModel(name="Centralized optimization, cost reduction (Pyomo)")
        m.I = pyo.RangeSet(0, d - 1)
        m.J = pyo.RangeSet(0, n - 1)

        m.x = pyo.Var(m.I, m.J, domain=pyo.Reals)

        # Polytope constraints A x[:, j] <= b for each item j
        def _polytope_constr_rule(_m, j, r):
            A, b = items[j].calc_A_b()
            return sum(A[r, i] * _m.x[i, j] for i in _m.I) <= b[r]

        A0, b0 = items[0].calc_A_b()
        m.R = pyo.RangeSet(0, A0.shape[0] - 1)
        m.polytope_constr = pyo.Constraint(m.J, m.R, rule=_polytope_constr_rule)

        # Objective: minimize sum_i p[i]*(sum_j x[i,j] + agg[i]) * dt
        def objective_expr(_m):
            return sum(
                self.energy_prices[:, i] * (sum(_m.x[i, j] for j in _m.J) + self._agg_power_demand[i]) * self.dt
                for i in _m.I
            )

        m.obj = pyo.Objective(rule=objective_expr, sense=pyo.minimize)
        return m

    def _solve_with_pyomo(self, model: pyo.ConcreteModel) -> 'pyo.SolverResults':
        with pyo.SolverFactory(self.solver) as opt:
            if not opt.available(False):
                raise RuntimeError(
                    f"Pyomo solver '{self.solver}' is not available. Install and ensure it's on PATH."
                )
            res = opt.solve(model, tee=False)
        return res

    def solve(self, items: list[EnergyStorage], minimize: bool = True) -> np.ndarray:
        model = self._build_model(items)
        results = self._solve_with_pyomo(model)

        if results.solver.status != pyo.SolverStatus.ok:
            raise RuntimeError(
                f"CentralizedCostControllerPyomo failed (status={results.solver.status}, "
                f"termination={results.solver.termination_condition})."
            )

        x = np.zeros((self._d, len(items)), dtype=float)
        for j, item in enumerate(items):
            for i in range(self._d):
                x[i, j] = pyo.value(model.x[i, j])
            item.set_load_profile(x[:, j])

        return x

    def calc_upr(self, **kwargs) -> float:
        return self._calc_cost_upr(**kwargs)
