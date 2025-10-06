import logging
from typing import List

import numpy as np
import pyomo.environ as pyo

from pyflexad.physical.energy_storage import EnergyStorage
from .centralized_controller import CentralizedController

logger = logging.getLogger(__name__)


class CentralizedPowerControllerPyomo(CentralizedController):
    """Pyomo-based centralized peak-shaving controller.

    Uses an open-source MILP/LP solver through Pyomo. Recommended solvers:
    - highs (HiGHS)
    - cbc
    - glpk
    """

    def __init__(self, power_demand: np.ndarray, solver: str = "appsi_highs") -> None:
        super().__init__(power_demand)
        self.solver = solver

    def _build_model(self, items: List[EnergyStorage]) -> pyo.ConcreteModel:
        d = self._d
        n = len(items)

        m = pyo.ConcreteModel(name="Centralized optimization, peak shaving (Pyomo)")
        m.I = pyo.RangeSet(0, d - 1)
        m.J = pyo.RangeSet(0, n - 1)

        m.x = pyo.Var(m.I, m.J, domain=pyo.Reals)
        m.t = pyo.Var(domain=pyo.NonNegativeReals)

        # Peak shaving constraints: -t <= agg + sum_j x[i,j] <= t
        def _upper_constr(_m, i):
            return self._agg_power_demand[i] + sum(_m.x[i, j] for j in _m.J) <= _m.t

        def _lower_constr(_m, i):
            return -_m.t <= self._agg_power_demand[i] + sum(_m.x[i, j] for j in _m.J)

        m.peak_upper = pyo.Constraint(m.I, rule=_upper_constr)
        m.peak_lower = pyo.Constraint(m.I, rule=_lower_constr)

        # Polytope constraints A x[:, j] <= b for each item j
        def _polytope_constr_rule(_m, j, r):
            A, b = items[j].calc_A_b()
            # row r of A dotted with x[:, j]
            return sum(A[r, i] * _m.x[i, j] for i in _m.I) <= b[r]

        # Build per-item, per-row constraints
        # Precompute shapes since Pyomo sets are static
        A0, b0 = items[0].calc_A_b()
        m.R = pyo.RangeSet(0, A0.shape[0] - 1)
        m.polytope_constr = pyo.Constraint(m.J, m.R, rule=_polytope_constr_rule)

        # Objective: minimize t
        m.obj = pyo.Objective(expr=m.t, sense=pyo.minimize)
        return m

    def _solve_with_pyomo(self, model: pyo.ConcreteModel, sense_minimize: bool = True) -> 'pyo.SolverResults':
        with pyo.SolverFactory(self.solver) as opt:
            if not opt.available(False):
                raise RuntimeError(
                    f"Pyomo solver '{self.solver}' is not available. Install and ensure it's on PATH."
                )
            # Set sense
            if sense_minimize:
                model.obj.sense = pyo.minimize
            else:
                model.obj.sense = pyo.maximize
            res = opt.solve(model, tee=False)
        return res

    def solve(self, items: list[EnergyStorage], minimize: bool = True) -> np.ndarray:
        model = self._build_model(items)

        # First pass: minimize
        results = self._solve_with_pyomo(model, sense_minimize=True)
        if results.solver.status != pyo.SolverStatus.ok:
            raise RuntimeError(
                f"CentralizedPowerControllerPyomo failed (status={results.solver.status}, "
                f"termination={results.solver.termination_condition})."
            )

        # Optional second pass: maximize t after min pass (to mimic original logic)
        if not minimize:
            results = self._solve_with_pyomo(model, sense_minimize=False)
            term = results.solver.termination_condition
            if term not in {pyo.TerminationCondition.optimal, pyo.TerminationCondition.unbounded}:
                raise RuntimeError(
                    f"CentralizedPowerControllerPyomo maximize failed (termination={term})."
                )

        # Extract solution and store to items
        x = np.zeros((self._d, len(items)), dtype=float)
        for j, item in enumerate(items):
            for i in range(self._d):
                x[i, j] = pyo.value(model.x[i, j])
            item.set_load_profile(x[:, j])

        return x

    def calc_upr(self, **kwargs) -> float:
        return self._calc_power_upr(**kwargs)
