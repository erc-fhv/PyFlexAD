from enum import StrEnum
from typing import Self

from pyomo.opt.base.solvers import check_available_solvers


class Algorithms(StrEnum):
    CENTRALIZED = "CENTRALIZED"
    EXACT = "EXACT"
    LPVG_GUROBIPY = "LPVG-GUROBIPY"
    LPVG_PYOMO_GUROBI = "LPVG-PYOMO-GUROBI"
    LPVG_PYOMO_GUROBI_APPSI = "LPVG-PYOMO-GUROBI-APPSI"
    LPVG_PYOMO_GUROBI_DIRECT = "LPVG-PYOMO-GUROBI-DIRECT"
    LPVG_PYOMO_CBC = "LPVG-PYOMO-CBC"
    LPVG_PYOMO_GLPK = "LPVG-PYOMO-GLPK"
    LPVG_PYOMO_HIGHS_APPSI = "LPVG-PYOMO-HIGHS-APPSI"
    IABVG = "IABVG"
    IABVG_JIT = "IABVG-JIT"

    @classmethod
    def get_aggregation_algorithms(cls) -> tuple[Self, ...]:
        return tuple(aggregation_algorithms)

    @classmethod
    def get_lpvg_pyomo_algorithms(cls) -> tuple[Self, ...]:
        return tuple(lpvg_pyomo_algorithms)

    def get_color(self) -> str:
        return colors.get(self, "black")

    def get_solver(self) -> str:
        return solvers.get(self, None)

    def is_solver_available(self) -> bool:

        if self in lpvg_pyomo_algorithms:
            solver = self.get_solver()
            available = bool(check_available_solvers(solver))
        else:
            available = True

        return available


lpvg_pyomo_algorithms = [Algorithms.LPVG_PYOMO_GUROBI,
                         Algorithms.LPVG_PYOMO_GUROBI_APPSI,
                         # Algorithms.LPVG_PYOMO_GUROBI_DIRECT, # excluded due to problems with "sense"
                         Algorithms.LPVG_PYOMO_CBC,
                         Algorithms.LPVG_PYOMO_GLPK,
                         Algorithms.LPVG_PYOMO_HIGHS_APPSI]

aggregation_algorithms = [Algorithms.LPVG_GUROBIPY, Algorithms.IABVG, Algorithms.IABVG_JIT] + lpvg_pyomo_algorithms

colors = {Algorithms.CENTRALIZED: "tab:red",
          Algorithms.EXACT: "tab:purple",
          Algorithms.LPVG_GUROBIPY: "tab:green",
          Algorithms.LPVG_PYOMO_GUROBI: "lime",
          Algorithms.LPVG_PYOMO_GUROBI_APPSI: "tab:orange",
          Algorithms.LPVG_PYOMO_GUROBI_DIRECT: "tab:olive",
          Algorithms.LPVG_PYOMO_CBC: "tab:brown",
          Algorithms.LPVG_PYOMO_GLPK: "tab:pink",
          Algorithms.LPVG_PYOMO_HIGHS_APPSI: "tab:gray",
          Algorithms.IABVG: "tab:cyan",
          Algorithms.IABVG_JIT: "tab:blue"}

solvers = {
    Algorithms.LPVG_PYOMO_GUROBI: "gurobi",
    Algorithms.LPVG_PYOMO_GUROBI_APPSI: "appsi_gurobi",
    Algorithms.LPVG_PYOMO_GUROBI_DIRECT: "gurobi_direct",
    Algorithms.LPVG_PYOMO_CBC: "cbc",
    Algorithms.LPVG_PYOMO_GLPK: "glpk",
    Algorithms.LPVG_PYOMO_HIGHS_APPSI: "appsi_highs"
}
