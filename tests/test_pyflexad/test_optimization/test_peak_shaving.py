import numpy as np
import pytest

import pyflexad.models.bess.tesla as tesla_bess
from pyflexad.math.signal_vectors import SignalVectors
from pyflexad.optimization.centralized_power_controller import CentralizedPowerController
from pyflexad.optimization.vertex_based_power_controller import VertexBasedPowerController
from pyflexad.physical.stationary_battery import StationaryBattery
from pyflexad.system.household_data import HouseholdsData
from pyflexad.utils.algorithms import Algorithms
from pyflexad.virtual.aggregator import Aggregator


class TestPeakShaving:
    testdata_solve_vertices = [
        (2, 2, 2, np.array([[-10., -10.], [10., -10.], [10., 10.], [-10., 10.]]), np.array([-0.985, -0.461])),
    ]

    testdata_solve_A_b = [
        (2, 2, 2, Algorithms.EXACT),
        (2, 2, 2, Algorithms.LPVG_GUROBIPY),
        (2, 2, 2, Algorithms.IABVG),
        (2, 2, 2, Algorithms.IABVG_JIT),

        (2, 10, 2, Algorithms.LPVG_GUROBIPY),
        (2, 10, 2, Algorithms.IABVG),
        (2, 10, 2, Algorithms.IABVG_JIT),

        (2, 10, 20, Algorithms.LPVG_GUROBIPY),
        (2, 10, 20, Algorithms.IABVG),
        (2, 10, 20, Algorithms.IABVG_JIT),
    ]

    @pytest.mark.parametrize("seed, d, n, vertices, expected", testdata_solve_vertices)
    def test_decentralized(self, seed: int, d, n, vertices, expected) -> None:
        """main"""
        np.random.seed(2)
        hh_data = HouseholdsData.from_random(min_power_demand=0.0, max_power_demand=1.0, n_households=n,
                                             n_time_periods=d)

        dco = VertexBasedPowerController(power_demand=hh_data.power_demand)
        power, _ = dco.solve(vertices)

        assert np.allclose(power, expected, atol=1e-3)

    @pytest.mark.parametrize("seed, d, n, algorithm", testdata_solve_A_b)
    def test_centralized(self, seed: int, d, n, algorithm) -> None:
        if not algorithm.is_solver_available():
            pytest.skip(f"pyomo solver {algorithm.get_solver()} not available")

        """dt"""
        dt = 0.25

        """main"""
        np.random.seed(2)
        hh_data = HouseholdsData.from_random(min_power_demand=0.0, max_power_demand=1.0, n_households=n,
                                             n_time_periods=d)

        esr_list = [StationaryBattery.random_usage(hardware=tesla_bess.power_wall_2, d=d, dt=dt) for _ in range(n)]

        co = CentralizedPowerController(power_demand=hh_data.power_demand)
        power_centralized = co.optimize(esr_list)

        agg = Aggregator.from_physical(esr_list, signal_vectors=SignalVectors.new(d), algorithm=algorithm)
        dco = VertexBasedPowerController(power_demand=hh_data.power_demand)
        power_decentralized, _ = dco.solve(agg.get_vertices())

        assert np.allclose(power_decentralized, power_centralized, atol=1e-3)
