import numpy as np
import pytest

import pyflexad.models.bess.tesla as tesla_bess
from pyflexad.math.signal_vectors import SignalVectors
from pyflexad.optimization.centralized_cost_controller import CentralizedCostController
from pyflexad.optimization.centralized_cost_controller_pyomo import CentralizedCostControllerPyomo
from pyflexad.optimization.vertex_based_cost_controller import VertexBasedCostController
from pyflexad.physical.stationary_battery import StationaryBattery
from pyflexad.system.energy_prices import EnergyPrices
from pyflexad.system.household_data import HouseholdsData
from pyflexad.utils.algorithms import Algorithms
from pyflexad.virtual.aggregator import Aggregator


class TestCostReduction:
    testdata_solve_vertices = [
        (2, 2, 2, np.array([[-10., -10.], [10., -10.], [10., 10.], [-10., 10.]]), np.array([-10, -10])),
        (2, 2, 10, np.array([[-50., -50.], [50., -50.], [50., 50.], [-50., 50.]]), np.array([-50, -50])),
    ]

    testdata_solve_A_b = [
        (2, 2, 2, Algorithms.EXACT),
        (2, 2, 2, Algorithms.LPVG_GUROBIPY),
        (2, 2, 2, Algorithms.IABVG),
        (2, 2, 2, Algorithms.IABVG_JIT),

        (2, 10, 2, Algorithms.LPVG_GUROBIPY),
        # (2, 10, 2, Algorithms.IABVGX),
        # (2, 10, 2, Algorithms.IABVGX_JIT),

        (2, 2, 10, Algorithms.LPVG_GUROBIPY),
        (2, 2, 10, Algorithms.IABVG),
        (2, 2, 10, Algorithms.IABVG_JIT),
    ]

    testdata_optimize = [
        (2, 2, 2, Algorithms.EXACT, False),
        (2, 2, 2, Algorithms.LPVG_GUROBIPY, False),
        (2, 2, 2, Algorithms.IABVG, False),
        (2, 2, 2, Algorithms.IABVG_JIT, False),

        (2, 10, 2, Algorithms.LPVG_GUROBIPY, False),
        # (2, 10, 2, Algorithms.IABVGX, True),
        # (2, 10, 2, Algorithms.IABVGX_JIT, True),

        (2, 2, 10, Algorithms.LPVG_GUROBIPY, True),
        (2, 2, 10, Algorithms.IABVG, True),
        (2, 2, 10, Algorithms.IABVG_JIT, True),
    ]

    @pytest.mark.parametrize("seed, d, n, vertices, expected", testdata_solve_vertices)
    def test_decentralized(self, seed: int, d, n, vertices, expected) -> None:
        """settings"""
        dt = 0.25
        n_cost_vectors = 1

        """main"""
        np.random.seed(2)
        hh = HouseholdsData.from_random(min_power_demand=0.0, max_power_demand=1.0, n_households=n, n_time_periods=d)
        energy_prices = EnergyPrices.from_random(min_energy_price=0.1, max_energy_price=0.8,
                                                 n_cost_vectors=n_cost_vectors,
                                                 n_time_periods=d)
        dco = VertexBasedCostController(power_demand=hh.power_demand, energy_prices=energy_prices, dt=dt)
        power, _ = dco.solve(vertices)

        assert np.allclose(power, expected, atol=1e-3)

    @pytest.mark.parametrize("seed, d, n, algorithm", testdata_solve_A_b)
    def test_centralized(self, seed: int, d, n, algorithm) -> None:
        if not algorithm.is_solver_available():
            pytest.skip(f"pyomo solver {algorithm.get_solver()} not available")

        """settings"""
        dt = 0.25
        n_cost_vectors = 1

        """main"""
        np.random.seed(2)
        hh_data = HouseholdsData.from_random(min_power_demand=0.0, max_power_demand=1.0, n_households=n,
                                             n_time_periods=d)
        energy_prices = EnergyPrices.from_random(min_energy_price=0.1, max_energy_price=0.8,
                                                 n_cost_vectors=n_cost_vectors,
                                                 n_time_periods=d)
        esr_list = [StationaryBattery.random_usage(hardware=tesla_bess.power_wall_2, d=d, dt=dt) for _ in range(n)]

        co = CentralizedCostController(power_demand=hh_data.power_demand, energy_prices=energy_prices, dt=dt)
        power_centralized = co.optimize(esr_list)

        agg = Aggregator.from_physical(esr_list, signal_vectors=SignalVectors.new(d), algorithm=algorithm)
        dco = VertexBasedCostController(power_demand=hh_data.power_demand, energy_prices=energy_prices, dt=dt)
        power_decentralized, _ = dco.solve(agg.get_vertices())

        assert np.allclose(power_decentralized, power_centralized, atol=1e-3)

    @pytest.mark.parametrize("seed, d, n, algorithm", testdata_solve_A_b)
    def test_centralized_pyomo(self, seed: int, d, n, algorithm) -> None:
        if not algorithm.is_solver_available():
            pytest.skip(f"pyomo solver {algorithm.get_solver()} not available")

        """settings"""
        dt = 0.25
        n_cost_vectors = 1

        """main"""
        np.random.seed(2)
        hh_data = HouseholdsData.from_random(min_power_demand=0.0, max_power_demand=1.0, n_households=n,
                                             n_time_periods=d)
        energy_prices = EnergyPrices.from_random(min_energy_price=0.1, max_energy_price=0.8,
                                                 n_cost_vectors=n_cost_vectors,
                                                 n_time_periods=d)
        esr_list = [StationaryBattery.random_usage(hardware=tesla_bess.power_wall_2, d=d, dt=dt) for _ in range(n)]

        co = CentralizedCostControllerPyomo(power_demand=hh_data.power_demand, energy_prices=energy_prices, dt=dt)
        power_centralized = co.optimize(esr_list)

        # Todo add VertexBasedCostControllerPyomo
        agg = Aggregator.from_physical(esr_list, signal_vectors=SignalVectors.new(d), algorithm=algorithm)
        dco = VertexBasedCostController(power_demand=hh_data.power_demand, energy_prices=energy_prices, dt=dt)
        power_decentralized, _ = dco.solve(agg.get_vertices())

        assert np.allclose(power_decentralized, power_centralized, atol=1e-3)
