import numpy as np
import pytest

import pyflexad.models.bess.tesla as tesla_bess
from pyflexad.math.signal_vectors import SignalVectors
from pyflexad.physical.stationary_battery import BESSUsage
from pyflexad.physical.stationary_battery import StationaryBattery
from pyflexad.utils.algorithms import Algorithms
from pyflexad.virtual.aggregator import Aggregator


class TestAggregator:
    testdata_aggregate = [
        (2, 2, 100, Algorithms.LPVG_GUROBIPY),
        (2, 2, 100, Algorithms.IABVG),
        (2, 2, 100, Algorithms.IABVG_JIT)
    ]

    testdata_stacked_aggregate = [
        (2, 2, 10, 10, 10, Algorithms.LPVG_GUROBIPY),
        (2, 2, 10, 10, 10, Algorithms.IABVG),
        (2, 2, 10, 10, 10, Algorithms.IABVG_JIT)
    ]

    testdata_aggregate_parallel = [
        (2, 10, 100, Algorithms.LPVG_GUROBIPY),
        (2, 10, 100, Algorithms.IABVG),
        (2, 10, 100, Algorithms.IABVG_JIT)
    ]

    @pytest.mark.parametrize("seed, d, n, algorithm", testdata_aggregate)
    def test_aggregate(self, seed: int, d, n, algorithm) -> None:
        if not algorithm.is_solver_available():
            pytest.skip(f"pyomo solver {algorithm.get_solver()} not available")

        """settings"""
        dt = 0.25

        """main"""
        np.random.seed(seed)
        signal_vectors = SignalVectors.new(d)

        usage = BESSUsage(initial_capacity=10.0, final_capacity=5.0, d=d, dt=dt)
        esr_list = [StationaryBattery.new(hardware=tesla_bess.power_wall_2, usage=usage) for _ in range(n)]

        agg_exact = Aggregator.from_physical(esr_list, algorithm=Algorithms.EXACT)
        agg_approx = Aggregator.from_physical(esr_list, algorithm=algorithm, signal_vectors=signal_vectors)

        assert np.allclose(np.sort(agg_exact.get_vertices(), axis=0), np.sort(agg_approx.get_vertices(), axis=0))

    @pytest.mark.parametrize("seed, d, n_1, n_2, n_3, algorithm", testdata_stacked_aggregate)
    def test_stacked_aggregate(self, seed: int, d, n_1, n_2, n_3, algorithm) -> None:
        if not algorithm.is_solver_available():
            pytest.skip(f"pyomo solver {algorithm.get_solver()} not available")

        """settings"""
        dt = 0.25

        """main"""
        np.random.seed(seed)
        signal_vectors = SignalVectors.new(d)
        usage = BESSUsage(initial_capacity=10.0, final_capacity=5.0, d=d, dt=dt)

        sub_aggs = []
        for n in [n_1, n_2, n_3]:
            esr_list = [StationaryBattery.new(hardware=tesla_bess.power_wall_2, usage=usage) for _ in range(n)]
            sub_aggs += [Aggregator.from_physical(esr_list, algorithm=algorithm, signal_vectors=signal_vectors)]

        agg_exact = Aggregator.aggregate(sub_aggs, algorithm=Algorithms.EXACT)
        agg_approx = Aggregator.aggregate(sub_aggs, algorithm=algorithm)

        assert np.allclose(np.sort(agg_exact.get_vertices(), axis=0), np.sort(agg_approx.get_vertices(), axis=0))
