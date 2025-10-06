import os

import numpy as np
import pytest

import pyflexad.models.bess.tesla as tesla_bess
from pyflexad.math.signal_vectors import SignalVectors
from pyflexad.optimization.centralized_power_controller import CentralizedPowerController
from pyflexad.optimization.vertex_based_power_controller import VertexBasedPowerController
from pyflexad.physical.stationary_battery import StationaryBattery
from pyflexad.system.household_data import HouseholdsData
from pyflexad.utils.algorithms import Algorithms
from pyflexad.utils.benchmark import Benchmark


class TestBenchmark:
    n_cpu = os.cpu_count()
    testdata_benchmark = [(2, (2, 4, 8), (2, 4, 8), (Algorithms.CENTRALIZED, Algorithms.IABVG_JIT), False),
                          pytest.param(2, (2, 4, 8), (2, 4, 8), (Algorithms.CENTRALIZED, Algorithms.IABVG_JIT), True,
                                       marks=pytest.mark.skipif(n_cpu > 2, reason="not enough CPU cores"))
                          ]

    @pytest.mark.parametrize("seed, d_list, n_flexibilities_list, algorithms, parallelize", testdata_benchmark)
    def test_benchmark(self, seed: int, d_list: list[int] | tuple[int], n_flexibilities_list: tuple, algorithms: tuple,
                       parallelize) -> None:

        """settings"""
        dt = 0.25
        n_times = 1
        disaggregate = False

        """main"""
        np.random.seed(seed)

        benchmark = Benchmark.from_algorithms(algorithms=algorithms, d_list=d_list,
                                              n_flexibilities_list=n_flexibilities_list,
                                              n_times=n_times, disaggregate=disaggregate)

        run_args = []
        for i, d in enumerate(d_list):
            for j, n_flexibilities in enumerate(n_flexibilities_list):
                hh = HouseholdsData.from_random(min_power_demand=0.0, max_power_demand=1.0,
                                                n_households=n_flexibilities,
                                                n_time_periods=d)
                dc_opt = VertexBasedPowerController(power_demand=hh.power_demand)
                c_opt = CentralizedPowerController(power_demand=hh.power_demand)

                esr_list = [StationaryBattery.random_usage(hardware=tesla_bess.power_wall_2, d=d, dt=dt)
                            for _ in range(n_flexibilities)]

                run_args.append((c_opt, dc_opt, esr_list, d, SignalVectors.g_of_d_exp_2(d), i, j))

            if parallelize:
                benchmark.run_parallel(run_args)
            else:
                benchmark.run_batch(run_args)
