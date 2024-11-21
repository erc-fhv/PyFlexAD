import logging
import random

import numpy as np

import pyflexad.models.bess.tesla as tesla_bess
from pyflexad.math.signal_vectors import SignalVectors
from pyflexad.optimization.centralized_power_controller import CentralizedPowerController
from pyflexad.optimization.vertex_based_power_controller import VertexBasedPowerController
from pyflexad.physical.stationary_battery import StationaryBattery
from pyflexad.system.household_data import HouseholdsData
from pyflexad.utils.algorithms import Algorithms
from pyflexad.utils.benchmark import Benchmark
from pyflexad.utils.timer import Timer


def main(d_list: list = (2, 4, 8, 16), n_flexibilities_list: list = (2, 4, 8, 16),
         algorithms: tuple = (Algorithms.CENTRALIZED, Algorithms.IABVG_JIT),
         disaggregate: bool = False,
         parallelize: bool = True,
         shuffle_execution_order: bool = False,
         n_times: int = 1,
         plot_2d_upr: bool = False,
         plot_3d_upr: bool = False,
         plot_heatmap_upr: bool = False,
         plot_2d_time: bool = False,
         plot_3d_time: bool = False,
         plot_heatmap_time: bool = False,
         plot_barchart_upr: bool = False,
         plot_barchart_time: bool = False
         ) -> Benchmark:
    """settings"""
    logging.basicConfig(level=logging.WARNING)
    dt = 0.25

    """main"""
    np.random.seed(2)

    benchmark = Benchmark.from_algorithms(algorithms=algorithms, d_list=d_list,
                                          n_flexibilities_list=n_flexibilities_list,
                                          n_times=n_times, disaggregate=disaggregate)

    run_args = []
    for i, d in enumerate(d_list):
        for j, n_flexibilities in enumerate(n_flexibilities_list):
            hh = HouseholdsData.from_random(min_power_demand=0.0, max_power_demand=1.0, n_households=n_flexibilities,
                                            n_time_periods=d)
            dc_opt = VertexBasedPowerController(power_demand=hh.power_demand)
            c_opt = CentralizedPowerController(power_demand=hh.power_demand)

            esr_list = [StationaryBattery.random_usage(hardware=tesla_bess.power_wall_2, d=d, dt=dt)
                        for _ in range(n_flexibilities)]

            run_args.append((c_opt, dc_opt, esr_list, d, SignalVectors.g_of_d_exp_2(d), i, j))

    if shuffle_execution_order:
        random.shuffle(run_args)

    with Timer() as main_timer:
        if parallelize:
            benchmark.run_parallel(run_args)
        else:
            benchmark.run_batch(run_args)

    print(f"Total time: {main_timer.dt:.2f}s")

    """plot"""
    benchmark.show(optimizer_name="Power Optimization",
                   plot_2d_upr=plot_2d_upr, plot_3d_upr=plot_3d_upr, plot_heatmap_upr=plot_heatmap_upr,
                   plot_2d_time=plot_2d_time, plot_3d_time=plot_3d_time, plot_heatmap_time=plot_heatmap_time,
                   plot_barchart_upr=plot_barchart_upr, plot_barchart_time=plot_barchart_time)

    return benchmark


if __name__ == '__main__':
    main(d_list=[2 ** i for i in range(3, 7)] + [96],
         n_flexibilities_list=[2 ** i for i in range(2, 7)],
         algorithms=(Algorithms.CENTRALIZED, Algorithms.IABVG_JIT),
         plot_3d_upr=True, plot_3d_time=True, plot_barchart_time=True)
