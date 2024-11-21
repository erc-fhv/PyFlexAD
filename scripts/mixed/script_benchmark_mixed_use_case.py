import logging
import os
import random

import numpy as np
import pandas as pd

import pyflexad.models.bess.tesla as tesla_bess
import pyflexad.models.ev.nissan as nissan_ev
import pyflexad.models.ev.tesla as tesla_ev
import pyflexad.models.tclc.generic as tclc_generic
import pyflexad.models.tclh.generic as tclh_generic
from pyflexad.math.signal_vectors import SignalVectors
from pyflexad.optimization.centralized_power_controller import CentralizedPowerController
from pyflexad.optimization.vertex_based_power_controller import VertexBasedPowerController
from pyflexad.physical.electric_vehicle import ElectricVehicle
from pyflexad.physical.stationary_battery import BESSUsage
from pyflexad.physical.stationary_battery import StationaryBattery
from pyflexad.physical.therm_load_cooling import TCLCUsage
from pyflexad.physical.therm_load_cooling import ThermostaticLoadCooling
from pyflexad.physical.therm_load_heating import TCLHUsage
from pyflexad.physical.therm_load_heating import ThermostaticLoadHeating
from pyflexad.system.ev_data import EVData
from pyflexad.system.household_data import HouseholdsData
from pyflexad.utils.algorithms import Algorithms
from pyflexad.utils.benchmark import Benchmark
from pyflexad.utils.file_utils import FileUtils
from pyflexad.utils.timer import Timer


def distribute_fixed_amount(fixed_amount, rows, cols):
    # Calculate the average amount per element
    average_amount = fixed_amount / (rows * cols)

    # Initialize the matrix with the average amount
    matrix = np.full((rows, cols), average_amount, dtype=int)

    # Distribute the remaining amount uniformly across all elements
    remaining_amount = fixed_amount - np.sum(matrix)
    matrix += remaining_amount // (rows * cols)

    # Adjust any remaining amount
    remaining_amount %= (rows * cols)

    # Distribute the remaining amount randomly to some elements
    indices = np.random.choice(rows * cols, remaining_amount, replace=False)
    for idx in indices:
        row = idx // cols
        col = idx % cols
        matrix[row][col] += 1

    if matrix.sum() != fixed_amount:
        raise ValueError(f"Sum of matrix elements ({matrix.sum()}) does not equal fixed_amount ({fixed_amount})")

    return matrix


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
         plot_barchart_time: bool = False,
         log_level: int = logging.WARNING
         ) -> Benchmark:
    """settings"""
    logging.basicConfig(level=log_level)

    dt = 0.25
    ev_models = nissan_ev.models + tesla_ev.models
    bess_models = tesla_bess.models
    # ev_models = (nissan_ev.nissan_leaf_6_6_kW_ac, )
    # bess_models = (tesla_bess.power_wall_2, )

    """data paths"""
    path_hh = os.path.join(FileUtils.data_dir, "processed_hh")
    path_ev = os.path.join(FileUtils.data_dir, "data_EV")

    """main"""
    np.random.seed(2)
    benchmark = Benchmark.from_algorithms(algorithms=algorithms, d_list=d_list,
                                          n_flexibilities_list=n_flexibilities_list,
                                          n_times=n_times, disaggregate=disaggregate)

    run_args = []
    for i, d in enumerate(d_list):
        for j, n_flexibilities in enumerate(n_flexibilities_list):
            matrix = distribute_fixed_amount(fixed_amount=n_flexibilities, rows=4, cols=3)
            entities = {
                "BESS": matrix[0, :],
                "EV": matrix[1, :],
                "TCLC": matrix[2, :],
                "TCLH": matrix[3, :],
                "PHES": [0, 0, 0],
                "Participants": [100, 100, 100]
            }

            n_entities = sum(entities["Participants"])
            n_aggregators = len(entities["Participants"])

            """dates"""
            dates_ev = pd.date_range('12/14/2014, 00:00', periods=d, freq='15 min')

            """
            data
            """
            hh_data = HouseholdsData.from_file(path_hh=path_hh, n_entities=n_entities, n_time_periods=d)

            ev_agg_av = np.zeros(d)
            ev_agg_dem = np.zeros(d)
            esr_list = []

            dc_opt = VertexBasedPowerController(power_demand=hh_data.power_demand)
            c_opt = CentralizedPowerController(power_demand=hh_data.power_demand)

            for _i in range(n_aggregators):
                ev_data = EVData.from_file(path_ev=path_ev, n_entities=entities["EV"][_i], n_time_periods=d, dt=dt,
                                           dates=dates_ev)

                ev_agg_av += ev_data.availability.sum(axis=0)
                ev_agg_dem += ev_data.power_demand.sum(axis=0)

                for _j in range(entities["EV"][_i]):
                    ev_model = np.random.choice(ev_models)
                    flexibility = ElectricVehicle.with_charging(hardware=ev_model,
                                                                initial_capacity=0.5 * ev_model.max_capacity,
                                                                availability=ev_data.availability[_j, :],
                                                                power_demand=ev_data.power_demand[_j, :],
                                                                charging=ev_data.charging[_j, :],
                                                                d=d,
                                                                dt=dt)
                    esr_list.append(flexibility)

                for _j in range(entities["BESS"][_i]):
                    bess_model = np.random.choice(bess_models)
                    init_capacity = np.random.uniform(0.5 * bess_model.max_capacity, bess_model.max_capacity)
                    usage = BESSUsage(initial_capacity=init_capacity, final_capacity=init_capacity, d=d, dt=dt)
                    flexibility = StationaryBattery.new(hardware=bess_model, usage=usage)
                    esr_list.append(flexibility)

                for _j in range(entities["TCLC"][_i]):
                    usage = TCLCUsage(theta_r=20, theta_a=30, theta_0=np.random.uniform(19, 21),
                                      delta=np.random.uniform(1.5, 2.5), d=d, dt=dt)
                    flexibility = ThermostaticLoadCooling.new(hardware=tclc_generic.air_conditioner_1, usage=usage)
                    esr_list.append(flexibility)

                for _j in range(entities["TCLH"][_i]):
                    demand = np.random.uniform(0, 4, d)
                    usage = TCLHUsage(theta_r=50, theta_a=30, theta_0=np.random.uniform(49, 51),
                                      delta=np.random.uniform(3, 7), demand=demand, d=d, dt=dt)
                    flexibility = ThermostaticLoadHeating.new(
                        hardware=tclh_generic.domestic_hot_water_heater_1, usage=usage)
                    esr_list.append(flexibility)

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
    benchmark.show(optimizer_name="Power optimization",
                   plot_2d_upr=plot_2d_upr, plot_3d_upr=plot_3d_upr, plot_heatmap_upr=plot_heatmap_upr,
                   plot_2d_time=plot_2d_time, plot_3d_time=plot_3d_time, plot_heatmap_time=plot_heatmap_time,
                   plot_barchart_upr=plot_barchart_upr, plot_barchart_time=plot_barchart_time)

    return benchmark


if __name__ == '__main__':
    main(d_list=[96, ],
         n_flexibilities_list=[50, 100, 150, 200],
         algorithms=(Algorithms.CENTRALIZED, Algorithms.IABVG_JIT),
         parallelize=True,
         plot_3d_upr=True, plot_3d_time=True, plot_barchart_upr=False, plot_barchart_time=False)
