import logging
import os

import matplotlib.pyplot as plt
import numpy as np

import pyflexad.models.ev.tesla as tesla_ev
from pyflexad.math.signal_vectors import SignalVectors
from pyflexad.optimization.centralized_cost_controller import CentralizedCostController
from pyflexad.optimization.centralized_power_controller import CentralizedPowerController
from pyflexad.optimization.vertex_based_cost_controller import VertexBasedCostController
from pyflexad.optimization.vertex_based_power_controller import VertexBasedPowerController
from pyflexad.physical.electric_vehicle import ElectricVehicle
from pyflexad.system.energy_prices import EnergyPrices
from pyflexad.system.household_data import HouseholdsData
from pyflexad.utils.algorithms import Algorithms
from pyflexad.utils.file_utils import FileUtils
from pyflexad.virtual.aggregator import Aggregator


def main() -> None:
    """settings"""
    logging.basicConfig(level=logging.INFO)

    d = 2
    dt = 24 / d
    n_entities = 2
    n_cost_vectors = 1
    cost_optimization = False
    use_random_data = True
    plot = True

    """data paths"""
    path_da = os.path.join(FileUtils.data_dir, "processed_da")
    path_hh = os.path.join(FileUtils.data_dir, "processed_hh")

    """main"""
    np.random.seed(2)

    if use_random_data:
        energy_prices = EnergyPrices.from_random(min_energy_price=0.1, max_energy_price=0.8,
                                                 n_cost_vectors=n_cost_vectors,
                                                 n_time_periods=d)
        hh_data = HouseholdsData.from_random(min_power_demand=0.0, max_power_demand=1.0, n_households=n_entities,
                                             n_time_periods=d)
    else:
        hh_data = HouseholdsData.from_file(path_hh=path_hh, n_entities=n_entities, n_time_periods=d)
        energy_prices = EnergyPrices.from_file(path_da=path_da, n_time_periods=d, n_cost_vectors=n_cost_vectors)

    esr_list = [ElectricVehicle.random_usage(hardware=tesla_ev.model_y_100kw_dc, d=d, dt=dt) for _ in range(n_entities)]

    """virtualize"""
    signal_vectors = SignalVectors.new(d, g=SignalVectors.g_of_2_d_10(d))

    """aggregate"""
    agg_exact = Aggregator.from_physical(esr_list, algorithm=Algorithms.EXACT)
    agg_iabvg = Aggregator.from_physical(esr_list, algorithm=Algorithms.LPVG_GUROBIPY, signal_vectors=signal_vectors)
    agg_iabvgx = Aggregator.from_physical(esr_list, algorithm=Algorithms.IABVG, signal_vectors=signal_vectors)

    if cost_optimization:
        dco = VertexBasedCostController(power_demand=hh_data.power_demand, energy_prices=energy_prices, dt=dt)
        co = CentralizedCostController(power_demand=hh_data.power_demand, energy_prices=energy_prices, dt=dt)
    else:
        dco = VertexBasedPowerController(power_demand=hh_data.power_demand)
        co = CentralizedPowerController(power_demand=hh_data.power_demand)

    co_power = co.optimize(esr_list)
    dco.optimize(agg_exact)
    dco.optimize(agg_iabvg)
    dco.optimize(agg_iabvgx)

    if plot:
        """plot polytopes"""
        fig, axes = plt.subplots(3, 1, figsize=(8, 16))

        for i, esr in enumerate(esr_list):
            power = esr.get_load_profile()
            axes[0].scatter(power[0], power[1], label=f"Centralized OperationPoint[{i}]", color='k', marker="x")
        axes[0].scatter(co_power[0], co_power[1], label=f"Centralized OperationPoint", color='k', marker="x")

        """Exact polytopes"""
        for i, v_esr in enumerate(agg_exact.get_items()):
            v_esr.plot_polytope_2d(axes[0], line_style='--', label=f"Flexibility[{i}]")
            v_esr.plot_load_profile_2d(axes[0], label=f"OperationPoint[{i}]")

        agg_exact.plot_polytope_2d(axes[0], label='Exact', color='r', title='Exact')
        agg_exact.plot_load_profile_2d(axes[0], label='OperationPoint Exact', color='r')

        """IABVG polytopes"""
        for i, v_esr in enumerate(agg_iabvg.get_items()):
            v_esr.plot_polytope_2d(axes[1], line_style='--', label=f"Flexibility[{i}]")
            v_esr.plot_load_profile_2d(axes[1], label=f"OperationPoint[{i}]")

        agg_iabvg.plot_polytope_2d(axes[1], label='IABVG', color='g', hatch='|', fill=True, title='IABVG')
        agg_iabvg.plot_load_profile_2d(axes[1], label='OperationPoint IABVG', color='g')
        agg_exact.plot_polytope_2d(axes[2], color='r')

        """IABVGX polytope"""
        agg_exact.plot_polytope_2d(axes[1], color='r')

        for i, v_esr in enumerate(agg_iabvgx.get_items()):
            v_esr.plot_polytope_2d(axes[2], line_style='--', label=f"Flexibility[{i}]")
            v_esr.plot_load_profile_2d(axes[2], label=f"OperationPoint[{i}]")

        agg_iabvgx.plot_polytope_2d(axes[2], label='IABVGX', color='b', hatch='//', fill=True, title='IABVGX')
        agg_iabvgx.plot_load_profile_2d(axes[2], label='OperationPoint IABVGX', color='b')
        agg_exact.plot_polytope_2d(axes[2], color='r')

        for ax in axes:
            ax.legend(loc='lower left')

        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    main()
