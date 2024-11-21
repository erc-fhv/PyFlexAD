import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import pyflexad.models.bess.tesla as tesla_bess
import pyflexad.models.ev.nissan as nissan_ev
import pyflexad.models.ev.tesla as tesla_ev
import pyflexad.models.tclc.generic as tclc_generic
import pyflexad.models.tclh.generic as tclh_generic
from pyflexad.math.signal_vectors import SignalVectors
from pyflexad.optimization.centralized_cost_controller import CentralizedCostController
from pyflexad.optimization.centralized_power_controller import CentralizedPowerController
from pyflexad.optimization.vertex_based_cost_controller import VertexBasedCostController
from pyflexad.optimization.vertex_based_power_controller import VertexBasedPowerController
from pyflexad.physical.electric_vehicle import ElectricVehicle
from pyflexad.physical.stationary_battery import BESSUsage
from pyflexad.physical.stationary_battery import StationaryBattery
from pyflexad.physical.therm_load_cooling import TCLCUsage
from pyflexad.physical.therm_load_cooling import ThermostaticLoadCooling
from pyflexad.physical.therm_load_heating import TCLHUsage
from pyflexad.physical.therm_load_heating import ThermostaticLoadHeating
from pyflexad.system.energy_prices import EnergyPrices
from pyflexad.system.ev_data import EVData
from pyflexad.system.household_data import HouseholdsData
from pyflexad.utils.algorithms import Algorithms
from pyflexad.utils.file_utils import FileUtils
from pyflexad.utils.network_graph import NetworkGraph
from pyflexad.utils.timer import Timer
from pyflexad.virtual.aggregator import Aggregator


def main(d: int = 96,
         algorithm: Algorithms = Algorithms.IABVG_JIT,
         plot_ev_availability: bool = False,
         plot_ev_consumption: bool = False,
         plot_power: bool = False,
         plot_network_graph: bool = False,
         log_level=logging.INFO,
         cost_optimization: bool = False) -> None:
    """settings"""
    logging.basicConfig(level=log_level)

    dt = 0.25
    n_cost_vectors = 1
    ev_models = nissan_ev.models + tesla_ev.models
    bess_models = tesla_bess.models

    entities = {
        "BESS": [10, 10, 10],
        "EV": [10, 10, 10],
        "TCLC": [10, 10, 10],
        "TCLH": [10, 10, 10],
        "PHES": [0, 0, 0],
        "Participants": [100, 100, 100]
    }

    """data paths"""
    path_hh = os.path.join(FileUtils.data_dir, "processed_hh")
    path_da = os.path.join(FileUtils.data_dir, "processed_da")
    path_ev = os.path.join(FileUtils.data_dir, "data_EV")

    """dates"""
    dates_ev = pd.date_range('12/14/2014, 00:00', periods=d, freq='15 min')
    dates_h = pd.date_range('12/14/2009, 00:00', periods=d, freq='15 min')

    """main"""
    np.random.seed(2)
    n_entities = sum(entities["Participants"])
    n_aggregators = len(entities["Participants"])

    """
    data
    """
    signal_vectors = SignalVectors.new(d, g=SignalVectors.g_of_d_exp_2(d))
    hh_data = HouseholdsData.from_file(path_hh=path_hh, n_entities=n_entities, n_time_periods=d)
    energy_prices = EnergyPrices.from_file(path_da=path_da, n_time_periods=d, n_cost_vectors=n_cost_vectors)

    sub_agg_list = []
    co_esr_list = []
    ev_availability = np.zeros(d)
    ev_demand = np.zeros(d)

    for i in range(n_aggregators):
        esr_list = []
        ev_data = EVData.from_file(path_ev=path_ev, n_entities=entities["EV"][i], n_time_periods=d, dt=dt,
                                   dates=dates_ev)

        ev_availability += ev_data.availability.sum(axis=0)
        ev_demand += ev_data.power_demand.sum(axis=0)

        for j in range(entities["EV"][i]):
            ev_model = np.random.choice(ev_models)
            flexibility = ElectricVehicle.with_charging(hardware=ev_model,
                                                        initial_capacity=0.5 * ev_model.max_capacity,
                                                        availability=ev_data.availability[j, :],
                                                        power_demand=ev_data.power_demand[j, :],
                                                        charging=ev_data.charging[j, :],
                                                        d=d,
                                                        dt=dt)
            esr_list.append(flexibility)

        for j in range(entities["BESS"][i]):
            bess_model = np.random.choice(bess_models)
            init_capacity = np.random.uniform(0.5 * bess_model.max_capacity, bess_model.max_capacity)
            usage = BESSUsage(initial_capacity=init_capacity, final_capacity=init_capacity, d=d, dt=dt)
            flexibility = StationaryBattery.new(hardware=bess_model, usage=usage)
            esr_list.append(flexibility)

        for j in range(entities["TCLC"][i]):
            usage = TCLCUsage.from_celsius(theta_r_deg_c=20, theta_a_deg_c=30,
                                           theta_0_deg_c=np.random.uniform(19, 21),
                                           delta=np.random.uniform(1.5, 2.5), d=d, dt=dt)
            flexibility = ThermostaticLoadCooling.new(hardware=tclc_generic.air_conditioner_1, usage=usage)
            esr_list.append(flexibility)

        for j in range(entities["TCLH"][i]):
            demand = np.random.uniform(0, 4, d)
            usage = TCLHUsage.from_celsius(theta_r_deg_c=50, theta_a_deg_c=30,
                                           theta_0_deg_c=np.random.uniform(49, 51),
                                           delta=np.random.uniform(3, 7), demand=demand, d=d, dt=dt)
            flexibility = ThermostaticLoadHeating.new(hardware=tclh_generic.domestic_hot_water_heater_1, usage=usage)
            esr_list.append(flexibility)

        """add items to centralized optimization list"""
        co_esr_list += esr_list

        """add items to decentralized optimization list"""
        sub_agg_list.append(Aggregator.from_physical(esr_list, signal_vectors=signal_vectors, algorithm=algorithm))

    """
    computation
    """
    top_agg = Aggregator.aggregate(sub_agg_list, algorithm=algorithm)

    if cost_optimization:
        dco = VertexBasedCostController(power_demand=hh_data.power_demand, energy_prices=energy_prices, dt=dt)
        co = CentralizedCostController(power_demand=hh_data.power_demand, energy_prices=energy_prices, dt=dt)
    else:
        dco = VertexBasedPowerController(power_demand=hh_data.power_demand)
        co = CentralizedPowerController(power_demand=hh_data.power_demand)

    with Timer("Decentralized Optimization") as t_agg:
        dco_power = dco.optimize(top_agg)

    with Timer("Centralized Optimization") as t_central:
        co_power = co.optimize(co_esr_list)

    print(t_agg)
    print(t_central)

    """
    results
    """

    """we assume that the additional energy needed is considered in the demand curve"""
    load_data = {"base load": hh_data.power_demand.sum(axis=0),
                 "aggregation": hh_data.power_demand.sum(axis=0) + dco_power - dco_power.mean() * np.ones(d),
                 "centralized": hh_data.power_demand.sum(axis=0) + co_power - co_power.mean() * np.ones(d)}

    df = pd.DataFrame(index=dates_h, data=load_data)

    """only interested in demand, negative values are supply not demand"""
    df["aggregation"] = df["aggregation"].apply(lambda x: 0 if x < 0 else x)
    df["centralized"] = df["centralized"].apply(lambda x: 0 if x < 0 else x)
    print(f"exact peak: {np.round(np.linalg.norm(df['centralized'].values, np.inf), 2)} kW")
    print(f"approx peak: {np.round(np.linalg.norm(df['aggregation'].values, np.inf), 2)} kW")

    """plots"""
    if plot_power:
        plt.figure()

        df.plot.area(stacked=False, ylabel="demand (kW)", grid=True)
        plt.show()

    if plot_ev_availability:
        plt.figure()
        df = pd.DataFrame(index=dates_h, data=ev_availability, columns=["EV availability"])
        df.plot(style="-o", grid=True)
        plt.show()

    if plot_ev_consumption:
        plt.figure()
        df = pd.DataFrame(index=dates_h, data=ev_demand, columns=["EV trip consumption"])
        df.plot.area(ylabel="power (kW)", grid=True, color="orange")
        plt.show()

    if plot_network_graph:
        graph = NetworkGraph.from_virtual([top_agg, ])
        graph.create_tree()
        graph.plot_tree(ax=plt.subplots(figsize=(20, 20))[1])
        plt.title("Network Graph: Mixed")
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    main(algorithm=Algorithms.IABVG_JIT, plot_power=True)
