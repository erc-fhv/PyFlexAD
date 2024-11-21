import copy
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
from pyflexad.utils.file_utils import FileUtils
from pyflexad.virtual.aggregator import Aggregator


def main() -> None:
    """settings"""
    d = 96
    dt = 0.25
    # algorithm = Algorithms.IABVG
    algorithm = Algorithms.IABVG_JIT
    penetrations = [0.20, 0.40]
    ev_models = nissan_ev.models + tesla_ev.models
    bess_models = tesla_bess.models
    # ev_model = nissan_ev.nissan_leaf_6_6_kW_ac
    # bess_model = tesla_bess.power_wall_2
    const_entities = {
        "BESS": [10, 10, 10],
        "EV": [10, 10, 10],
        "TCLC": [10, 10, 10],
        "TCLH": [10, 10, 10],
        "PHES": [0, 0, 0],
        "Participants": [100, 100, 100]
    }

    """data paths"""
    path_hh = os.path.join(FileUtils.data_dir, "processed_hh")
    path_ev = os.path.join(FileUtils.data_dir, "data_EV")

    """dates"""
    dates_ev = pd.date_range('12/14/2014, 00:00', periods=d, freq='15 min')
    dates_h = pd.date_range('12/14/2009, 00:00', periods=d, freq='15 min')

    """main"""
    np.random.seed(2)

    n_entities = sum(const_entities["Participants"])
    n_aggregators = len(const_entities["Participants"])

    signal_vectors = SignalVectors.new(d, g=SignalVectors.g_of_d_exp_2(d))
    hh_data = HouseholdsData.from_file(path_hh=path_hh, n_entities=n_entities, n_time_periods=d)

    res_dict = {"base load": hh_data.power_demand.sum(axis=0)}

    for penetration in penetrations:
        entities = copy.deepcopy(const_entities)

        for i in range(len(const_entities["BESS"])):
            entities["BESS"][i] = int(const_entities["BESS"][i] * penetration)
        for i in range(len(const_entities["EV"])):
            entities["EV"][i] = int(const_entities["EV"][i] * penetration)
        for i in range(len(const_entities["TCLC"])):
            entities["TCLC"][i] = int(const_entities["TCLC"][i] * penetration)
        for i in range(len(const_entities["TCLH"])):
            entities["TCLH"][i] = int(const_entities["TCLH"][i] * penetration)
        for i in range(len(const_entities["PHES"])):
            entities["PHES"][i] = int(const_entities["PHES"][i] * penetration)

        """
        data
        """
        sub_agg_list = []
        for i in range(n_aggregators):
            flexibilities = []
            ev_data = EVData.from_file(path_ev=path_ev, n_entities=entities["EV"][i], n_time_periods=d, dt=dt,
                                       dates=dates_ev)

            for j in range(entities["EV"][i]):
                ev_model = np.random.choice(ev_models)
                esr_list = ElectricVehicle.with_charging(hardware=ev_model,
                                                         initial_capacity=0.5 * ev_model.max_capacity,
                                                         availability=ev_data.availability[j, :],
                                                         power_demand=ev_data.power_demand[j, :],
                                                         charging=ev_data.charging[j, :],
                                                         d=d,
                                                         dt=dt)
                flexibilities.append(esr_list)

            for j in range(entities["BESS"][i]):
                bess_model = np.random.choice(bess_models)
                init_capacity = np.random.uniform(0.5 * bess_model.max_capacity, bess_model.max_capacity)
                usage = BESSUsage(initial_capacity=init_capacity, final_capacity=init_capacity, d=d, dt=dt)
                esr_list = StationaryBattery.new(hardware=bess_model, usage=usage)
                flexibilities.append(esr_list)

            for j in range(entities["TCLC"][i]):
                usage = TCLCUsage(theta_r=20, theta_a=30, theta_0=np.random.uniform(19, 21),
                                  delta=np.random.uniform(1.5, 2.5), d=d, dt=dt)
                esr_list = ThermostaticLoadCooling.new(hardware=tclc_generic.air_conditioner_1, usage=usage)
                flexibilities.append(esr_list)

            for j in range(entities["TCLH"][i]):
                demand = np.random.uniform(0, 4, d)
                usage = TCLHUsage(theta_r=50, theta_a=30, theta_0=np.random.uniform(49, 51),
                                  delta=np.random.uniform(3, 7), demand=demand, d=d, dt=dt)
                esr_list = ThermostaticLoadHeating.new(hardware=tclh_generic.domestic_hot_water_heater_1, usage=usage)
                flexibilities.append(esr_list)

            sub_agg_list.append(
                Aggregator.from_physical(flexibilities, algorithm=algorithm, signal_vectors=signal_vectors))

        """
        computation
        """
        top_agg = Aggregator.aggregate(items=sub_agg_list, algorithm=algorithm)

        dco = VertexBasedPowerController(power_demand=hh_data.power_demand)
        dco_power = dco.optimize(top_agg)

        res_dict[f"{penetration * 100} %"] = (hh_data.power_demand.sum(axis=0)
                                              + dco_power
                                              - dco_power.mean() * np.ones(d))

        """only interested in demand, negative values are supply not demand"""
        res_dict[f"{penetration * 100} %"][res_dict[f"{penetration * 100} %"] < 0] = 0

    """
    results and plots
    """
    plt.figure()
    df = pd.DataFrame(index=dates_h, data=res_dict)
    df.plot.area(stacked=False, ylabel="demand (kW)", grid=True)
    plt.show()


if __name__ == '__main__':
    main()
