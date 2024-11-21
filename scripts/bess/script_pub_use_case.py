import matplotlib.pyplot as plt
import numpy as np

import pyflexad.models.bess.tesla as tesla
from pyflexad.math.signal_vectors import SignalVectors
from pyflexad.optimization.centralized_power_controller import CentralizedPowerController
from pyflexad.optimization.vertex_based_power_controller import VertexBasedPowerController
from pyflexad.physical.stationary_battery import BESSUsage
from pyflexad.physical.stationary_battery import StationaryBattery
from pyflexad.utils.algorithms import Algorithms
from pyflexad.virtual.aggregator import Aggregator

"""settings"""
d = 2  # number of time intervals
dt = 0.25  # interval duration in hours
system_power_demand = np.array([[23.0, 21.0]])  # total power demand for each interval in kW
algorithm = Algorithms.IABVG  # virtualization and aggregation algorithm
S_0 = 6.5  # initial battery capacity in kWh
S_f = 5.0  # final battery capacity in kWh

"""instantiate energy storage resources -> 2x Tesla Power Wall 2"""
usage = BESSUsage(initial_capacity=S_0, final_capacity=S_f, d=d, dt=dt)
bess_1 = StationaryBattery.new(hardware=tesla.power_wall_2, usage=usage)
bess_2 = StationaryBattery.new(hardware=tesla.power_wall_2, usage=usage)

"""virtualize"""
direction_vectors = SignalVectors.new(d)
virtual_ess_1 = bess_1.to_virtual(algorithm, direction_vectors)
virtual_ess_2 = bess_2.to_virtual(algorithm, direction_vectors)

"""aggregate"""
aggregator = Aggregator.aggregate([virtual_ess_1, virtual_ess_2], algorithm)

"""optimize power with centralized controller"""
centralized_controller = CentralizedPowerController(system_power_demand)
cc_power = centralized_controller.optimize([bess_1, bess_2])

"""optimize power with vertex-based controller"""
vertex_controller = VertexBasedPowerController(system_power_demand)
vc_power = vertex_controller.optimize(aggregator)

print(f"No Controller: {system_power_demand}")
print(f"Centralized Controller: {system_power_demand + cc_power}")
print(f"Vertex-Based Controller: {system_power_demand + vc_power}")

# %% exact feasibility region
"""virtualize exact"""
virtual_ess_ex_1 = bess_1.to_virtual(Algorithms.EXACT)
virtual_ess_ex_2 = bess_2.to_virtual(Algorithms.EXACT)

"""aggregate exact"""
aggregator_ex = Aggregator.aggregate([virtual_ess_ex_1, virtual_ess_ex_2], Algorithms.EXACT)

"""optimize power with vertex-based controller"""
vc_power_ex = vertex_controller.optimize(aggregator_ex)

# %% plotting
s = 200
fontsize = 13

"""plot polytopes"""
fig, ax = plt.subplots(figsize=(10, 10))

aggregator_ex.plot_polytope_2d(ax, label="Exact Aggregate Polytope", color='tab:red', line_style='-.')
aggregator.plot_polytope_2d(ax, label="Approx. Aggregate Polytope", color='tab:green', line_style='--')

aggregator_ex.plot_load_profile_2d(ax, label="Centralized / Exact Power Profile", color='tab:red', marker="$E$",
                                   edgecolors=None, s=s)
aggregator.plot_load_profile_2d(ax, label="Approx. Power Profile", color='tab:green', marker="$A$",
                                edgecolors=None, s=s)

virtual_ess_ex_1.plot_polytope_2d(ax, label="Exact Polytope: BESS 1 & 2", color='tab:red', line_style='-', hatch='//',
                                  fill=True, zorder=2)
virtual_ess_1.plot_polytope_2d(ax, label="Approx. Polytope: BESS 1 & 2", color='tab:green', fill=True, line_style='--')

virtual_ess_ex_1.plot_load_profile_2d(ax, label="Exact Power Profile: BESS 1 & 2", color='tab:red', marker="v", s=s)
virtual_ess_1.plot_load_profile_2d(ax, label="Approx. Power Profile: BESS 1 & 2", color='tab:green', marker="^", s=s)

bess_1.plot_load_profile_2d(ax, label="Centralized Power Profile: BESS 1", color='k', marker="$C_1$", s=s)
bess_2.plot_load_profile_2d(ax, label="Centralized Power Profile: BESS 2", color='k', marker="$C_2$", s=s)

ax.tick_params(axis='both', which='major', labelsize=fontsize)
ax.tick_params(axis='both', which='minor', labelsize=fontsize)
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=13)
ax.set_xlabel("$x_1$", fontsize=fontsize)
ax.set_ylabel("$x_2$", fontsize=fontsize)
ax.set_aspect('equal', adjustable='box')

plt.tight_layout()
plt.show()

# fig.savefig("plot_bess_disaggregation.pdf", bbox_inches="tight")
