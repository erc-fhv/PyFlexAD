import matplotlib.pyplot as plt

import pyflexad.models.bess.tesla as tesla
from pyflexad.math.signal_vectors import SignalVectors
from pyflexad.physical.stationary_battery import BESSUsage
from pyflexad.physical.stationary_battery import StationaryBattery
from pyflexad.utils.algorithms import Algorithms
from pyflexad.virtual.aggregator import Aggregator

"""settings"""
d = 2  # number of time intervals
dt = 0.25  # interval duration in hours
algorithm = Algorithms.IABVG  # virtualization algorithm
S_0 = 6.5  # initial battery capacity in kWh
S_f = 5.0  # final battery capacity in kWh

"""instantiate energy storage resources"""
usage_1 = BESSUsage(initial_capacity=13, final_capacity=11, d=d, dt=dt)
usage_2 = BESSUsage(initial_capacity=S_0, final_capacity=5.5, d=d, dt=dt)

bess_1 = StationaryBattery.new(hardware=tesla.power_wall_2, usage=usage_1)
bess_2 = StationaryBattery.new(hardware=tesla.power_wall_plus, usage=usage_2)

"""virtualize"""
direction_vectors = SignalVectors.new(d)
virtual_ess_1 = bess_1.to_virtual(algorithm, direction_vectors)
virtual_ess_2 = bess_2.to_virtual(algorithm, direction_vectors)

"""aggregate"""
aggregator = Aggregator.aggregate([virtual_ess_1, virtual_ess_2], algorithm)

"""virtualize exact"""
virtual_ess_ex_1 = bess_1.to_virtual(Algorithms.EXACT)
virtual_ess_ex_2 = bess_2.to_virtual(Algorithms.EXACT)

"""aggregate exact"""
aggregator_ex = Aggregator.aggregate([virtual_ess_ex_1, virtual_ess_ex_2], Algorithms.EXACT)

# %% plotting
s = 400
fontsize = 20
line_width = 4
# marker_size = 10

"""plot polytopes"""
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 20), squeeze=True)

"""axis 1"""
aggregator_ex.plot_polytope_2d(ax, label="$Exact$ $Aggregate$ $Polytope$", color='k', line_style='-.',
                               line_width=line_width)
virtual_ess_ex_1.plot_polytope_2d(ax, label="$Exact$ $Polytope$ $1$", color='tab:green', line_style='-',
                                  line_width=line_width)
virtual_ess_ex_2.plot_polytope_2d(ax, label="$Exact$ $Polytope$ $2$", color='blue', line_style='--',
                                  line_width=line_width)

v1 = virtual_ess_1.get_vertices()
v2 = virtual_ess_2.get_vertices()
v3 = aggregator.get_vertices()

"""axis 2"""
aggregator.plot_polytope_2d(ax, label="$Approx.$ $Aggregate$ $Polytope$", color=('darkorange', 0.5), line_style='-.',
                            line_width=line_width, fill=True)
virtual_ess_1.plot_polytope_2d(ax, label="$Approx.$ $Polytope$ $1$", color=('tab:green', 0.5), line_style='-',
                               line_width=line_width, fill=True)
virtual_ess_2.plot_polytope_2d(ax, label="$Approx.$ $Polytope$ $2$", color='blue', line_style='--',
                               line_width=line_width, fill=True, hatch="//"
                               )

ax.scatter(v3[:, 0], v3[:, 1], marker='p', color='k', label='$Extreme$ $Actions:$ $Aggregate$ $Polytope$',
           zorder=10, s=s)
ax.scatter(v1[:, 0], v1[:, 1], marker='o', color='tab:green', edgecolor="k",
           label='$Extreme$ $Actions:$ $Polytope$ $1$', zorder=10,
           s=s)
ax.scatter(v2[:, 0], v2[:, 1], marker='^', color='blue', label='$Extreme$ $Actions:$ $Polytope$ $2$', zorder=10, s=s)

_fontsize = fontsize + 10
ax.annotate("$v^{(1, 1)}$", xy=(v3[3, 0] + 1, v3[3, 1]), size=_fontsize)
ax.annotate("$v^{(1, -1)}$", xy=(v3[2, 0] + 1, v3[2, 1]), size=_fontsize)
ax.annotate("$v^{(-1, 1)}$", xy=(v3[1, 0] - 3.2, v3[1, 1]), size=_fontsize)
ax.annotate("$v^{(-1, -1)}$", xy=(v3[0, 0] - 3.2, v3[0, 1]), size=_fontsize)

dy1 = 2
dy2 = -2
head_offset = 0.2

ax.annotate("$y_1^{(1, 1)}$", xy=(10, v1[3, 1] + dy1), size=_fontsize, color="tab:green", zorder=15)
ax.annotate("", xytext=(10, v1[3, 1] + dy1), xy=(v1[3, 0] + head_offset, v1[3, 1]),
            arrowprops=dict(facecolor='tab:green', width=5))

ax.annotate("$y_2^{(1, 1)}$", xy=(10, v2[3, 1] + dy2), size=_fontsize, color="blue", zorder=15)
ax.annotate("", xytext=(10, v2[3, 1] + dy2), xy=(v2[3, 0] + head_offset, v2[3, 1]),
            arrowprops=dict(facecolor='blue', width=5))

ax.grid(True)
ax.tick_params(axis='both', which='major', labelsize=fontsize)
ax.tick_params(axis='both', which='minor', labelsize=fontsize)
ax.set_xlabel("$x_1$", fontsize=fontsize)
ax.set_ylabel("$x_2$", fontsize=fontsize)
ax.set_aspect('equal', adjustable='box')
ax.scatter(0, 0, marker='+', color='k', zorder=10, s=s)
ax.set_xlim(-15, 15)
ax.set_ylim(-15, 15)
ax.legend(loc='upper left', fontsize=fontsize)
ax.set_aspect('equal', adjustable='box')
ax.legend(loc='lower left', fontsize=fontsize, framealpha=0.2,
          # fancybox=True, shadow=True, frameon=True, edgecolor="grey"
          )

# plt.tight_layout()
plt.show()
