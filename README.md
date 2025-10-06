# PyFlexAD - Python Flexibility Aggregation and Disaggregation

## About

The software package PyFlexAD is designed to apply the **vertex-based aggregation** of various energy storage devices 
for optimization purposes. 
The software is implemented in the Python 3 programming language and licensed under the MIT license. 
The python package is available for download from the Python Project Index (PyPI). 
The source code and additional materials can be accessed on [GitHub](https://github.com/erc-fhv/PyFlexAD).

## 📦 Installation

### From PyPI (recommended)

Once published, you can install the package directly from PyPI:
```bash
pip install pyflexad
```

### From GitHub (development or latest version)

You can also install the latest version directly from the GitHub repository:
```bash
pip install git+https://github.com/erc-fhv/PyFlexAD.git
```

### Optional solver extras
Install optional solver backends as needed:

- Gurobi (commercial, license required):
  ```bash
  pip install pyflexad[gurobi]
  ```
- HiGHS (open source):
  ```bash
  pip install pyflexad[highs]
  ```
- All optional solvers:
  ```bash
  pip install pyflexad[all-solvers]
  ```

Note: Gurobi requires a valid license to run. See https://www.gurobi.com/ for licensing details. HiGHS may require
platform-specific system packages on some environments.

### Verify installation
```bash
python -c "import pyflexad; print(pyflexad.__version__)"
```

### Dependencies

The latest version of PyFlexAD requires the installation of at least one mathematical programming solver, 
which is supported by the [Pyomo](http://www.pyomo.org/) optimization modeling library.

We recommend one of the following solvers:

- [CBC](https://projects.coin-or.org/Cbc): open source solver
- [HiGHs](https://highs.dev/): open source solver
- [GLPK](https://www.gnu.org/software/glpk/): open source solver
- [Gurobi](http://www.gurobi.com/): commercial solver, license required (academic license available)


## ▶️ Example usage

```python
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

ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
ax.set_aspect('equal', adjustable='box')

plt.tight_layout()
plt.show()
```

## Tutorials

The PyFlexAD package comes with several example scripts in the folder 
[./scripts](https://github.com/erc-fhv/pyflexad/scripts) on the GitHub repository.

The unit tests can be found in folder [./tests](https://github.com/erc-fhv/pyflexad/tests) on the GitHub repository.

## License

The PyFlexAD package is released by the [Energy Research Centre](https://www.fhv.at/en/research/energy) of the 
[University of Applied Sciences Vorarlberg](https://www.fhv.at/en) under the [MIT License](https://opensource.org/licenses/MIT).

## 📖 Related Literature

- [1] E. Öztürk, K. Kaspar, T. Faulwasser, K. Worthmann, P. Kepplinger, and K. Rheinberger, “Towards Efficient Aggregation of Storage Flexibilities in Power Grids,” Mar. 29, 2024, arXiv: arXiv:2403.20104. Accessed: Apr. 02, 2024. [Online]. Available: http://arxiv.org/abs/2403.20104
- [2] E. Öztürk, T. Faulwasser, K. Worthmann, M. Preißinger, and K. Rheinberger, “Alleviating the Curse of Dimensionality in Minkowski Sum Approximations of Storage Flexibility,” Feb. 28, 2024, arXiv: arXiv:2311.01614. Accessed: Mar. 08, 2024. [Online]. Available: http://arxiv.org/abs/2311.01614
- [3] E. Öztürk, K. Rheinberger, T. Faulwasser, K. Worthmann, and M. Preißinger, “Aggregation of Demand-Side Flexibilities: A Comparative Study of Approximation Algorithms,” Energies, vol. 15, no. 7, p. 2501, Mar. 2022, doi: 10.3390/en15072501.

## ℹ️ Further Information

### Energy Storage and Subclasses

The package supports multiple types of energy storage systems, including BESS’s, TCL’s, EV’s and PHES’s. 
These models extend the general EnergyStorage class. 
The package focuses on modeling energy storage systems and does not include energy resources without storage properties. 
Inflexible/non-steerable loads are either included as power demand of the devices (e.g. EV’s or TCLH’s) 
or must be managed by the optimizing entity alongside the energy storage systems. 
In order to calculate the general parameters of an energy storing system hardware and usage parameters must be provided. 
Each subclass of EnergyStorage requires a specific set of parameters 
e.g. to instantiate the class StationaryBattery instances of BESSHardware and BESSUsage are required. 
For ease of use, the package already includes a sample of hardware parameters for models from BESS manufacturers 
like Tesla and GENERAC, of EV manufactures like Tesla, Nissan and Renault for EV models, 
and generic air conditioning and water heaters for TCL models.

### Abstraction via Virtualization

A fundamental concept in the PyFlexAD package is the virtualization of energy storage systems. 
This involves abstracting individual physical energy storages into virtual representations, 
which encapsulate the essential characteristics of the physical devices. 
The virtualization process begins with the calculation of polytope extreme actions 
to delineate the feasible operation regions of energy storage systems. 
Polytope vertices of energy storages can be approximated effectively 
using the inner-approximation by vertex-generation algorithm (IABVG) and a set of direction vectors J. 
This approximation of vertices to extreme actions is crucial 
since the calculation of exact vertices becomes computationally intractable with increasing dimensions.
By utilizing extreme actions, the computational effort required for flexibility optimization is significantly reduced. 
Moreover, calculating the extreme actions can be parallelized for each energy storage device, 
further decreasing the computational load on the optimization entity.


### Flexibility Provision via Aggregation

To describe the collective flexibility of virtual energy storages, their corresponding extreme actions are summed, 
resulting in an aggregate virtual energy storage. 
This aggregated virtual energy storage can be controlled like a virtual power plant 
and enables the optimization entity to optimize power profiles across the entire system.