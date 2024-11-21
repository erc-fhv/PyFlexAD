import numpy as np
import pytest

import pyflexad.models.tclh.generic as tclh_generic
from pyflexad.math.signal_vectors import SignalVectors
from pyflexad.physical.therm_load_heating import TCLHUsage
from pyflexad.physical.therm_load_heating import ThermostaticLoadHeating
from pyflexad.utils.algorithms import Algorithms


class TestThermostaticLoadCooling:
    testdata_calc_vertices = [
        (2, 2, Algorithms.EXACT, np.array([[0., 0.], [0.596, 0.], [0.596, 0.049], [0., 0.645]])),
        (2, 2, Algorithms.LPVG_GUROBIPY, np.array([[0., 0.], [0., 0.645], [0.596, 0.], [0.596, 0.049]])),
        (2, 2, Algorithms.IABVG, np.array([[0., 0.], [0., 0.645], [0.596, 0.], [0.596, 0.049]])),
        (2, 2, Algorithms.IABVG_JIT, np.array([[0., 0.], [0., 0.645], [0.596, 0.], [0.596, 0.049]])),
    ]

    @pytest.mark.parametrize("seed, d, algorithm, expected", testdata_calc_vertices)
    def test_calc_vertices(self, seed: int, d, algorithm, expected) -> None:
        if not algorithm.is_solver_available():
            pytest.skip(f"pyomo solver {algorithm.get_solver()} not available")

        """settings"""
        dt = 0.25

        """main"""
        np.random.seed(seed)
        demand = np.random.uniform(0, 4, d)
        signal_vectors = SignalVectors.new(d)

        usage = TCLHUsage(theta_r=50, theta_a=20, theta_0=55, delta=5, demand=demand, d=d, dt=dt)
        physical = ThermostaticLoadHeating.new(hardware=tclh_generic.domestic_hot_water_heater_1, usage=usage)
        virtual = physical.to_virtual(signal_vectors=signal_vectors, algorithm=algorithm)

        assert np.allclose(np.sort(virtual.get_vertices(), axis=0), np.sort(expected, axis=0), atol=1e-3)
