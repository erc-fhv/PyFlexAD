import numpy as np
import pytest

import pyflexad.models.tclc.generic as tclc_generic
from pyflexad.math.signal_vectors import SignalVectors
from pyflexad.physical.therm_load_cooling import TCLCUsage
from pyflexad.physical.therm_load_cooling import ThermostaticLoadCooling
from pyflexad.utils.algorithms import Algorithms


class TestThermostaticLoadCooling:
    testdata_calc_vertices = [
        (2, 2, Algorithms.EXACT, np.array([[0., 0.], [5., 0.], [5., 3.703], [3.619, 5.], [0., 5.]])),
        (2, 2, Algorithms.LPVG_GUROBIPY, np.array([[0., 0.], [0., 5.], [5., 0.], [5., 3.703]])),
        (2, 2, Algorithms.IABVG, np.array([[0., 0.], [0., 5.], [5., 0.], [5., 3.703]])),
        (2, 2, Algorithms.IABVG_JIT, np.array([[0., 0.], [0., 5.], [5., 0.], [5., 3.703]])),
    ]

    @pytest.mark.parametrize("seed, d, algorithm, expected", testdata_calc_vertices)
    def test_calc_vertices(self, seed: int, d, algorithm, expected) -> None:
        if not algorithm.is_solver_available():
            pytest.skip(f"pyomo solver {algorithm.get_solver()} not available")

        """settings"""
        dt = 0.25

        """main"""
        np.random.seed(seed)
        signal_vectors = SignalVectors.new(d)

        usage_params = TCLCUsage(theta_r=20, theta_a=30, theta_0=np.random.uniform(19, 21),
                                 delta=np.random.uniform(1.5, 2.5), d=d, dt=dt)
        physical = ThermostaticLoadCooling.new(hardware=tclc_generic.air_conditioner_1, usage=usage_params)

        virtual = physical.to_virtual(signal_vectors=signal_vectors, algorithm=algorithm)

        assert np.allclose(np.sort(virtual.get_vertices(), axis=0), np.sort(expected, axis=0), atol=1e-3)
