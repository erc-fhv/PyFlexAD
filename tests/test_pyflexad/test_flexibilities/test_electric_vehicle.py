import numpy as np
import pytest

import pyflexad.models.ev.tesla as tesla_ev
from pyflexad.math.signal_vectors import SignalVectors
from pyflexad.physical.electric_vehicle import ElectricVehicle
from pyflexad.utils.algorithms import Algorithms


class TestElectricVehicle:
    testdata_calc_vertices_1 = [
        (2, 2, Algorithms.LPVG_GUROBIPY, np.array([[-11., -11.], [-11., 11.], [11., -11.], [11., 11.]])),
        (2, 2, Algorithms.IABVG, np.array([[-11., -11.], [-11., 11.], [11., -11.], [11., 11.]])),
        (2, 2, Algorithms.IABVG_JIT, np.array([[-11., -11.], [-11., 11.], [11., -11.], [11., 11.]]))
    ]

    testdata_calc_vertices_2 = [
        (2, 2, Algorithms.LPVG_GUROBIPY),
        (2, 2, Algorithms.IABVG),
        (2, 2, Algorithms.IABVG_JIT)
    ]

    @pytest.mark.parametrize("seed, d, algorithm, expected", testdata_calc_vertices_1)
    def test_calc_vertices_1(self, seed: int, d, algorithm, expected) -> None:
        if not algorithm.is_solver_available():
            pytest.skip(f"pyomo solver {algorithm.get_solver()} not available")

        """settings"""
        dt = 0.25

        """main"""
        np.random.seed(seed)
        signal_vectors = SignalVectors.new(d)

        physical = ElectricVehicle.random_usage(hardware=tesla_ev.model_3_rwd_11kw_ac, d=d, dt=dt)
        virtual = physical.to_virtual(signal_vectors=signal_vectors, algorithm=algorithm)

        assert np.allclose(virtual.get_vertices(), expected)

    @pytest.mark.parametrize("seed, d, algorithm", testdata_calc_vertices_2)
    def test_calc_vertices_2(self, seed: int, d, algorithm) -> None:
        if not algorithm.is_solver_available():
            pytest.skip(f"pyomo solver {algorithm.get_solver()} not available")

        """settings"""
        dt = 0.25

        """main"""
        np.random.seed(seed)
        signal_vectors = SignalVectors.new(d)

        physical = ElectricVehicle.random_usage(hardware=tesla_ev.model_3_rwd_11kw_ac, d=d, dt=dt)
        virtual = physical.to_virtual(signal_vectors=signal_vectors, algorithm=algorithm)

        """exact method"""
        virtual_exact = physical.to_virtual(algorithm=Algorithms.EXACT)

        assert np.allclose(np.sort(virtual.get_vertices(), axis=0), np.sort(virtual_exact.get_vertices(), axis=0))
