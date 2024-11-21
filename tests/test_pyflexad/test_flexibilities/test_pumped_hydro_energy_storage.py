import numpy as np
import pytest

from pyflexad.math.signal_vectors import SignalVectors
from pyflexad.physical.pumped_hydro_energy_storage import PHESHardware
from pyflexad.physical.pumped_hydro_energy_storage import PHESUsage
from pyflexad.physical.pumped_hydro_energy_storage import PumpedHydroEnergyStorage
from pyflexad.utils.algorithms import Algorithms


class TestPumpedHydroEnergyStorage:
    testdata_calc_vertices = [
        (2, 2, Algorithms.EXACT, np.array([[5., 5.9], [5.9, 5.], [20., 5.], [20., 20.], [5., 20.]])),
        (2, 2, Algorithms.LPVG_GUROBIPY, np.array([[5.9, 5.], [5., 20.], [20., 5.], [20., 20.]])),
        (2, 2, Algorithms.IABVG, np.array([[5., 5.9], [5., 20.], [20., 5.], [20., 20.]])),
        (2, 2, Algorithms.IABVG_JIT, np.array([[5., 5.9], [5., 20.], [20., 5.], [20., 20.]])),
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

        hardware = PHESHardware(
            name="test",
            max_discharging_power=5,
            max_charging_power=20,
            max_volume=1000,
            min_volume=100,
            delta_h=10000
        )
        usage = PHESUsage(initial_volume=hardware.min_volume, final_volume=hardware.min_volume * 2, d=d, dt=dt)

        physical = PumpedHydroEnergyStorage.new(hardware=hardware, usage=usage)
        virtual = physical.to_virtual(signal_vectors=signal_vectors, algorithm=algorithm)

        assert np.allclose(np.sort(virtual.get_vertices(), axis=0), np.sort(expected, axis=0), atol=1e-3)
