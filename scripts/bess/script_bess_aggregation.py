import matplotlib.pyplot as plt

import pyflexad.models.bess.tesla as tesla_bess
from pyflexad.math.signal_vectors import SignalVectors
from pyflexad.physical.stationary_battery import StationaryBattery, BESSUsage
from pyflexad.utils.algorithms import Algorithms


def main() -> None:
    """settings"""
    d = 2
    dt = 0.25

    """main"""
    usage_params = BESSUsage(initial_capacity=7.0, final_capacity=5.0, d=d, dt=dt)
    # usage_params = BESSUsage(initial_capacity=10.0, final_capacity=5.0, d=d, dt=dt)

    esr = StationaryBattery.new(hardware=tesla_bess.power_wall_2, usage=usage_params)

    """polytope calculation"""
    signal_vectors = SignalVectors.new(d, g=SignalVectors.g_of_2_d_10(d))

    v_exact = esr.to_virtual(algorithm=Algorithms.EXACT)
    v_iabvg = esr.to_virtual(algorithm=Algorithms.LPVG_GUROBIPY, signal_vectors=signal_vectors)
    v_iabvgx = esr.to_virtual(algorithm=Algorithms.IABVG, signal_vectors=signal_vectors)

    """plot polytopes"""
    _, ax = plt.subplots(1, 1)
    v_exact.plot_polytope_2d(ax, label=Algorithms.EXACT, color='r')
    v_iabvg.plot_polytope_2d(ax, label=Algorithms.LPVG_GUROBIPY, fill=True, color='g', hatch='\\')
    v_iabvgx.plot_polytope_2d(ax, label=Algorithms.IABVG, fill=True, color='b', hatch='//')

    ax.set_title('BESS')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
