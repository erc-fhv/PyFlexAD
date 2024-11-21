import matplotlib.pyplot as plt
import numpy as np

from pyflexad.math.signal_vectors import SignalVectors
# import pyflexad.models.bess.tesla as tesla_bess
from pyflexad.models.bess import bess_models
from pyflexad.physical.stationary_battery import StationaryBattery
from pyflexad.utils.algorithms import Algorithms
from pyflexad.utils.network_graph import NetworkGraph
from pyflexad.virtual.aggregator import Aggregator


def main() -> None:
    """settings"""
    d = 2
    dt = 0.25
    n_entities = 5

    """main"""
    np.random.seed(2)

    agg_list = []
    signal_vectors = SignalVectors.new(d, g=SignalVectors.g_of_2_d_10(d))
    for index, bess_model in enumerate(bess_models):
        esr_list_1 = [StationaryBattery.random_usage(hardware=bess_model, d=d, dt=dt, id=f"{bess_model.name}[{i + 1}]")
                      for i in range(n_entities)]
        agg_list += [
            Aggregator.from_physical(
                items=esr_list_1, algorithm=Algorithms.IABVG, signal_vectors=signal_vectors,
                id=f"Sub-Aggregator[{index+1}]")
        ]

    agg = Aggregator.aggregate(agg_list, id="Aggregator", algorithm=Algorithms.IABVG)

    """plot network graph"""
    graph = NetworkGraph.from_virtual([agg, ])

    graph.create_tree()

    _, ax = plt.subplots(figsize=(10, 10))
    # graph.plot_tree(ax=ax, layout="planar", node_size=100, parent_node_size=2000)
    graph.plot_tree(ax=ax, node_size=100, parent_node_size=500)
    ax.set_title('Network Graph: BESS')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
