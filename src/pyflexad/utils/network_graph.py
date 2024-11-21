from dataclasses import dataclass, field

import networkx as nx
from typing_extensions import Self

from pyflexad.virtual.virtual_energy_storage import VirtualEnergyStorage
from pyflexad.virtual.aggregator import Aggregator


@dataclass
class TreeNode:
    id: str
    label: str
    children: list = field(default_factory=list)


class NetworkGraph:

    @classmethod
    def from_virtual(cls, items: list[VirtualEnergyStorage]) -> Self:
        nodes = []
        graph = nx.Graph()
        return cls(items=items, nodes=nodes, graph=graph)

    def __init__(self, items: list[VirtualEnergyStorage], nodes: list[TreeNode], graph: nx.Graph) -> None:
        self.__items = items
        self.__nodes = nodes
        self.__graph = graph

    def create_tree(self) -> None:
        for item in self.__items:
            node = TreeNode(item.get_id(), item.get_id())
            self.__nodes.append(node)
            self.__update_nodes_recursive(node, item)

        for node in self.__nodes:
            self.__graph.add_node(node.id, label=node.label)
            self.__update_tree_recursive(node)

    def __update_nodes_recursive(self, parent_node: TreeNode, item: VirtualEnergyStorage) -> None:
        if isinstance(item, Aggregator):
            for child_item in item.get_items():
                child_node = TreeNode(child_item.get_id(), child_item.get_id())
                parent_node.children.append(child_node)
                self.__update_nodes_recursive(child_node, child_item)

    def __update_tree_recursive(self, node: TreeNode) -> None:
        for child in node.children:
            self.__graph.add_node(child.id, label=child.label)
            self.__graph.add_edge(node.id, child.id)
            self.__update_tree_recursive(child)

    def plot_tree(self, ax=None, layout: str = 'kamada_kawai', with_labels: bool = True,
                  node_size: int = 2000, parent_node_size: int = None,
                  default_node_color: str = 'skyblue', parent_node_color: str = 'red', font_size: int = 10):
        labels = nx.get_node_attributes(self.__graph, 'label')

        if not parent_node_size:
            parent_node_size = node_size

        if layout == "spring":
            pos = nx.spring_layout(self.__graph)
        elif layout == "planar":
            pos = nx.planar_layout(self.__graph)
        elif layout == "kamada_kawai":
            pos = nx.kamada_kawai_layout(self.__graph)
        else:
            raise ValueError(f"layout {layout} is not supported")

        has_children = [1 if self.__graph.degree[node] > 1 else 0 for node in self.__graph.nodes]

        """change color if parent node"""
        node_colors = [parent_node_color if has_children[i] == 1 else default_node_color for i in
                       range(len(has_children))]

        """change size if parent node"""
        node_size = [parent_node_size if has_children[i] == 1 else node_size for i in
                     range(len(has_children))]

        nx.draw_networkx(self.__graph, pos, labels=labels, with_labels=with_labels, node_size=node_size,
                         node_color=node_colors, font_size=font_size, ax=ax)
