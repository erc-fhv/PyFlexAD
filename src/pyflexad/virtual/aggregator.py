from typing import Self

import numpy as np

from pyflexad.math.polytope import Polytope
from pyflexad.math.signal_vectors import SignalVectors
from pyflexad.physical.energy_storage import EnergyStorage
from pyflexad.physical.energy_storage import virtualized
from pyflexad.utils.algorithms import Algorithms
from pyflexad.virtual.virtual_energy_storage import VirtualEnergyStorage


class Aggregator(VirtualEnergyStorage):

    def __init__(self, id: str, vertices: np.ndarray, items: list[VirtualEnergyStorage], load_profile: np.ndarray) \
            -> None:
        super().__init__(id=id, vertices=vertices, load_profile=load_profile)
        self.__items = items

    def get_items(self) -> list[VirtualEnergyStorage]:
        return self.__items

    @classmethod
    def from_physical(cls, items: list[EnergyStorage], algorithm: Algorithms, signal_vectors: SignalVectors = None,
                      id: str = None) -> Self:
        virtual = virtualized(items=items, algorithm=algorithm, signal_vectors=signal_vectors)
        return cls.aggregate(items=virtual, algorithm=algorithm, id=id)

    @classmethod
    def aggregate(cls, items: list[VirtualEnergyStorage], algorithm: Algorithms, id: str = None) -> Self:
        if id is None:
            id = EnergyStorage.create_id()

        if algorithm == Algorithms.EXACT:
            """exact minkowski sum"""
            polytopes = [Polytope(item.get_vertices()) for item in items]
            vertices = Polytope.sum(polytopes).V_sorted()
        elif algorithm in Algorithms.get_aggregation_algorithms():
            vertices = np.sum([item.get_vertices() for item in items], axis=0)
        else:
            raise Exception(f"algorithm {algorithm} is not supported")

        d = vertices.shape[1]

        load_profile = EnergyStorage.empty_load_profile(d=d)
        return cls(id=id, vertices=vertices, items=items, load_profile=load_profile)

    def disaggregate(self, alphas: np.ndarray) -> None:
        self.calculate_load_profile(alphas)
        for item in self.__items:
            item.calculate_load_profile(alphas)
