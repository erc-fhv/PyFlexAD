import matplotlib.pyplot as plt
import numpy as np

from pyflexad.math.polytope import Polytope


class VirtualEnergyStorage:
    def __init__(self, id: str, vertices: np.ndarray, load_profile: np.ndarray) -> None:
        self.__id = id
        self.__load_profile = load_profile
        self.__vertices = vertices

    def get_id(self) -> str:
        return self.__id

    def get_vertices(self) -> np.ndarray:
        return self.__vertices

    def get_load_profile(self) -> np.ndarray:
        return self.__load_profile

    def calculate_load_profile(self, alphas: np.ndarray):
        if alphas.shape[0] != self.__vertices.shape[0]:
            raise ValueError(f"Unable to calculate load profile. "
                             f"alphas.shape[0] = {alphas.shape[0]} != vertices.shape[0] = {self.__vertices.shape[0]}")
        self.__load_profile = self.__vertices.T @ alphas

    def plot_polytope_2d(self, ax: plt.axes = None, label: str = "", title: str = "", color: str = None,
                         marker: str = "none",
                         line_style: str = '-', fill: bool = False, hatch: str = None, zorder: int = None) -> None:
        polytope = Polytope(self.__vertices)

        if not ax:
            ax = plt.gca()

        polytope.plot2d(ax, label=label, title=title, color=color, marker=marker, line_style=line_style, fill=fill,
                        hatch=hatch, zorder=zorder)

    def plot_load_profile_2d(self, ax: plt.axes = None, label: str = "", color: str = None, marker: str = "o",
                             edgecolors: str | None = "k",
                             s: int = 100, zorder: int = 3) -> None:
        ax.scatter(self.__load_profile[0], self.__load_profile[1], color=color, marker=marker, label=label, s=s,
                   edgecolors=edgecolors, zorder=zorder)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.__id}, load_profile={self.__load_profile})"
