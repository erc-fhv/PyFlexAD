from typing import Self

import cdd
import numpy as np
from scipy.spatial import ConvexHull


class Polytope:

    @classmethod
    def sum(cls, polytopes: list[Self]) -> Self:
        """
        Calculate the sum of a list of polytopes.

        Parameters
        ----------
        polytopes : list[Self]
            A list of polytopes to sum.

        Returns
        -------
        Self
            The resulting polytope after the summation operation.
        """
        return np.sum(polytopes, axis=0)

    @staticmethod
    def calc_vertices_from_A_b(A: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Calculate the vertices of a polytope given the coefficients matrix A and the constant vector b.

        Parameters:
        A: np.ndarray
            The coefficients matrix of the polytope.
        b: np.ndarray
            The constant vector of the polytope.

        Returns
        -------
        np.ndarray: The vertices of the polytope.

        Raises
        ------
        ValueError: If the number of rows in A is not equal to the number of rows in b.
        ValueError: If the polytope is empty.

        Note
        ----
        The polytope is represented by the set of points that satisfy the inequalities defined
        by the coefficients matrix A and the constant vector b.
        The vertices of the polytope are the points that satisfy all the inequalities.

        Example:
        >>> A = np.array([[1, 2], [3, 4]])
        >>> b = np.array([5, 6])
        >>> calc_vertices_from_A_b(A, b)
        array([[1. , 1. ],
               [0. , 0.5],
               [0.5, 0. ]])
        """

        A = np.array(A, ndmin=2)
        b = np.squeeze(b)[:, np.newaxis]

        if b.shape[0] != A.shape[0]:
            raise ValueError(f'A has {A.shape[0]} rows; b has {b.shape[0]} rows!')

        b_mA = np.hstack((b, -A))  # [b, -A]
        H = cdd.Matrix(b_mA, number_type='float')
        H.rep_type = cdd.RepType.INEQUALITY
        H_P = cdd.Polyhedron(H)

        P_tV: cdd.Matrix = H_P.get_generators()
        tV = np.array(P_tV[:])

        if tV.any():
            """check if Polytope is empty"""
            indices = tV[:, 0] == 1
            vertices = tV[indices, 1:]
        else:
            raise ValueError('The Polytope is empty!')
        return vertices

    @classmethod
    def from_A_b(cls, A: np.ndarray, b: np.ndarray) -> Self:
        """
        Create a Polytope object from the coefficients matrix A and the constant vector b.

        Parameters
        ----------
        A: np.ndarray
            The coefficients matrix of the polytope.
        b: np.ndarray
            The constant vector of the polytope.

        Returns
        -------
        Polytope
            The created Polytope object.
        """
        vertices = cls.calc_vertices_from_A_b(A, b)
        return cls(vertices=vertices)

    def __init__(self, vertices: np.ndarray) -> None:
        """
        Initialize the Polytope object.

        Parameters
        ----------
        vertices: np.ndarray
            The vertices of the polytope.
        """
        self._V = vertices

    @property
    def V(self) -> np.ndarray:
        """
        Get the vertices of the polytope.

        Returns
        -------
        np.ndarray
            The vertices of the polytope.
        """
        return self._V

    @property
    def nV(self) -> int:
        """
        Get the number of vertices of the polytope.

        Returns
        -------
        int
            The number of vertices of the polytope.
        """
        return self.V.shape[0]

    @property
    def n(self) -> int:
        """
        Get the number of dimensions of the polytope.

        Returns
        -------
        int
            The number of dimensions of the polytope.
        """
        return self.V.shape[1]

    @property
    def centroid(self) -> np.ndarray:
        """
        Compute the centroid of the polytope.

        Returns
        -------
        np.ndarray
            The centroid of the polytope.
        """
        return np.sum(self.V, axis=0) / self.nV

    def V_sorted(self) -> np.ndarray:
        """
        Sort vertices (increasing angle: the point (x1, x2) = (1, 0) has angle 0).
        np.arctan2(y, x) returns angles in the range [-pi, pi].

        Vertices are sorted clockwise.
        Mainly for plotting and not implemented for n != 2.

        Returns
        -------
        np.ndarray
        """

        if self.n != 2:
            raise NotImplementedError('V_sorted() not implemented for n != 2')

        c = self.centroid
        order = np.argsort(np.arctan2(self.V[:, 1] - c[1], self.V[:, 0] - c[0]))
        return self.V[order, :]

    def minimize_vertices(self) -> None:
        """
        Minimize the number of vertices used to represent the polytope by removing
        redundant points from the vertex list.
        Indices of the unique vertices forming the convex hull
        """

        i_v_minimal = ConvexHull(self.V).vertices
        self._V = self.V[i_v_minimal, :]

    def minkowski_sum(self, other: Self, minimize_vertices: bool = True) -> Self:
        """
        Minkowski sum of two convex polytopes P and Q:
        P + Q = {p + q in R^n : p in P, q in Q}.
        In vertex representation, this is the convex hull of the pairwise sum of all
        combinations of points in P and Q.

        Parameters
        ----------
        other : Polytope
        minimize_vertices: bool

        Returns
        -------
        m_sum : Polytope
        """

        # Vectorized pairwise addition of vertices using broadcasting
        m_sum_vertices = (self.V[None, :, :] + other.V[:, None, :]).reshape(-1, self.n)

        m_sum = Polytope(m_sum_vertices)

        if minimize_vertices:
            m_sum.minimize_vertices()
        return m_sum

    def plot2d(self, ax=None, label: str = "", title: str = "", color: str = None, marker: str = "none",
               line_style: str = '-', line_width: int = None, fill: bool = False, hatch: str = None,
               zorder: int = None):
        """
        Plot a 2D representation of the polytope.

        Parameters
        ----------
        ax : plt.axes, optional
            The axes object on which to plot. If None, a new axes object is created.
        label : str, optional
            The label for the plot.
        title : str, optional
            The title for the plot.
        color : str, optional
            The color of the plot.
        marker : str, optional
            The marker style for the plot.
        line_style : str, optional
            The line style for the plot.
        line_width: int, optional
            The line width for the plot.
        fill : bool, optional
            Whether to fill the plot.
        hatch : str, optional
            The hatch style for the plot.
        zorder : int, optional
            The z-order of the plot.

        Raises
        ------
        NotImplementedError
            If the number of dimensions is not equal to 2.

        Returns
        -------
        None
        """
        if self.n != 2:
            raise NotImplementedError('plot2d() is not implemented for n != 2')

        # Import matplotlib lazily to avoid heavy import cost at module import time
        import matplotlib.pyplot as plt

        vertices = self.V_sorted()

        x_list = vertices[:, 0].tolist()
        x_list.append(vertices[0, 0])
        y_list = vertices[:, 1].tolist()
        y_list.append(vertices[0, 1])

        if not ax:
            ax = plt.gca()

        if fill and hatch is None:
            ax.fill(x_list, y_list, color=color, label=label, zorder=zorder)
        elif fill and hatch is not None:
            ax.fill(x_list, y_list, edgecolor=color, label=label, hatch=hatch, facecolor='none', zorder=zorder)
        else:
            ax.plot(x_list, y_list, color=color, marker=marker, label=label, linestyle=line_style,
                    linewidth=line_width, zorder=zorder)

        ax.grid(True)
        if title != "":
            ax.set_title(title)
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")

    def __add__(self, other) -> Self:
        """
        Add two polytopes using the Minkowski sum operation.

        Parameters
        ----------
        other : Polytope
            The other polytope to add.

        Returns
        -------
        Self
            The resulting polytope after the addition operation.
        """
        return self.minkowski_sum(other)
