import functools
from typing import Self

import numpy as np

from pyflexad.math.signal_vectors import SignalVectors
from pyflexad.parameters.general_parameters import GeneralParameters


class IABVG:
    """Inner approximation by vertex generation algorithm"""
    tol = 10 ** (-15)  # correction # (- 10**(-15) due to numerical issues)

    @classmethod
    def from_general_params(cls, general_params: GeneralParameters) -> Self:
        return cls(**general_params.__dict__)

    def __init__(self, x_lower: np.ndarray, x_upper: np.ndarray,
                 s_lower: np.ndarray, s_upper: np.ndarray,
                 s_initial: float,
                 alpha: float,
                 d: int,
                 dt: float,
                 x0: float) -> None:
        self.__x_lower = x_lower
        self.__x_upper = x_upper
        self.__s_lower = s_lower
        self.__s_upper = s_upper
        self.__s_initial = s_initial
        self.__alpha = alpha
        self.__d = d
        self.__dt = dt
        self.__x0 = x0

    def approx_vertices(self, signal_vectors: SignalVectors) -> np.ndarray:
        x_list = []
        for item in signal_vectors.signals:
            s_list = [self.__s_initial]
            x_elem_list = []
            for elem, t in zip(item, range(self.__d)):
                if elem == 1:
                    x_opt, s_t = self.get_x_max_alpha(s_list[-1], t_s=t, t_x=t)
                    x_elem_list.append(x_opt)
                    s_list.append(s_t)
                    x_elem_list = self.correction_down(s_t, s_list, x_elem_list, t)
                    x_elem_list = self.correction_up(s_t, s_list, x_elem_list, t)
                elif elem == -1:
                    x_opt, s_t = self.get_x_min_alpha(s_list[-1], t_s=t, t_x=t)
                    x_elem_list.append(x_opt)
                    s_list.append(s_t)
                    x_elem_list = self.correction_down(s_t, s_list, x_elem_list, t)
                    x_elem_list = self.correction_up(s_t, s_list, x_elem_list, t)
            x_list.append(np.array(x_elem_list))
        return np.array(x_list) + self.__x0

    @functools.cache
    def get_x_max_alpha(self, s_initial_t: float, t_s: int, t_x: int) -> tuple[float, float]:
        s_upper_t = self.__s_upper[t_s]
        x_upper_t = self.__x_upper[t_x]
        x_lower_t = self.__x_lower[t_x]

        x_max = max(min((s_upper_t - self.__alpha * s_initial_t) / self.__dt, x_upper_t), x_lower_t)
        s_t = self.__alpha * s_initial_t + x_max * self.__dt
        return x_max, s_t

    @functools.cache
    def get_x_min_alpha(self, s_initial_t: float, t_s: int, t_x: int) -> tuple[float, float]:
        s_lower_t = self.__s_lower[t_s]
        x_upper_t = self.__x_upper[t_x]
        x_lower_t = self.__x_lower[t_x]

        x_min = min(max((s_lower_t - self.__alpha * s_initial_t) / self.__dt, x_lower_t), x_upper_t)
        s_t = self.__alpha * s_initial_t + x_min * self.__dt
        return x_min, s_t

    def correction_down(self, s_t: float, s_list: list, x_elem_list: list, t: int) -> list:
        if s_t > self.__s_upper[t] + self.tol:
            aux = self.__x_lower[:t + 1] == 0
            k = list(aux)[::-1].index(False) + 1
            i = k
            while s_t > self.__s_lower[t] + self.tol:
                x_elem_list_temp = x_elem_list.copy()
                s_list_temp = s_list.copy()
                del x_elem_list_temp[-i:]
                del s_list_temp[-i:]
                s_t = s_list_temp[-1]
                for j in range(i, k, -1):
                    x_opt, s_t = self.get_x_min_alpha(s_t, t_s=-j, t_x=-j)
                    x_elem_list_temp.append(x_opt)
                    s_list_temp.append(s_t)
                x_opt, s_t = self.get_x_max_alpha(s_t, t_s=t, t_x=-k)
                x_elem_list_temp.append(x_opt)
                s_list_temp.append(s_t)
                i += 1
            x_elem_list = x_elem_list_temp  # FIXME x_elem_list_temp might be referenced before assignment
            x_elem_list += (k - 1) * [0]
        return x_elem_list

    def correction_up(self, s_t: float, s_list: list, x_elem_list: list, t: int) -> list:
        if s_t < self.__s_lower[t] - self.tol:
            aux = self.__x_upper[:t + 1] == 0
            k = list(aux)[::-1].index(False) + 1
            i = k
            while s_t < self.__s_lower[t] - self.tol:
                x_elem_list_temp = x_elem_list.copy()
                s_list_temp = s_list.copy()
                del x_elem_list_temp[-i:]
                del s_list_temp[-i:]
                s_t = s_list_temp[-1]
                for j in range(i, k, -1):
                    x_opt, s_t = self.get_x_max_alpha(s_t, t_s=-j, t_x=-j)
                    x_elem_list_temp.append(x_opt)
                    s_list_temp.append(s_t)
                x_opt, s_t = self.get_x_min_alpha(s_t, t_s=t, t_x=-k)
                x_elem_list_temp.append(x_opt)
                s_list_temp.append(s_t)
                i += 1
            x_elem_list = x_elem_list_temp  # FIXME x_elem_list_temp might be referenced before assignment
            x_elem_list += (k - 1) * [0]
        return x_elem_list
