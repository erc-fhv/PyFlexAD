from typing import Self

import numba as nb
import numpy as np

from pyflexad.math.signal_vectors import SignalVectors
from pyflexad.parameters.general_parameters import GeneralParameters

nb.config.DISABLE_JIT = False
NO_PYTHON = True
CACHE = True
NO_GIL = True
FASTMATH = True
USE_SIGNATURE = True


@nb.jit(["float64(float64, float64, float64, float64, float64, float64)"] if USE_SIGNATURE else None,
        nopython=NO_PYTHON, cache=CACHE, nogil=NO_GIL, fastmath=FASTMATH)
def get_x_max_alpha(s_initial_t: float, s_upper_t: float,
                    x_upper_t: float, x_lower_t: float,
                    alpha: float, dt: float) -> float:
    x_max = max(min((s_upper_t - alpha * s_initial_t) / dt, x_upper_t), x_lower_t)
    return x_max


@nb.jit(["float64(float64, float64, float64, float64, float64, float64)"] if USE_SIGNATURE else None,
        nopython=NO_PYTHON, cache=CACHE, nogil=NO_GIL, fastmath=FASTMATH)
def get_x_min_alpha(s_initial_t: float, s_lower_t: float,
                    x_upper_t: float, x_lower_t: float,
                    alpha: float, dt: float) -> float:
    x_min = min(max((s_lower_t - alpha * s_initial_t) / dt, x_lower_t), x_upper_t)
    return x_min


@nb.jit(["float64(float64, float64, float64, float64)"] if USE_SIGNATURE else None,
        nopython=NO_PYTHON, cache=CACHE, nogil=NO_GIL, fastmath=FASTMATH)
def get_s_t(s_initial_t: float, x: float, alpha: float, dt: float) -> float:
    s_t = alpha * s_initial_t + x * dt
    return s_t


@nb.jit(nopython=NO_PYTHON, cache=CACHE, nogil=NO_GIL, fastmath=FASTMATH)
def correction_down(s_t: float, s_list: list, x_elem_list: list, t: int,
                    s_upper: np.ndarray[float], s_lower: np.ndarray[float],
                    x_upper: np.ndarray[float], x_lower: np.ndarray[float],
                    alpha: float, dt: float, tol: float) -> list:
    if s_t > s_upper[t] + tol:
        aux = x_lower[:t+1] == 0
        k = list(aux)[::-1].index(False) + 1
        i = k
        x_elem_list_temp = [np.float64(x) for x in range(0)]
        while s_t > s_lower[t] + tol:
            x_elem_list_temp = x_elem_list.copy()
            s_list_temp = s_list.copy()
            del x_elem_list_temp[-i:]
            del s_list_temp[-i:]
            s_t = s_list_temp[-1]
            for j in range(i, k, -1):
                x_opt = get_x_min_alpha(s_initial_t=s_t,
                                        s_lower_t=s_lower[-j],
                                        x_upper_t=x_upper[-j],
                                        x_lower_t=x_lower[-j],
                                        alpha=alpha,
                                        dt=dt)
                s_t = get_s_t(s_initial_t=s_t, x=x_opt, alpha=alpha, dt=dt)
                x_elem_list_temp.append(x_opt)
                s_list_temp.append(s_t)
            x_opt = get_x_max_alpha(s_initial_t=s_t,
                                    s_upper_t=s_upper[t],
                                    x_upper_t=x_upper[-k],
                                    x_lower_t=x_lower[-k],
                                    alpha=alpha,
                                    dt=dt)
            s_t = get_s_t(s_initial_t=s_t, x=x_opt, alpha=alpha, dt=dt)
            x_elem_list_temp.append(x_opt)
            s_list_temp.append(s_t)
            i += 1
        x_elem_list = x_elem_list_temp
        x_elem_list += (k - 1) * [0]
    return x_elem_list


@nb.jit(nopython=NO_PYTHON, cache=CACHE, nogil=NO_GIL, fastmath=FASTMATH)
def correction_up(s_t: float, s_list: list, x_elem_list: list, t: int,
                  s_upper: np.ndarray[float], s_lower: np.ndarray[float],
                  x_upper: np.ndarray[float], x_lower: np.ndarray[float],
                  alpha: float, dt: float, tol: float) -> list:
    if s_t < s_lower[t] - tol:
        aux = x_upper[:t+1] == 0
        k = list(aux)[::-1].index(False) + 1
        i = k
        x_elem_list_temp = [np.float64(x) for x in range(0)]
        while s_t < s_lower[t] - tol:
            x_elem_list_temp = x_elem_list.copy()
            s_list_temp = s_list.copy()
            del x_elem_list_temp[-i:]
            del s_list_temp[-i:]
            s_t = s_list_temp[-1]
            for j in range(i, k, -1):
                x_opt = get_x_max_alpha(s_initial_t=s_t,
                                        s_upper_t=s_upper[-j],
                                        x_upper_t=x_upper[-j],
                                        x_lower_t=x_lower[-j],
                                        alpha=alpha,
                                        dt=dt)
                s_t = get_s_t(s_initial_t=s_t, x=x_opt, alpha=alpha, dt=dt)
                x_elem_list_temp.append(x_opt)
                s_list_temp.append(s_t)
            x_opt = get_x_min_alpha(s_initial_t=s_t,
                                    s_lower_t=s_lower[t],
                                    x_upper_t=x_upper[-k],
                                    x_lower_t=x_lower[-k],
                                    alpha=alpha,
                                    dt=dt)
            s_t = get_s_t(s_initial_t=s_t, x=x_opt, alpha=alpha, dt=dt)
            x_elem_list_temp.append(x_opt)
            s_list_temp.append(s_t)
            i += 1
        x_elem_list = x_elem_list_temp
        x_elem_list += (k - 1) * [0]
    return x_elem_list


@nb.jit(nopython=NO_PYTHON, cache=CACHE, nogil=NO_GIL, fastmath=FASTMATH)
def get_vertices(item: np.ndarray,
                 s_initial: float,
                 s_upper: np.ndarray[float], s_lower: np.ndarray[float],
                 x_upper: np.ndarray[float], x_lower: np.ndarray[float],
                 alpha: float, dt: float, d: int, tol: float
                 ) -> np.ndarray:
    s_list = [s_initial]
    x_elem_list = [np.float64(x) for x in range(0)]
    for elem, t in zip(item, range(d)):
        if elem == 1:
            x_opt = get_x_max_alpha(s_initial_t=s_list[-1],
                                    s_upper_t=s_upper[t],
                                    x_upper_t=x_upper[t],
                                    x_lower_t=x_lower[t],
                                    alpha=alpha,
                                    dt=dt)
            s_t = get_s_t(s_initial_t=s_list[-1], x=x_opt, alpha=alpha, dt=dt)
            x_elem_list.append(x_opt)
            s_list.append(s_t)
            x_elem_list = correction_down(s_t, s_list, x_elem_list, t,
                                          s_upper, s_lower,
                                          x_upper, x_lower,
                                          alpha, dt, tol)
            x_elem_list = correction_up(s_t, s_list, x_elem_list, t,
                                        s_upper, s_lower,
                                        x_upper, x_lower,
                                        alpha, dt, tol
                                        )
        elif elem == -1:
            x_opt = get_x_min_alpha(s_initial_t=s_list[-1],
                                    s_lower_t=s_lower[t],
                                    x_upper_t=x_upper[t],
                                    x_lower_t=x_lower[t],
                                    alpha=alpha,
                                    dt=dt)
            s_t = get_s_t(s_initial_t=s_list[-1], x=x_opt, alpha=alpha, dt=dt)
            x_elem_list.append(x_opt)
            s_list.append(s_t)
            x_elem_list = correction_down(s_t, s_list, x_elem_list, t,
                                          s_upper, s_lower,
                                          x_upper, x_lower,
                                          alpha, dt, tol)
            x_elem_list = correction_up(s_t, s_list, x_elem_list, t,
                                        s_upper, s_lower,
                                        x_upper, x_lower,
                                        alpha, dt, tol
                                        )
    return np.array(x_elem_list)


def pre_compile() -> None:
    get_vertices(np.array([1]),
                 0,
                 np.array([1]), np.array([0]),
                 np.array([1]), np.array([0]),
                 1, 1, 1, 1)


class IABVG_JIT:
    """Inner approximation by vertex generation algorithm using Just-in-time compilation for better performance"""

    tol = 10 ** (-15)  # correction (- 10**(-15) due to numerical issues)

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
        vertices = np.zeros((signal_vectors.signals.shape[0], self.__d))

        for index, signal in enumerate(signal_vectors.signals):
            vertices[index, :] = get_vertices(
                item=signal,
                s_initial=self.__s_initial,
                s_upper=self.__s_upper,
                s_lower=self.__s_lower,
                x_upper=self.__x_upper,
                x_lower=self.__x_lower,
                alpha=self.__alpha,
                dt=self.__dt,
                d=self.__d,
                tol=self.tol
            )
        return vertices + self.__x0


if __name__ == '__main__':
    pre_compile()
