import logging
import numpy as np


class UPR:

    @staticmethod
    def __calc(approx: float, best: float, worst: float, decimals: int = 6) -> float:
        numerator = np.round(approx - best, decimals=decimals)
        denominator = np.round(worst - best, decimals=decimals)

        if numerator == 0:
            upr = 0.0
        elif denominator == 0:
            upr = float("inf")
        else:
            upr = float(100 * numerator / denominator)

        if upr < 0:
            logging.warning("UPR cannot be negative")
            # raise ValueError("UPR cannot be negative")
        return upr

    def __calc_peak_power(self, power: np.ndarray) -> float:
        peak_power = np.linalg.norm(power + self._agg_power_demand, np.inf)
        return peak_power

    def _calc_power_upr(self, power_approx: np.ndarray, power_best: np.ndarray,
                        power_worst: np.ndarray = None) -> float:
        approx = self.__calc_peak_power(power_approx)
        best = self.__calc_peak_power(power_best)

        if power_worst is None:
            worst = self.__calc_peak_power(np.zeros_like(power_best))
        else:
            worst = self.__calc_peak_power(power_worst)

        upr = self.__calc(approx=approx, best=best, worst=worst)

        logging.info(f"Peak Power UPR: {upr:.2f} %")
        return upr

    def __calc_energy_costs(self, power: np.ndarray) -> float:
        total_energy_costs = self._energy_prices @ (power + self._agg_power_demand)
        return total_energy_costs

    def _calc_cost_upr(self, power_approx: np.ndarray, power_best: np.ndarray,
                       power_worst: np.ndarray = None) -> float:

        approx = self.__calc_energy_costs(power_approx)
        best = self.__calc_energy_costs(power_best)

        if power_worst is None:
            worst = self.__calc_energy_costs(np.zeros_like(power_best))
        else:
            worst = self.__calc_energy_costs(power_worst)

        upr = self.__calc(approx=approx, best=best, worst=worst)

        logging.info(f"Energy cost UPR: {upr:.2f} %")
        return upr
