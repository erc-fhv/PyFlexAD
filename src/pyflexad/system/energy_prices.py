import os

import numpy as np
import pandas as pd


class EnergyPrices:

    @staticmethod
    def from_file(path_da: str, n_cost_vectors: int, n_time_periods: int) -> np.ndarray:
        """
        Load energy prices from a file and return a subset of the prices matrix
        based on the provided number of cost vectors and time periods.

        Parameters
        ----------
        path_da: str
            The path to the file containing the energy prices.
        n_cost_vectors: int
            The number of cost vectors to extract from the prices' matrix.
        n_time_periods: int
            The number of time periods to extract from the prices' matrix.

        Returns
        -------
        np.ndarray
            The subset of the energy prices matrix with shape (n_cost_vectors, n_time_periods).
        """
        path_da = os.path.abspath(path_da)

        """get the prices"""
        price_file = os.path.join(path_da, "da_df.pickle")
        df_prices = pd.read_pickle(price_file)
        prices = df_prices.values
        prices = prices.reshape(int(len(prices) / 96), 96) / 1000  # convert to EUR/kWh

        if n_cost_vectors > prices.shape[0]:
            raise ValueError(f"n_cost_vectors ({n_cost_vectors}) > prices.shape[0] ({prices.shape[0]})")
        if n_time_periods > prices.shape[1]:
            raise ValueError(f"n_time_periods ({n_time_periods}) > prices.shape[1] ({prices.shape[1]})")

        return prices[:n_cost_vectors, :n_time_periods]

    @staticmethod
    def from_random(min_energy_price: float, max_energy_price: float, n_cost_vectors: int,
                    n_time_periods: int) -> np.ndarray:
        """
        Generate a random energy price matrix.

        Parameters
        ----------
        min_energy_price: float
            The minimum energy price.
        max_energy_price: float
            The maximum energy price.
        n_cost_vectors: int
            The number of cost vectors.
        n_time_periods: int
            The number of time intervals.

        Returns
        -------
        np.ndarray: The random energy price matrix with shape (n_cost_vectors, n_time_periods).
        """
        return np.random.uniform(low=min_energy_price, high=max_energy_price, size=(n_cost_vectors, n_time_periods))
