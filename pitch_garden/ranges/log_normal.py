"""A module for the F0 range estimator based on the log-normal distribution."""

import numpy as np

from .base import BaseRangeEstimator


class LogNormalRangeEstimator(BaseRangeEstimator):
    """An estimator giving [exp(mu - gamma * sigma), exp(mu + gamma * sigma)].

    The voiced F0 is assumed to follow a log-normal distribution, where mu and sigma
    are the mean and the standard deviation of log F0, respectively.

    """

    def __init__(self, gamma: float = 3.0, min_octaves: float = 0.5) -> None:
        """Initialize the estimator.

        Parameters
        ----------
        gamma : float, optional
            The factor multiplied to the standard deviation of log F0.

        min_octaves : float, optional
            The minimum width of the range in octaves.

        Raises
        ------
        ValueError
            If gamma is not positive or min_octaves is negative.

        """
        super().__init__(min_octaves)
        if gamma <= 0:
            raise ValueError("gamma must be positive.")
        self.gamma = gamma

    def estimate(self, values: np.ndarray) -> tuple[float, float]:
        """Estimate the F0 search range.

        Parameters
        ----------
        values : np.ndarray
            The voiced F0 values in Hz.

        Returns
        -------
        out : tuple[float, float]
            The lower and upper bounds in Hz.

        """
        log_values = np.log(values)
        mu, sigma = np.mean(log_values), np.std(log_values)
        margin = self.gamma * sigma
        return float(np.exp(mu - margin)), float(np.exp(mu + margin))
