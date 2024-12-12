__all__ = [
    'FrangiSigmas'
]

from itertools import product
import torch
from torch.utils.data import Dataset


class FrangiSigmas(Dataset):
    """
    A PyTorch Dataset that generates sigma values for the Frangi filter.

    This dataset calculates a set of sigma values in the format
    (lower_bound, upper_bound, sigma_step) expected by the
    Frangi function of skimage [`skimage.filters.frangi`]

    Attributes
    ----------
    sigmas : list of tuples
        A list containing tuples of (lower_bound, upper_bound, sigma_step).
    """
    def __init__(
        self,
        min_sigma: float = 0.25,
        max_sigma: float = 5.0,
        sigma_bounds_steps: float = 0.25,
        n_sigma_steps: int = 5
    ):
        """
        Initialize the FrangiSigmas dataset.

        Parameters
        ----------
        min_sigma : float, optional
            The minimum sigma value, by default 0.25.
        max_sigma : float, optional
            The maximum sigma value, by default 5.0.
        sigma_bounds_steps : float, optional
            The step size between sigma bounds when making the initial list
            before the cartesian product, by default 0.25.
        n_sigma_steps : int, optional
            Number of incremental sigma steps between each pair of bounds for
            the skimage frangi filter, by default 5.
        """
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        sigma_bounds_steps = sigma_bounds_steps
        self.n_sigma_steps = n_sigma_steps
        self.sigmas = self._make_sigmas()

    def _make_sigma_bounds(self) -> tuple:
        """
        Construct all possible pairs of lower and upper bounds for sigma.

        This function calculates the Cartesian product of a range of sigma
        values, then removes all ranges such that the lower bound is greater than
        the upper bound.
        """

        # Generate a list of sigmas values with specified step size
        sigmas = torch.arange(self.min_sigma, self.max_sigma)
        sigma_lower_bounds = sigmas[:-1]
        sigma_upper_bounds = sigmas[1:]

        # Compute Cartesian product
        all_bounds = list(product(sigma_lower_bounds, sigma_upper_bounds))

        # Remove pairs where the lower bound is larger than the upper bound
        valid_bounds = [
            sigma_bound for sigma_bound in all_bounds 
            if sigma_bound[0] <= sigma_bound[1]
        ]

        return valid_bounds

    def _make_sigmas(self) -> tuple:
        """
        Generate a list of sigma configurations based on the sigma bounds.

        For each pair of (lower_bound, upper_bound), this method divides the
        interval into `n_sigma_steps` and computes incremental sigma steps.
        Each configuration is stored as a tuple containing (lower_bound,
        upper_bound, sigma_step). Which is the proper format as expected by
        `skimage.filters.frangi`

        Returns
        -------
        list of tuple of float
            A list of (lower_bound, upper_bound, sigma_step) tuples.
        """
        # Get all valid sigma bounds
        sigma_bounds = self._make_sigma_bounds()
        # Initialize sigmas list to which the configuration will be appended
        sigmas = []

        # Compute each configuration and add to sigmas list
        for bounds in sigma_bounds:
            # Extract upper and lower bounds for efficiency
            lower_sigma_bound = bounds[0]
            upper_sigma_bound = bounds[1]
            for n in range(1, self.n_sigma_steps):
                sigma_step = (upper_sigma_bound - lower_sigma_bound) / n
                sigmas.append(
                    (lower_sigma_bound, upper_sigma_bound, sigma_step)
                )
        return sigmas

    def __len__(self) -> int:
        """
        Return the total number of sigma configurations in the dataset.

        Returns
        -------
        int
            Total number of sigma configurations.
        """
        return len(self.sigmas)

    def __getitem__(self, idx: int) -> tuple:
        """
        Retrieve the sigma configuration at the specified index.

        Parameters
        ----------
        idx : int
            The index of the desired sigma configuration.

        Returns
        -------
        tuple of float
            A tuple containing (lower_bound, upper_bound, sigma_step).
        """
        return self.sigmas[idx]
