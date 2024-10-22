__all__ = [
    "ExponentialAttenuator",
    "GaussianAttenuator"
    "SinusoidalAttenuator",
]

import torch
from torch import nn


class SinusoidalAttenuator(nn.Module):
    """
    A PyTorch module for generating a sinusoidal patch attenuator
    in n dimensions.
    """
    def __init__(self, size: int = 128, dimensions: int = 3):
        super().__init__()
        self.size = size
        self.dimensions = dimensions

    def forward(self):
        """
        Forward pass to generate the sinusoidal patch attenuator.

        Parameters
        ----------
        size : int
            Size of each dimension for attenuator. All dimensions are assumed to
            have equal sides.
        dimensions : int
            Number of dimensions for attenuator.

        Returns
        -------
        torch.Tensor
            The n-dimensional sinusoidal attenuator.
        """
        return self.make_attenuator()

    def make_attenuator(self):
        """
        Generate the sinusoidal patch attenuator.

        Returns
        -------
        torch.Tensor
            The n-dimensional sinusoidal attenuator.
        """

        def stationary_points(n):
            """
            Calculate the nth stationary point of sin(x)

            Parameters
            ----------
            n : int
                Integer index of stationary point.

            Returns
            -------
            torch.Tensor
                The value in the domain of sin(x) corresponding to the nth
                stationary point.
            """
            return torch.pi * (n - 0.5)

        # Calculate lower bound of sin(x) domain at first stationary point.
        a = stationary_points(0)
        # Calculate upper bound of sin(x) domain at third stationary point.
        b = stationary_points(2)

        # Make the domain of the sin function.
        x = torch.linspace(a, b, self.size)
        # Calculate the range of the function.
        y = x.sin()
        # Normalize to [0, 1]
        y -= y.min()
        y /= y.max()

        # Make the attenuator in n dimensions
        attenuator_nd = torch.clone(y)
        for _ in range(1, self.dimensions):
            attenuator_nd = attenuator_nd.unsqueeze(-1) * y
        return attenuator_nd

# Example for 3d attenuator generation and visualization
# attenuator = SinusoidalAttenuator(size=128, dimensions=1)()
# x = torch.arange(len(attenuator))
# plt.plot(x, attenuator)


class ExponentialAttenuator(nn.Module):
    """
    A PyTorch module for generating an exponential patch attenuator in n
    dimensions.
    """
    def __init__(self, size: int = 128, dimensions: int = 3,
                 decay_rate: float = 2.0):
        super().__init__()
        self.size = size
        self.dimensions = dimensions
        self.decay_rate = decay_rate

    def forward(self):
        """
        Forward pass to generate the exponential patch attenuator.

        Parameters
        ----------
        size : int
            Size of each dimension for attenuator. All dimensions are assumed to
            have equal sides.
        dimensions : int
            Number of dimensions for attenuator.
        decay_rate : float
            The rate at which the attenuation decreases

        Returns
        -------
        torch.Tensor
            The n-dimensional exponential attenuator.
        """
        return self.make_attenuator()

    def make_attenuator(self):
        """
        Generate the sinusoidal patch attenuator.

        Returns
        -------
        torch.Tensor
            The n-dimensional sinusoidal attenuator.
        """
        # Make the domain of the sin function. (only half so we can mirror)
        x = torch.linspace(0, 1, self.size // 2)
        # Mirror
        x = torch.cat([x.flip(-1), x])
        # Apply exponential to prodiuct of rate decay and the domain
        y = torch.exp(-self.decay_rate * x)
        # Normalize!
        y -= y.min()
        y /= y.max()

        # Make the attenuator in n dimensions
        attenuator_nd = torch.clone(y)
        for _ in range(1, self.dimensions):
            attenuator_nd = attenuator_nd.unsqueeze(-1) * y
        return attenuator_nd

# Example for 3d attenuator generation and visualization
# attenuator = ExponentialAttenuator(size=128, dimensions=1)()
# x = torch.arange(len(attenuator))
# plt.plot(x, attenuator)


class GaussianAttenuator(nn.Module):
    """
    A PyTorch module for generating a Gaussian patch attenuator in n
    dimensions.
    """
    def __init__(self, size: int = 128, dimensions: int = 3, std: float = 1.0):
        """
        Initialize the Gaussian attenuator.

        Parameters
        ----------
        size : int
            Size of each dimension for attenuator. All dimensions are assumed
            to have equal sides.
        dimensions : int
            Number of dimensions for attenuator.
        std : float
            Standard deviation for the Gaussian function.

        Returns
        -------
        torch.Tensor
            The n-dimensional gaussian attenuator.
        """
        super().__init__(size, dimensions)
        self.std = std

    def make_attenuator(self) -> torch.Tensor:
        """
        Generate the Gaussian patch attenuator.

        Returns
        -------
        torch.Tensor
            The n-dimensional Gaussian attenuator.
        """
        # Create the domain from -3σ to +3σ
        x = torch.linspace(-3 * self.std, 3 * self.std, self.size)
        # Compute Gaussian.
        y = torch.exp(-0.5 * (x / self.std) ** 2)
        # Normalize to [0, 1]
        y = (y - y.min()) / (y.max() - y.min())

        # Make the attenuator in n dimensions.
        attenuator_nd = y
        for _ in range(1, self.dimensions):
            attenuator_nd = attenuator_nd.unsqueeze(-1) * y
        return attenuator_nd
