__all__ = [
    "SinusoidalAttenuator"
]

import torch
from torch import nn


class SinusoidalAttenuator(nn.Module):
    """
    A PyTorch module for generating a sinusoidal patch attenuator
    in n dimensions.

    Parameters
    ----------
    size : int
        Size of each dimension for attenuator. All dimensions are assumed to
        have equal sides.
    dimensions : int
        Number of dimensions for attenuator.
    """
    def __init__(self, size: int = 128, dimensions: int = 3):
        super().__init__()
        self.size = size
        self.dimensions = dimensions

    def forward(self):
        """
        Forward pass to generate the sinusoidal patch attenuator.

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
# attenuator = SinusoidalAttenuator(size=128, dimensions=3)()
# plt.imshow(attenuator[64])
