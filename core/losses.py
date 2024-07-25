__all__ = [
    '_dot',
    '_make_activation',
    'Loss',
    'DiceLoss'
]

from torch import nn
import torch
import inspect
from . import utils


def _dot(x, y):
    """Dot product along the last dimension"""
    return x.unsqueeze(-2).matmul(y.unsqueeze(-1)).squeeze(-1).squeeze(-1)


def _make_activation(activation):
    if isinstance(activation, str):
        activation = getattr(nn, activation)
    activation = (activation() if inspect.isclass(activation)
                  else activation if callable(activation)
                  else None)
    return activation


class Loss(nn.Module):
    """Base class for losses"""

    def __init__(self, reduction='mean'):
        """
        Parameters
        ----------
        reduction : {'mean', 'sum'} or callable
            Reduction to apply across batch elements
        """
        super().__init__()
        self.reduction = reduction

    def reduce(self, x):
        if not self.reduction:
            return x
        if isinstance(self.reduction, str):
            if self.reduction.lower() == 'mean':
                return x.mean()
            if self.reduction.lower() == 'sum':
                return x.sum()
            raise ValueError(f'Unknown reduction "{self.reduction}"')
        if callable(self.reduction):
            return self.reduction(x)
        raise ValueError(f'Don\'t know what to do with reduction: '
                         f'{self.reduction}')


class DiceLoss(Loss):
    r"""Soft Dice Loss

    By default, each class is weighted identically.
    The `weighted` mode allows classes to be weighted by frequency.

    References
    ----------
    ..  "V-Net: Fully convolutional neural networks for volumetric
         medical image segmentation"
        Milletari, Navab and Ahmadi
        3DV (2016)
        https://arxiv.org/abs/1606.04797
    ..  "Generalised dice overlap as a deep learning loss function for
         highly unbalanced segmentations"
        Sudre, Li, Vercauteren, Ourselin and Cardoso
        DLMIA (2017)
        https://arxiv.org/abs/1707.03237
    ..  "The Dice loss in the context of missing or empty labels:
         introducing $\Phi$ and $\epsilon$"
        Tilborghs, Bertels, Robben, Vandermeulen and Maes
        MICCAI (2022)
        https://arxiv.org/abs/2207.09521
    """

    def __init__(self, square=True, weighted=False, labels=None,
                 eps=None, reduction='mean', activation=None):
        """

        Parameters
        ----------
        square : bool, default=True
            Square the denominator in SoftDice.
        weighted : bool or list[float], default=False
            If True, weight the Dice of each class by its frequency in the
            reference. If a list, use these weights for each class.
        labels : list[int], default=range(nb_class)
            Label corresponding to each one-hot class. Only used if the
            reference is an integer label map.
        eps : float or list[float], default=1/K
            Stabilization of the Dice loss.
            Optimally, should be equal to each class' expected frequency
            across the whole dataset. See Tilborghs et al.
        reduction : {'mean', 'sum', None} or callable, default='mean'
            Type of reduction to apply across minibatch elements.
        activation : nn.Module or str
            Activation to apply to the prediction before computing the loss
        """
        super().__init__(reduction)
        self.square = square
        self.weighted = weighted
        self.labels = labels
        self.eps = eps
        self.activation = _make_activation(activation)

    def forward_onehot(self, pred, ref, mask, weights, eps):

        nb_classes = pred.shape[1]
        if ref.shape[1] != nb_classes:
            raise ValueError(f'Number of classes not consistent. '
                             f'Expected {nb_classes} but got {ref.shape[1]}.')

        ref = ref.to(pred)
        if mask is not None:
            pred = pred * mask
            ref = ref * mask
        pred = pred.reshape([*pred.shape[:2], -1])       # [B, C, N]
        ref = ref.reshape([*ref.shape[:2], -1])          # [B, C, N]

        # Compute SoftDice
        inter = _dot(pred, ref)                          # [B, C]
        if self.square:
            pred = pred.square()
            ref = ref.square()
        pred = pred.sum(-1)                              # [B, C]
        ref = ref.sum(-1)                                # [B, C]
        union = pred + ref
        loss = (2 * inter + eps) / (union + eps)

        # Simple or weighted average
        if weights is not False:
            if weights is True:
                # weights = ref // ref.sum(dim=1, keepdim=True)
                weights = ref / ref.sum(dim=1, keepdim=True)
            loss = loss * weights
            loss = loss.sum(-1)
        else:
            loss = loss.mean(-1)

        # Minibatch reduction
        loss = 1 - loss
        return self.reduce(loss)

    def forward_labels(self, pred, ref, mask, weights, eps):

        nb_classes = pred.shape[1]
        labels = self.labels or list(range(nb_classes))

        loss = 0
        sumweights = 0
        for index, label in enumerate(labels):
            if label is None:
                continue
            pred1 = pred[:, index]
            eps1 = eps[index]
            ref1 = (ref == label).squeeze(1)
            if mask is not None:
                pred1 = pred1 * mask
                ref1 = ref1 * mask

            pred1 = pred1.reshape([len(pred1), -1])           # [B, N]
            ref1 = ref1.reshape([len(ref1), -1])              # [B, N]

            # Compute SoftDice
            inter = (pred1 * ref1).sum(-1)                    # [B]
            if self.square:
                pred1 = pred1.square()
            pred1 = pred1.sum(-1)                             # [B]
            ref1 = ref1.sum(-1)                               # [B]
            union = pred1 + ref1
            loss1 = (2 * inter + eps1) / (union + eps1)

            # Simple or weighted average
            if weights is not False:
                if weights is True:
                    weight1 = ref1
                else:
                    weight1 = float(weights[index])
                loss1 = loss1 * weight1
                sumweights += weight1
            else:
                sumweights += 1
            loss += loss1

        # Minibatch reduction
        loss = loss / sumweights
        loss = 1 - loss
        return self.reduce(loss)

    def forward(self, pred, ref, mask=None):
        """

        Parameters
        ----------
        pred : (batch, nb_class, *spatial) tensor
            Predicted classes.
        ref : (batch, nb_class|1, *spatial) tensor
            Reference classes (or their expectation).
        mask : (batch, 1, *spatial) tensor, optional
            Loss mask

        Returns
        -------
        loss : scalar or (batch,) tensor
            The output shape depends on the type of reduction used.
            If 'mean' or 'sum', this function returns a scalar tensor.

        """
        if self.activation:
            pred = self.activation(pred)

        nb_classes = pred.shape[1]
        backend = dict(dtype=pred.dtype, device=pred.device)
        nvox = pred.shape[2:].numel()

        eps = self.eps or 1/nb_classes
        eps = utils.make_vector(eps, nb_classes, **backend)
        eps = eps * nvox

        # prepare weights
        weighted = self.weighted
        if not torch.is_tensor(weighted) and not weighted:
            weighted = False
        if not isinstance(weighted, bool):
            weighted = utils.make_vector(weighted, nb_classes, **backend)

        if ref.dtype.is_floating_point:
            return self.forward_onehot(pred, ref, mask, weighted, eps)
        else:
            return self.forward_labels(pred, ref, mask, weighted, eps)
