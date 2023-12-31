r"""Implemetation of Total Variation metric, based on article
 remi.flamary.com/demos/proxtv.html and www.wikiwand.com/en/Total_variation_denoising
"""

import torch
from torch.nn.modules.loss import _Loss
from piq.utils import _validate_input, _reduce


def total_variation(x: torch.Tensor, reduction: str = 'mean', norm_type: str = 'l2') -> torch.Tensor:
    r"""Compute Total Variation metric

    Args:
        x: Tensor. Shape :math:`(N, C, H, W)`.
        reduction: Specifies the reduction type:
            ``'none'`` | ``'mean'`` | ``'sum'``. Default:``'mean'``
        norm_type: ``'l1'`` | ``'l2'`` | ``'l2_squared'``,
            defines which type of norm to implement, isotropic  or anisotropic.

    Returns:
        Total variation of a given tensor

    References:
        https://www.wikiwand.com/en/Total_variation_denoising

        https://remi.flamary.com/demos/proxtv.html
    """
    _validate_input([x, ], dim_range=(4, 4), data_range=(0, -1))

    if norm_type == 'l1':
        w_variance = torch.sum(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]), dim=[1, 2, 3])
        h_variance = torch.sum(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]), dim=[1, 2, 3])
        score = (h_variance + w_variance)
    elif norm_type == 'l2':
        w_variance = torch.sum(torch.pow(x[:, :, :, 1:] - x[:, :, :, :-1], 2), dim=[1, 2, 3])
        h_variance = torch.sum(torch.pow(x[:, :, 1:, :] - x[:, :, :-1, :], 2), dim=[1, 2, 3])
        score = torch.sqrt(h_variance + w_variance)
    elif norm_type == 'l2_squared':
        w_variance = torch.sum(torch.pow(x[:, :, :, 1:] - x[:, :, :, :-1], 2), dim=[1, 2, 3])
        h_variance = torch.sum(torch.pow(x[:, :, 1:, :] - x[:, :, :-1, :], 2), dim=[1, 2, 3])
        score = (h_variance + w_variance)
    else:
        raise ValueError("Incorrect norm type, should be one of {'l1', 'l2', 'l2_squared'}")

    return _reduce(score, reduction)


class TVLoss(_Loss):
    r"""Creates a criterion that measures the total variation of the
    the given input :math:`x`.


    If :attr:`norm_type` set to ``'l2'`` the loss can be described as:

    .. math::
        TV(x) = \sum_{N}\sqrt{\sum_{H, W, C}(|x_{:, :, i+1, j} - x_{:, :, i, j}|^2 +
        |x_{:, :, i, j+1} - x_{:, :, i, j}|^2)}

    Else if :attr:`norm_type` set to ``'l1'``:

    .. math::
        TV(x) = \sum_{N}\sum_{H, W, C}(|x_{:, :, i+1, j} - x_{:, :, i, j}| +
        |x_{:, :, i, j+1} - x_{:, :, i, j}|)

    where :math:`N` is the batch size, `C` is the channel size.

    Args:
        norm_type: one of ``'l1'`` | ``'l2'`` | ``'l2_squared'``
        reduction: Specifies the reduction type:
            ``'none'`` | ``'mean'`` | ``'sum'``. Default:``'mean'``

    Examples:

        >>> loss = TVLoss()
        >>> x = torch.rand(3, 3, 256, 256, requires_grad=True)
        >>> output = loss(x)
        >>> output.backward()

    References:
        https://www.wikiwand.com/en/Total_variation_denoising

        https://remi.flamary.com/demos/proxtv.html
    """
    def __init__(self, norm_type: str = 'l2', reduction: str = 'mean'):
        super().__init__()

        self.norm_type = norm_type
        self.reduction = reduction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Computation of Total Variation (TV) index as a loss function.

        Args:
            x: An input tensor. Shape :math:`(N, C, H, W)`.

        Returns:
            Value of TV loss to be minimized.
        """
        score = total_variation(x, reduction=self.reduction, norm_type=self.norm_type)
        return score
