

from typing import Callable, Tuple
import torch
from torch.nn import functional as F


def functional_xent(
    params: Tuple[torch.nn.Parameter, ...],
    model: Callable[[Tuple[torch.nn.Parameter, ...], torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    t: torch.Tensor,
    ) -> torch.Tensor:
    
    y = model(params, x)
    loss = regularization_xent(y, t, params[2])
    
    return loss

# L2 loss & regularization loss.
def regularization_xent(x: torch.Tensor, t: torch.Tensor, w: torch.Tensor) -> torch.Tensor:

    loss = (1/2) * torch.sum( torch.abs(x-t)**2 / torch.tensor(x.shape[0])) + (0.05 / 2)*torch.sum( w**2 )

    return loss

# L2 loss
def _xent(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:

    loss = (1/2) * torch.sum( torch.abs(x-t)**2 / torch.tensor(x.shape[0]))

    return loss

