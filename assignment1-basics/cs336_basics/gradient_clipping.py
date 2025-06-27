import torch
from typing import Iterable

def clip_gradients(params: Iterable[torch.nn.Parameter], max_l2: float, eps: float = 1e-6) -> None:
    grads = [param.grad for param in params if param.grad is not None]
    if not grads:
        return
    stacked_grads = torch.stack(grads)
    l2_norm = torch.norm(stacked_grads, p=2)
    if l2_norm > max_l2:
        for param in params:
            if param.grad is None:
                return
            param.grad = param.grad * max_l2 / (l2_norm + eps)
        