import torch
import einx
import math

from typing import Optional, Callable


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3)-> None:
        assert lr > 0, ValueError(f"Invalid learning rate: {lr}")
        super().__init__(params, {"lr": lr})
    
    def step(self, closure: Optional[Callable] = None) -> None:
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 0)
                grad = p.grad.data
                p.data -= lr/ math.sqrt(t + 1) * grad
                state["t"] = t + 1
        return loss


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), weight_decay=0, eps=1e-9) -> None:
        assert lr > 0, ValueError(f"Invalid learning rate: {lr}")
        self.eps = eps
        super().__init__(
            params,
            {
                "lr": lr,
                "beta1": betas[0],
                "beta2": betas[1],
                "decay": weight_decay,
            }
        )
        # Initialize momentums to zero
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["m"] = torch.zeros_like(p)
                state["v"] = torch.zeros_like(p)
        
    def step(self, closure: Optional[Callable] = None) -> None:
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            decay = group["decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 1)
                m = state.get("m")
                v = state.get("v")
                grad = p.grad.data
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * grad.pow(2)
                # import pdb; pdb.set_trace()
                lr_t = lr * math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)
                p.data -= lr_t / (torch.sqrt(v) + self.eps) * m
                p.data -= lr * decay * p.data
                state["t"] = t + 1
                state["m"] = m
                state["v"] = v
        return loss
