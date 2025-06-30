import torch
import math


def lr_cosine_schedule(t: int, a_max: float, a_min: float, T_w: int, T_c: int) -> float:
    if t < T_w:
        return t / T_w * a_max
    elif T_w <= t <= T_c:
        return a_min + 0.5 * (1 + math.cos((t - T_w)/(T_c - T_w) * math.pi)) * (a_max - a_min)
    else:
        return a_min


class CosineAnnealing:
    def __init__(self, optimizer: torch.optim.Optimizer, a_max: float, a_min: float, T_w: int, T_c: int) -> None:
        self.optimizer = optimizer
        self.a_max = a_max
        self.a_min = a_min
        self.T_w = T_w
        self.T_c = T_c
    
    def step(self, t: int) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = lr_cosine_schedule(t=t, a_max=self.a_max, a_min=self.a_min, T_w=self.T_w, T_c=self.T_c)