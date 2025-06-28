import numpy as np
import torch


def data_loader(
    data: np.ndarray,
    batch_size: int,
    context_length: int,
    device: torch.device = torch.device("cpu"),
) -> tuple[torch.Tensor, torch.Tensor]:
    start_range = len(data) - context_length
    data_points = np.random.randint(0, start_range, size=batch_size)
    x = np.zeros((batch_size, context_length))
    y = np.zeros((batch_size, context_length))
    for i, start in enumerate(data_points):
        x[i] = data[start:start + context_length]
        y[i] = data[start+1:start + context_length+1]
    x = torch.from_numpy(x)
    y = torch.from_numpy(y)
    if device.type == "cuda":
        x = x.pin_memory().to(device=device, non_blocking=True)
        y = y.pin_memory().to(device=device, non_blocking=True)
    else:
        x = x.to(device=device)
        y = y.to(device=device)
    return x, y