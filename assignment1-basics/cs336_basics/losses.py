import torch
import einx

def cross_entropy_loss(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    max_x = torch.max(x, dim=-1).values # This is a bug with PyTorch
    denom = einx.sum("... [d]", (x - max_x.unsqueeze(-1)).exp())
    output = torch.log(torch.clamp_min(denom, eps)) + max_x - einx.get_at("... [l], ... -> ...", x, y)
    return einx.mean("[...]", output)

def perplexity_loss(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return cross_entropy_loss(x, y, eps).exp()
