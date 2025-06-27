import torch
import einx
import math

def softmax(x: torch.Tensor) -> torch.Tensor:
    max_val = torch.max(x, dim=-1, keepdim=True).values # This is a bug in pytorch
    exp_x = (x - max_val).exp()
    return exp_x / torch.sum(exp_x, dim=-1, keepdim=True)

def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None,
) -> torch.Tensor:
    innerproduct = einx.dot("b ... q d, b ... k d -> b ... q k", query, key) / math.sqrt(key.shape[-1])
    masked_innerproduct = torch.where(mask, innerproduct, float("-inf")) if mask is not None else innerproduct
    return einx.dot("b ... l v, b ... v d -> b ... l d", softmax(masked_innerproduct), value)

def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)

def negative_log_likelihood(x: torch.Tensor, eps=1e-18) -> torch.Tensor:
    return einx.mean("... -> ", -torch.log(torch.clamp_min(x, eps)))